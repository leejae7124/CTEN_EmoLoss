import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.resnet import pretrained_resnet101


class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value), attention


class MultiHeadAttentionOp(nn.Module):
    def __init__(self, in_features, head_num, bias=True, activation=F.relu):
        super(MultiHeadAttentionOp, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(
                "`in_features`({}) should be divisible by `head_num`({})".format(
                    in_features, head_num
                )
            )
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attn = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attn

    @staticmethod
    def gen_history_mask(x):
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(
            1, seq_len, seq_len
        ).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return (
            x.reshape(batch_size, seq_len, self.head_num, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.head_num, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return (
            x.reshape(batch_size, self.head_num, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def extra_repr(self):
        return "in_features={}, head_num={}, bias={}, activation={}".format(
            self.in_features, self.head_num, self.bias, self.activation
        )


class NonLocalBlock(nn.Module):
    def __init__(self, dim_in=2048, dim_out=2048, dim_inner=256):
        super(NonLocalBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        self.theta = nn.Linear(dim_in, dim_inner)
        self.phi = nn.Linear(dim_in, dim_inner)
        self.g = nn.Linear(dim_in, dim_inner)

        self.out = nn.Linear(dim_inner, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)
        self.alpha = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        residual = x

        batch_size, seq = x.shape[:2]
        x = x.view(batch_size * seq, -1)

        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        theta = theta.view(batch_size, seq, -1).transpose(1, 2).contiguous()
        phi = phi.view(batch_size, seq, -1).transpose(1, 2).contiguous()
        g = g.view(batch_size, seq, -1).transpose(1, 2).contiguous()

        theta_phi = torch.bmm(theta.transpose(1, 2), phi)
        theta_phi_sc = theta_phi * (self.dim_inner ** -0.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.transpose(1, 2).contiguous().view(batch_size * seq, -1)

        out = self.out(t)
        out = self.bn(out)
        out = out.view(batch_size, seq, -1)

        out = out + self.alpha * residual
        return out


class VAANetEraseIntensity(nn.Module):
    """CTEN/VAANetErase with a Grad-CAM branch for saliency-intensity alignment.

    Normal forward:
        output, gamma = model([visual, audio])

    Grad-CAM forward, used only during training for intensity loss:
        output, gamma, cam_map = model(
            [visual, audio], target_class=target, compute_gradcam=True
        )

    Returned cam_map shape:
        [B, Seq, sample_size, sample_size]
    """

    def __init__(
        self,
        snippet_duration,
        sample_size,
        n_classes,
        seq_len,
        pretrained_resnet101_path,
        audio_embed_size=768,
        audio_time=100,
        audio_n_segments=10,
    ):
        super(VAANetEraseIntensity, self).__init__()
        self.snippet_duration = snippet_duration
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.ft_begin_index = 5

        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size

        a_resnet = torchvision.models.resnet18(pretrained=True)
        a_conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False
        )
        a_avgpool = nn.AvgPool2d(kernel_size=[4, 8])
        a_modules = [a_conv1] + list(a_resnet.children())[1:-2] + [a_avgpool]
        self.a_resnet = nn.Sequential(*a_modules)
        self.a_fc = nn.Sequential(
            nn.Linear(a_resnet.fc.in_features, self.audio_embed_size),
            nn.BatchNorm1d(self.audio_embed_size),
            nn.Tanh(),
        )

        self.pretrained_resnet101_path = pretrained_resnet101_path
        self.drop = nn.Dropout(p=0.2)

        self._init_norm_val()
        self._init_hyperparameters()
        self._init_encoder()
        self._init_gradcam_adapter()
        self._init_nonlocal()
        self._init_attention_subnets()

    def _init_norm_val(self):
        self.NORM_VALUE = 255.0
        self.MEAN = 100.0 / self.NORM_VALUE

    def _init_encoder(self):
        resnet, _ = pretrained_resnet101(
            snippet_duration=self.snippet_duration,
            sample_size=self.sample_size,
            n_classes=self.n_classes,
            ft_begin_index=self.ft_begin_index,
            pretrained_resnet101_path=self.pretrained_resnet101_path,
        )
        children = list(resnet.children())
        if len(children) < 2:
            raise RuntimeError("Unexpected ResNet structure: cannot split pool/fc.")

        # Original vaanet_erase.py uses children[:-1], which includes avgpool and
        # removes only fc. For Grad-CAM we need the avgpool *input*, so we split it.
        self.resnet_backbone = nn.Sequential(*children[:-2])  # conv trunk, keeps T/H/W
        self.resnet_pool = children[-2]  # usually AdaptiveAvgPool3d/AvgPool3d

        for param in self.resnet_backbone.parameters():
            param.requires_grad = False

    def _init_gradcam_adapter(self):
        # VAANet_intensity uses trainable conv0 as the CAM target after frozen
        # ResNet. CTEN did not have such a layer, so we add an identity-initialized
        # 1x1x1 adapter and attach the Grad-CAM hook here.
        self.cam_adapter = nn.Conv3d(2048, 2048, kernel_size=1, bias=True)
        self._init_identity_conv3d(self.cam_adapter)

        self._gc_enable = False
        self._gc_activ = None
        self._gc_handle = self.cam_adapter.register_forward_hook(
            self._gc_save_activation
        )

    @staticmethod
    def _init_identity_conv3d(conv: nn.Conv3d):
        if conv.kernel_size != (1, 1, 1) or conv.in_channels != conv.out_channels:
            raise ValueError("Identity initialization requires square 1x1x1 Conv3d.")
        with torch.no_grad():
            conv.weight.zero_()
            idx = torch.arange(conv.in_channels)
            conv.weight[idx, idx, 0, 0, 0] = 1.0
            if conv.bias is not None:
                conv.bias.zero_()

    def _gc_save_activation(self, module, module_input, module_output):
        if not self._gc_enable:
            return
        # cam_adapter is outside torch.no_grad(), so this tensor is graph-connected.
        self._gc_activ = module_output

    def remove_gradcam_hook(self):
        if getattr(self, "_gc_handle", None) is not None:
            self._gc_handle.remove()
            self._gc_handle = None

    def _init_hyperparameters(self):
        self.hp = {"nc": 2048, "k": 512, "m": 16, "hw": 4}

    def _init_attention_subnets(self):
        self.ta_net = nn.ModuleDict(
            {
                "conv": nn.Sequential(
                    nn.Conv1d(2048 + self.audio_embed_size, 1, 1, bias=False),
                    nn.BatchNorm1d(1),
                    nn.Tanh(),
                ),
                "fc": nn.Linear(self.seq_len, self.seq_len, bias=True),
                "relu": nn.ReLU(),
            }
        )
        self.fc = nn.Linear(2048 + self.audio_embed_size, self.n_classes)

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def _init_nonlocal(self):
        self.nl = nn.Sequential(NonLocalBlock())
        self.nl_a = nn.Sequential(NonLocalBlock())
        self.v2a_attn = MultiHeadAttentionOp(in_features=2048, head_num=8)
        self.a2v_attn = MultiHeadAttentionOp(in_features=2048, head_num=8)

    def _forward_visual(self, v):
        # input shape: [B, Seq, 3, T, H, W]
        # Avoid in-place normalization because train_new.py reuses original tensors
        # to create erased/subset views after the first forward.
        v = v.float().div(self.NORM_VALUE).sub(self.MEAN)

        batch, seq_len, nc, snippet_duration, sample_size, _ = v.size()
        v = v.view(batch * seq_len, nc, snippet_duration, sample_size, sample_size)

        with torch.no_grad():
            feat = self.resnet_backbone(v)  # [B*Seq, 2048, T', H', W']

        feat = self.cam_adapter(feat)  # Grad-CAM hook target; graph starts here.
        pooled = self.resnet_pool(feat)
        pooled = torch.flatten(pooled, start_dim=1).contiguous()  # [B*Seq, 2048]
        pooled = pooled.view(batch, seq_len, -1)
        pooled = self.nl(pooled)
        return pooled, batch, seq_len, sample_size

    def _forward_audio(self, a, batch, seq_len):
        bs = a.size(0)
        a = a.transpose(0, 1).contiguous()
        a = a.chunk(self.audio_n_segments, dim=0)
        a = torch.stack(a, dim=0).contiguous()
        a = a.transpose(1, 2).contiguous()
        a = torch.flatten(a, start_dim=0, end_dim=1)
        a = torch.unsqueeze(a, dim=1)
        a = self.a_resnet(a)
        a = torch.flatten(a, start_dim=1).contiguous()
        a = self.a_fc(a)
        a = a.view(self.audio_n_segments, bs, self.audio_embed_size).contiguous()
        a = a.permute(1, 0, 2).contiguous()  # [B, audio_n_segments, D]

        if a.size(1) != seq_len:
            raise RuntimeError(
                "audio_n_segments must match visual seq_len for the current "
                "VAANetErase fusion code. Got audio segments={} and visual seq_len={}.".format(
                    a.size(1), seq_len
                )
            )

        a = self.nl_a(a)
        return a

    def _forward_classifier(self, v, a, batch, seq_len):
        v2a, _ = self.v2a_attn(q=a, k=v, v=v)
        a2v, _ = self.a2v_attn(q=v, k=a, v=a)
        v2 = v + v2a
        a2 = a + a2v

        feat = torch.cat((v2, a2), dim=-1)  # [B, Seq, 2048+audio_embed]
        feat = feat.transpose(1, 2).contiguous()  # [B, C, Seq]

        Ht = self.ta_net["conv"](feat)
        Ht = torch.squeeze(Ht, dim=1)
        Ht = self.ta_net["fc"](Ht)
        At = self.ta_net["relu"](Ht)
        gamma = At.view(batch, seq_len)

        feat = torch.mul(
            feat,
            torch.unsqueeze(At, dim=1).repeat(1, 2048 + self.audio_embed_size, 1),
        )
        feat = torch.mean(feat, dim=2)
        feat = self.drop(feat)
        output = self.fc(feat)
        return output, gamma

    def _compute_cam_map(self, output, target_class, batch, seq_len, sample_size, create_graph=True):
        if self._gc_activ is None:
            raise RuntimeError(
                "Grad-CAM activation is None. Hook may not be attached correctly."
            )

        if target_class is None:
            target_class = output.detach().argmax(dim=1)
        target_class = target_class.view(-1).long().to(output.device)

        score = output.gather(1, target_class.view(-1, 1)).sum()
        grads = torch.autograd.grad(
            score,
            self._gc_activ,
            retain_graph=self.training,
            create_graph=create_graph,
        )[0]

        A = self._gc_activ
        if A.dim() == 5:
            # A: [B*Seq, C, T', H', W']
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)
            cam3 = torch.relu((weights * A).sum(dim=1))  # [B*Seq, T', H', W']
            cam2 = cam3.mean(dim=1)  # [B*Seq, H', W']
        elif A.dim() == 4:
            # Fallback for Conv2d-style activations: [B*Seq, C, H', W']
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam2 = torch.relu((weights * A).sum(dim=1))
        else:
            raise RuntimeError(
                "Unexpected Grad-CAM activation shape: {}".format(tuple(A.shape))
            )

        cam2_up = F.interpolate(
            cam2.unsqueeze(1),
            size=(sample_size, sample_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        cam_map = cam2_up.view(batch, seq_len, sample_size, sample_size).contiguous()
        return cam_map

    def forward(
        self,
        input: torch.Tensor,
        target_class=None,
        compute_gradcam=False,
        create_graph=True,
    ):
        v, a = input

        self._gc_enable = compute_gradcam
        self._gc_activ = None

        v, batch, seq_len, sample_size = self._forward_visual(v)
        a = self._forward_audio(a, batch=batch, seq_len=seq_len)
        output, gamma = self._forward_classifier(v, a, batch=batch, seq_len=seq_len)

        if not compute_gradcam:
            self._gc_enable = False
            return output, gamma

        cam_map = self._compute_cam_map(
            output=output,
            target_class=target_class,
            batch=batch,
            seq_len=seq_len,
            sample_size=sample_size,
            create_graph=create_graph,
        )
        self._gc_enable = False
        return output, gamma, cam_map


# Backward-compatible alias. If core/model.py imports VAANetErase from this file,
# it will still work.
VAANetErase = VAANetEraseIntensity


if __name__ == "__main__":
    model = VAANetEraseIntensity(
        snippet_duration=16,
        sample_size=112,
        n_classes=8,
        seq_len=16,
        audio_embed_size=2048,
        audio_n_segments=16,
        pretrained_resnet101_path="/path/to/r3d101.pth",
    ).cuda()

    visual = torch.randn(2, 16, 3, 16, 112, 112).cuda()
    audio = torch.randn(2, 4096, 32).cuda()  # 4096 / 16 = 256 per audio segment
    target = torch.randint(0, 8, (2,), device="cuda")

    output, gamma = model([visual, audio])
    output, gamma, cam_map = model([visual, audio], target_class=target, compute_gradcam=True)
    print(output.shape, gamma.shape, cam_map.shape)
