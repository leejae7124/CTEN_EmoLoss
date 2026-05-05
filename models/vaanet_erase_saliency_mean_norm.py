import torch
import torch.nn as nn
import torchvision
from models.resnet import pretrained_resnet101
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value), attention


class MultiHeadAttentionOp(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttentionOp, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
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
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

class NonLocalBlock(nn.Module):
    def __init__(self, dim_in=2048, dim_out=2048, dim_inner=256):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        self.theta = nn.Linear(dim_in, dim_inner) #2048 -> 256
        self.phi = nn.Linear(dim_in, dim_inner) #2048 -> 256
        self.g = nn.Linear(dim_in, dim_inner) #2048 -> 256

        self.out = nn.Linear(dim_inner, dim_out) #256 -> 2048
        self.bn = nn.BatchNorm1d(dim_out)
        self.alpha=nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        residual = x

        batch_size,seq = x.shape[:2]
        x=x.view(batch_size*seq,-1) #x의 shape를 [B*seq, D]로 변경

        theta = self.theta(x) #더 작은 공간으로 선형 투영
        phi = self.phi(x) #더 작은 공간으로 선형 투영
        g = self.g(x) #더 작은 공간으로 선형 투영

        #shape의 1, 2번 순서를 바꿈.
        theta,phi,g=theta.view(batch_size,seq,-1).transpose(1,2).contiguous(),phi.view(batch_size,seq,-1).transpose(1,2).contiguous(),g.view(batch_size,seq,-1).transpose(1,2).contiguous()

        #bmm(batch matrix multiplication): 3차원 텐서 간의 행렬 곱셈을 수행하는 함수
        #theta(query)와 phi(key)의 곱
        theta_phi = torch.bmm(theta.transpose(1, 2), phi)  # (8, 16, 784) * (8, 1024, 784) => (8, 784, 784)

        theta_phi_sc = theta_phi * (self.dim_inner ** -.5) #스케일링?
        p = F.softmax(theta_phi_sc, dim=-1) #마지막 축(j)에 대해 softmax

        t = torch.bmm(g, p.transpose(1, 2)) #g(value)를 어텐션으로 집계
        t = t.transpose(1,2).contiguous().view(batch_size*seq,-1)

        out = self.out(t) #원래 차원으로 돌린다.
        out = self.bn(out)
        out=out.view(batch_size,seq,-1)

        out = out + self.alpha*residual
        return out


class VAANetErase_mean(nn.Module):
    def __init__(self,
                 snippet_duration,
                 sample_size,
                 n_classes,
                 seq_len,
                 pretrained_resnet101_path,
                 audio_embed_size=768,
                 audio_time=100,
                 audio_n_segments=10,
                  # NEW ↓↓↓
                 saliency_level='input'):
        super(VAANetErase_mean, self).__init__()
        self.snippet_duration = snippet_duration
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.ft_begin_index = 5 #fine-tuning을 어디(stage)부터 시작할지 가리키는 인덱스. 여러 블록을 담고 있는 것을 layer라고 한다.
        
        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size
        self.saliency_level = saliency_level
        
        a_resnet = torchvision.models.resnet18(pretrained=True)
        a_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        a_avgpool = nn.AvgPool2d(kernel_size=[4, 8])
        a_modules = [a_conv1] + list(a_resnet.children())[1:-2] + [a_avgpool]
        self.a_resnet = nn.Sequential(*a_modules)
        self.a_fc = nn.Sequential(
            nn.Linear(a_resnet.fc.in_features, self.audio_embed_size),
            nn.BatchNorm1d(self.audio_embed_size),
            nn.Tanh()
        )
        
        
        self.pretrained_resnet101_path = pretrained_resnet101_path
        self.drop = nn.Dropout(p=.2)
        self._init_norm_val()
        self._init_hyperparameters()
        self._init_encoder()
        self._init_nonlocal()
        self._init_attention_subnets()

    def _init_norm_val(self):
        self.NORM_VALUE = 255.0
        self.MEAN = 100.0 / self.NORM_VALUE

    def _init_encoder(self): #resnet_backbone 분리를 위해 짠 함수. self.resnet_backbone, self.resnet_pool이 나중에 사용된다.
        resnet, _ = pretrained_resnet101(snippet_duration=self.snippet_duration,
                                         sample_size=self.sample_size,
                                         n_classes=self.n_classes,
                                         ft_begin_index=self.ft_begin_index,
                                         pretrained_resnet101_path=self.pretrained_resnet101_path)
        children = list(resnet.children())


        self.resnet_backbone = nn.Sequential(*children[:-2])   # conv stem + layer1~4 (avgpool, fc를 분리)
        self.resnet_pool     = children[-2]                    # AdaptiveAvgPool3d(1)
        for p in self.resnet_backbone.parameters(): p.requires_grad = False
        for p in self.resnet_pool.parameters():     p.requires_grad = False


        # self.resnet = nn.Sequential(*children[:-1])  # delete the last fc
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

    def _init_hyperparameters(self):
        self.hp = {
            'nc': 2048,
            'k': 512,
            'm': 16,
            'hw': 4
        }

    def _init_attention_subnets(self):

        self.ta_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(2048+self.audio_embed_size, 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.seq_len, self.seq_len, bias=True),
            'relu': nn.ReLU()
        })
        self.fc = nn.Linear(2048+self.audio_embed_size, self.n_classes)

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _init_nonlocal(self):
        self.nl=nn.Sequential(NonLocalBlock())#,NonLocalBlock(),NonLocalBlock())
        self.nl_a=nn.Sequential(NonLocalBlock())#,NonLocalBlock(),NonLocalBlock())
        self.v2a_attn=MultiHeadAttentionOp(in_features=2048, head_num=8)
        self.a2v_attn=MultiHeadAttentionOp(in_features=2048, head_num=8)


    def forward(self, input: torch.Tensor):
        v, a, s = input
        # input.shape=[batch, seq_len,  3, 16, 112, 112]
        v.div_(self.NORM_VALUE).sub_(self.MEAN)
        s = s.div(self.NORM_VALUE)
        batch, seq_len, nc, snippet_duration, sample_size, _ = v.size()


        if self.saliency_level == 'input':
            # saliency_map: [B, Seq, 1, D, H, W] → [Seq, B, 1, D, H, W] 📦 [Train] saliency_map shape: torch.Size([32, 12, 1, 16, 112, 112])
            # saliency_map = s.transpose(0, 1).contiguous()  # [Seq, B, 1, D, H, W]
            # print("saliency map shape(input): ", saliency_map.shape)
            # visual: [Seq, B, C, D, H, W]
            s = s.to(v.device, dtype=v.dtype)
            # print(f"[INPUT] saliency_map after transpose (Seq,B,1,D,H,W): {saliency_map.shape}")
            saliency_mask = s.expand_as(v)  # [Seq, B, C, D, H, W] saliency의 채널 차원을 visual과 동일하게 맞춤
            # print(f"[INPUT] saliency_mask_v expand_as visual: {saliency_mask.shape}")
            # Residual 방식과 유사한 soft masking (중요한 영역 강조가 아닌, 덜 중요한 영역 감쇠의 방식)
            v = 0.5 * v + (1 - 0.5) * (v * saliency_mask)


        v = v.view(batch*seq_len, nc, snippet_duration, sample_size, sample_size)
        print(f"visual flat shape:   {tuple(v.shape)}")
        with torch.no_grad():
            feat = self.resnet_backbone(v) # [B*S, 2048, T', H', W']
            print("feat shape: ", {tuple(feat.shape)})
             # --- [추가] Feature Map 레벨 Saliency 적용 로직 ---
        # if self.saliency_level == 'feature_map':
        #     # saliency_map = saliency_map.transpose(0, 1).contiguous()
        #     saliency_map_flat = s.view(batch*seq_len, 1, snippet_duration, sample_size, sample_size)
        #     print(f"saliency flat shape: {tuple(saliency_map_flat.shape)}")
        #     print(f"same (D,H,W)? {v.shape[-3:] == saliency_map_flat.shape[-3:]}")
        #     # print("saliency map shape(feature_map): ", saliency_map.shape)
        #     # print("saliency map flat shape(feature_map): ", saliency_map_flat.shape)
        #     # saliency map 크기를 feature map에 맞게 리사이즈
        #     saliency_map_flat = saliency_map_flat.to(feat.device, dtype=feat.dtype)
        #     saliency_resized = nn.functional.adaptive_avg_pool3d(saliency_map_flat, (feat.size(2), feat.size(3), feat.size(4)))
        #     print("saliency resize shape: ", {tuple(saliency_resized.shape)})
            
        #     # 리사이즈 후
        #     # print("saliency_resized:", saliency_resized.shape)  # 기대: [384, 1, T', H', W']
        #     # saliency_map shape: [B, Seq, 1, D, H, W] -> [B*Seq, 1, D, H, W]
            
        #     # 채널 차원으로 확장하여 마스크 생성
        #     saliency_mask = saliency_resized.expand_as(feat)
        #     print("saliency mask: ", {tuple(saliency_mask.shape)})
        #     # print("saliency_mask == F:", saliency_mask.shape, F.shape)  # 동일해야 OK
            
        #     # Saliency 적용
        #     feat = 0.5 * feat + (1 - 0.5) * (feat * saliency_mask)
        if self.saliency_level == 'feature_map':
            # saliency_map = saliency_map.transpose(0, 1).contiguous()
            saliency_map_flat = s.view(batch*seq_len, 1, snippet_duration, sample_size, sample_size)
            print(f"saliency flat shape: {tuple(saliency_map_flat.shape)}")
            print(f"same (D,H,W)? {v.shape[-3:] == saliency_map_flat.shape[-3:]}")
            # print("saliency map shape(feature_map): ", saliency_map.shape)
            # print("saliency map flat shape(feature_map): ", saliency_map_flat.shape)
            # saliency map 크기를 feature map에 맞게 리사이즈
            saliency_map_flat = saliency_map_flat.to(feat.device, dtype=feat.dtype)
            saliency_resized = nn.functional.adaptive_avg_pool3d(saliency_map_flat, (feat.size(2), feat.size(3), feat.size(4)))
            print("saliency resize shape: ", {tuple(saliency_resized.shape)})
            
            # 리사이즈 후
            # print("saliency_resized:", saliency_resized.shape)  # 기대: [384, 1, T', H', W']
            # saliency_map shape: [B, Seq, 1, D, H, W] -> [B*Seq, 1, D, H, W]

            S = saliency_resized.clamp_min(0)

            # 옵션 A: 시간축(T')은 유지하고, 공간(H',W')만 평균=1로 맞추기 (추천)
            S_mean = S.mean(dim=(-2, -1), keepdim=True)   # [N,1,T',1,1]
            S_norm = S / (S_mean + 1e-6)

            print("raw mean:", S_norm.mean(dim=(-2,-1)).mean().item())

            # 너무 커지는 것 방지
            S_norm = S_norm.clamp(0.0, 3.0)

            if not hasattr(self, "_printed_eff"):
                eff = 0.5 + 0.5 * S_norm
                print("[eff] min/max:", eff.min().item(), eff.max().item())
                print("[eff] mean (over H'W'):", eff.mean(dim=(-2,-1)).mean().item())
                self._printed_eff = True
            
            # 채널 차원으로 확장하여 마스크 생성
            saliency_mask = S_norm.expand_as(feat)
            print("saliency mask: ", {tuple(saliency_mask.shape)})
            # print("saliency_mask == F:", saliency_mask.shape, F.shape)  # 동일해야 OK
            
            # Saliency 적용
            feat = 0.5 * feat + (1 - 0.5) * (feat * saliency_mask)
        with torch.no_grad():                # 풀링은 그대로 동결
            v = self.resnet_pool(feat).flatten(1)          #  [512,2048,1,1,1]에서 1인 것들 제외시킴. -> [B*S, 2048], 
        v=v.view(batch,seq_len,-1)# B S D. 다시 배치, 시퀀스, 채널로 돌림
        v=self.nl(v)# B S D #intra modal
        
        bs = a.size(0)
        a = a.transpose(0, 1).contiguous()
        a = a.chunk(self.audio_n_segments, dim=0)
        a = torch.stack(a, dim=0).contiguous()
        a = a.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        a = torch.flatten(a, start_dim=0, end_dim=1)  # [B x 256 x 32]
        a = torch.unsqueeze(a, dim=1)
        a = self.a_resnet(a)
        a = torch.flatten(a, start_dim=1).contiguous()
        a = self.a_fc(a)
        a = a.view(self.audio_n_segments, bs, self.audio_embed_size).contiguous()
        a = a.permute(1, 0, 2).contiguous() # B S D
        a = self.nl_a(a)# B S D, intra modal
        
        v2a, _ = self.v2a_attn(q=a, k=v, v=v) #inter modal
        a2v, _ = self.a2v_attn(q=v, k=a, v=a) # inter modal
        v2 = v + v2a
        a2 = a + a2v

        output=torch.cat((v2,a2),dim=-1)
        output=output.transpose(1,2).contiguous()
        Ht = self.ta_net['conv'](output)
        Ht = torch.squeeze(Ht, dim=1)
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        gamma = At.view(batch, seq_len)

        output = torch.mul(output, torch.unsqueeze(At, dim=1).repeat(1, 2048+self.audio_embed_size, 1))
        output = torch.mean(output, dim=2)
        output = self.drop(output)
        output = self.fc(output)
        return output, gamma

if __name__ == '__main__':
    model=VAANetErase_mean(
        snippet_duration=16,
        sample_size=112,
        n_classes=8,
        seq_len=16,
        audio_embed_size=2048,
        audio_n_segments=16,
        pretrained_resnet101_path='/home/ubuntu/zzc/code/vsenti/VAANet-master/data/r3d101.pth'
    ).cuda()

    visual=torch.randn(32,16,3,16,112,112).cuda()
    audio=torch.randn(32,1600,128).cuda()
    saliency=torch.randn(32,16,1,16,112,112).cuda()


    output,gamma=model([visual,audio, saliency])