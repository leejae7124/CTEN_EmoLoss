# vaannet_erase.py (CTEN 코드)에서 VAANetErase 클래스만 핵심 수정 예시

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAANetErase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # ... (기존 init 그대로)
        # self._init_norm_val()
        # self._init_hyperparameters()
        # self._init_encoder()
        # self._init_nonlocal()
        # self._init_attention_subnets()

        # ✅ Grad-CAM 상태
        self._gc_enable = False
        self._gc_activ = None
        self._gc_handle = None

    def _init_encoder(self):
        resnet, _ = pretrained_resnet101(
            snippet_duration=self.snippet_duration,
            sample_size=self.sample_size,
            n_classes=self.n_classes,
            ft_begin_index=self.ft_begin_index,
            pretrained_resnet101_path=self.pretrained_resnet101_path
        )
        children = list(resnet.children())

        # ✅ (핵심) avgpool, fc 분리: backbone은 spatial feature 유지
        self.resnet_backbone = nn.Sequential(*children[:-2])   # conv stem + layer1~4
        self.resnet_pool     = children[-2]                    # AdaptiveAvgPool3d(1)

        # 기존 코드처럼 backbone freeze
        for p in self.resnet_backbone.parameters():
            p.requires_grad = False
        for p in self.resnet_pool.parameters():
            p.requires_grad = False

        # ✅ Grad-CAM hook target: backbone의 "마지막 conv가 포함된 블록 출력"
        target = self.resnet_backbone  # backbone 출력 자체에 hook
        self._gc_handle = target.register_forward_hook(self._gc_save_activation)

    def _gc_save_activation(self, module, inp, out):
        if not self._gc_enable:
            return
        # backbone이 freeze면 out이 grad 추적이 안 붙는 경우가 많아서 "새 출발점" 생성
        out_det = out.detach()
        out_det.requires_grad_(True)
        self._gc_activ = out_det
        return out_det  # ✅ 출력 교체(= downstream은 out_det 사용)

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        target_class: torch.Tensor = None,
        compute_gradcam: bool = False,
        create_graph: bool = False,   # 2차미분까지 허용하려면 True
        retain_graph: bool = True
    ):
        # --- 기존 입력형(list/tuple)도 호환하고 싶으면 ---
        if audio is None and isinstance(visual, (list, tuple)):
            visual, audio = visual

        v = visual
        a = audio

        v.div_(self.NORM_VALUE).sub_(self.MEAN)
        batch, seq_len, nc, snippet_duration, sample_size, _ = v.size()
        v = v.view(batch * seq_len, nc, snippet_duration, sample_size, sample_size)

        # ✅ hook enable
        self._gc_enable = compute_gradcam
        self._gc_activ = None

        # --- visual backbone ---
        if compute_gradcam:
            A = self.resnet_backbone(v)        # hook이 여기서 activ 저장/교체
        else:
            with torch.no_grad():
                A = self.resnet_backbone(v)

        # --- pooling → per-segment vector (기존과 동일 흐름 유지) ---
        with torch.no_grad() if not compute_gradcam else torch.enable_grad():
            v_vec = self.resnet_pool(A).flatten(1)   # [B*Seq, 2048]

        v_vec = v_vec.view(batch, seq_len, -1)       # [B, Seq, 2048]
        v_vec = self.nl(v_vec)                       # (기존) NonLocal

        # --- audio branch (기존 코드 그대로) ---
        # a -> [B, Seq, D] 만든 뒤 self.nl_a, cross-attn ...
        # (여기는 너의 원래 forward 그대로 두면 됨)

        # ... v2a/a2v, concat, temporal attention, fc ...
        output = self.fc(...)                        # [B, n_classes]
        gamma = ...                                  # [B, Seq]

        cam_map = None
        if compute_gradcam:
            if target_class is None:
                raise ValueError("Grad-CAM을 만들려면 target_class(예: GT label)를 넘겨야 합니다.")

            # score: [B] → sum scalar
            idx = torch.arange(batch, device=output.device)
            score = output[idx, target_class.long()].sum()

            # self._gc_activ: [B*Seq, C, T', H', W']
            grads = torch.autograd.grad(
                score,
                self._gc_activ,
                retain_graph=retain_graph,
                create_graph=create_graph,
                allow_unused=False
            )[0]

            # 3D Grad-CAM → 2D로 축약
            w = grads.mean(dim=(2, 3, 4), keepdim=True)                 # [B*Seq,C,1,1,1]
            cam3d = torch.relu((w * self._gc_activ).sum(dim=1))         # [B*Seq,T',H',W']
            cam2d = cam3d.mean(dim=1, keepdim=True)                     # [B*Seq,1,H',W']

            # saliency(112x112)에 맞춰 업샘플
            cam2d = F.interpolate(cam2d, size=(self.sample_size, self.sample_size),
                                  mode="bilinear", align_corners=False) # [B*Seq,1,112,112]
            cam_map = cam2d.view(batch, seq_len, 1, self.sample_size, self.sample_size)

        # ✅ CTEN 쪽은 alpha/beta가 없으니 output/gamma/cam_map만 반환(네 run_model 쪽에서 맞추면 됨)
        return output, gamma, cam_map