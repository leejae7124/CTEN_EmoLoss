from opts_auc import parse_opts
# test_main.py 상단 import
import torch.nn as nn
# 시퀀스 마다, 실제 오디오

from core.model import generate_vaaerase_test_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform, get_saliency_transform
from core.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader

from transforms.temporal import TSN
from transforms.target import ClassLabel

from validation import val_epoch
from AUC_GT_TF import test_epoch
import torch

from torch.utils.data import DataLoader
from torch.cuda import device_count

from tensorboardX import SummaryWriter

import cv2
import numpy as np
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import os

# --- 라이브러리 적용을 위한 래퍼 클래스 정의 ---
# VisualStream 모델을 감싸서 pytorch-grad-cam 라이브러리와 호환되도록 함
# test_main.py
# test_main.py
class GradCAMForVideo(GradCAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._printed = False
    def get_target_width_height(self, input_tensor):
        # 다양한 케이스를 안전하게 처리:
        # [B, C, T, H, W], [C, T, H, W], [B, C, H, W], [H, W, 3] 등
        if isinstance(input_tensor, torch.Tensor):
            shape = tuple(input_tensor.shape)
        else:
            # 일부 버전에서 numpy shape나 torch.Size가 들어올 수 있음
            shape = tuple(input_tensor)

        if len(shape) >= 5:
            # [*, *, *, H, W]
            h, w = int(shape[-2]), int(shape[-1])
        elif len(shape) == 4:
            # [*, *, H, W]
            h, w = int(shape[-2]), int(shape[-1])
        elif len(shape) == 3:
            # [H, W, C] 같은 경우: 채널(C) 버리고 H, W만
            h, w = int(shape[0]), int(shape[1])
        elif len(shape) == 2:
            # 이미 (H, W)
            h, w = int(shape[0]), int(shape[1])
        else:
            raise ValueError(f"Unexpected input shape passed to CAM: {shape}")
        size = (w, h)
        if not self._printed:
            print("[CAM target_size]", size)
            self._printed = True
        # OpenCV는 (width, height)를 기대합니다!
        return size


class VisualStreamWrapper(torch.nn.Module):
    def __init__(self, model):
        super(VisualStreamWrapper, self).__init__()
        self.model = model
        self.audio_tensor_single_clip = None
        self.saliency_tensor_single_clip = None

    def set_audio(self, audio_tensor):
        self.audio_tensor_single_clip = audio_tensor
    
    def set_saliency(self, saliency_tensor):
        self.saliency_tensor_single_clip = saliency_tensor

    def forward(self, x):
        """
        x_clip: [1, C, T, H, W] (CAM에서 클립 하나씩 들어옴)
        CTEN이 기대하는 [B, seq_len, C, T, H, W]로 만들어서
        self.model([v_seq, a_full])을 호출한 뒤, logits만 리턴.
        """
        # 1) resnet에 gradient 허용 (CAM용)
        # 1) backbone 쪽 gradient 허용
        for p in self.model.resnet_backbone.parameters():
            p.requires_grad = True
        for p in self.model.resnet_pool.parameters():
            p.requires_grad = True
        
        # 2) 입력 클립 모양 정리
        if x.dim() == 4:   # [C, T, H, W]
            x = x.unsqueeze(0)  # [1, C, T, H, W]
        B = x.size(0)
        assert B == 1, "현재 래퍼는 한 번에 클립 하나만 처리한다고 가정함"

        # 3) CTEN의 seq_len 읽어서, 같은 클립을 seq_len만큼 반복, CTEN은 [B, seq_len, 3, 16, 112, 112]를 기대함

        if not hasattr(self.model, "seq_len"):
            raise RuntimeError("model.seq_len이 정의되어 있지 않습니다.")
        seq_len = int(self.model.seq_len)

        x = x.unsqueeze(1).repeat(1, seq_len, 1, 1, 1, 1)

        assert self.audio_tensor_single_clip is not None, "Call set_audio() before forward."
        assert self.saliency_tensor_single_clip is not None, "Call set_saliency() first."
        out, gamma = self.model([x, self.audio_tensor_single_clip, self.saliency_tensor_single_clip])
        return out
    
# 3D 출력을 [B, C, H, W]로 바꾸는 reshape_transform
def reshape_transform(feats):
    # feats: [B, C, T, H, W] 형태일 가능성 큼
    if feats.dim() == 5:
        # 시간축 T에 대해 평균 (또는 중앙 프레임 인덱스 선택)
        feats = feats.mean(dim=2)  # -> [B, C, H, W]
    return feats
def main():
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)

    # audio_mean, audio_std = -6.4479, 68.5871 # split 1

    # audio_mean, audio_std = -6.613560199737549, 70.09141540527344 #split2

    # audio_mean, audio_std = -6.461667060852051, 68.90550231933594 #split3
    # audio_mean, audio_std = -6.570328712463379, 69.36627197265625 #split4
    # audio_mean, audio_std = -6.5693535804748535, 69.82913208007812 #split5
    
    # opt.audio_mean = audio_mean # tensor에서 float 값으로 변환
    # opt.audio_std = audio_std
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.saliency_level = 'feature_map'
    # print(f"Calculated Audio Stats -> Mean: {opt.audio_mean:.4f}, Std: {opt.audio_std:.4f}")

    model, parameters = generate_vaaerase_test_model(opt)
    # === (추가) 모델의 기대 seq_len과 옵션을 동기화 ===
    expected_seq = None
    # ta_net['fc']가 있는 경우: in_features를 기대 시퀀스 길이로 사용
    if hasattr(model, "ta_net") and isinstance(model.ta_net, nn.ModuleDict) \
       and "fc" in model.ta_net and isinstance(model.ta_net["fc"], nn.Linear):
        expected_seq = int(model.ta_net["fc"].in_features)
    elif hasattr(model, "seq_len"):
        expected_seq = int(model.seq_len)
    elif hasattr(model, "hp") and isinstance(model.hp, dict) and "L" in model.hp:
        expected_seq = int(model.hp["L"])
    
    if expected_seq is not None and opt.seq_len != expected_seq:
        print(f"[INFO] Adjusting opt.seq_len from {opt.seq_len} to model expected {expected_seq}")
        opt.seq_len = expected_seq

    if opt.checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path, map_location='cuda:0') # GPU 0에 로드
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else: # state_dict만 직접 저장된 경우
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {opt.checkpoint_path}")
    else:
        print("Warning: No checkpoint path provided. Model will run inference on randomly initialized weights or default pretrained weights.")

    model = model.cuda() # 모델을 GPU로 이동
    model.eval() # 모델을 평가 모드로 설정 (매우 중요!)
    print("Enabling gradients for resnet backbone for CAM analysis...")
    # 1) backbone 쪽 gradient 허용
    print("Enabling gradients for resnet backbone for CAM analysis...")
    for param in model.resnet_backbone.parameters():
        param.requires_grad = True
    for param in model.resnet_pool.parameters():
        param.requires_grad = True
    for param in model.a_fc.parameters():
        param.requires_grad = True

    

    # --- 시각화 결과 저장을 위한 디렉토리 생성 ---
    cam_save_dir = os.path.join(opt.result_path, "cam_visualizations")
    os.makedirs(cam_save_dir, exist_ok=True)
    print(f"CAM visualizations will be saved to: {cam_save_dir}")


    criterion = get_loss(opt)
    criterion = criterion.cuda()
    # optimizer = get_optim(opt, parameters)

    writer = SummaryWriter(logdir=opt.log_path)

    # train

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    print(f"[CFG before sync] opt.seq_len={opt.seq_len}, opt.batch_size={opt.batch_size}")
    # expected_seq 동기화 코드 실행…
    print(f"[CFG after  sync] opt.seq_len={opt.seq_len}, opt.batch_size={opt.batch_size}")

    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()
    saliency_transform = get_saliency_transform(opt, 'test', spatial_transform)
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)

    # --- CAM 객체 생성 ---
    wrapped_model = VisualStreamWrapper(model)
    # 감정 태스크를 학습하는 conv0를 타겟으로 지정

        # test_main.py - cam 생성 직전 부분 교체
    
    def get_last_conv3d(module: nn.Module):
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv3d):
                last = m
        return last
    
    # 3D-resnet 내부의 "마지막 Conv3d"를 타겟으로
    last_conv = get_last_conv3d(wrapped_model.model.resnet_backbone)

    
    if last_conv is not None:
        # ⬇⬇⬇ 훅 등록: cam 생성 전에, 한 번만 shape 확인용으로 출력
        _printed = {"done": False}
        def debug_hook(m, inp, out):
            if not _printed["done"]:
                print(f"[HOOK] {m.__class__.__name__} output shape: {tuple(out.shape)}")
                _printed["done"] = True  # 첫 1회만 출력
    
        _ = last_conv.register_forward_hook(debug_hook)
        # ⬆⬆⬆ 여기까지가 훅
        target_layers = [last_conv]
    
        # 3D feature를 2D로 변환
        def reshape_transform_3d(feats):
            # feats: [B, C, T, H, W]일 가능성이 큼
            if feats.dim() == 5:
                feats = feats.mean(dim=2)  # T 평균 -> [B, C, H, W]
            return feats
    
        cam = GradCAMForVideo(
            model=wrapped_model,
            target_layers=target_layers,
            reshape_transform=reshape_transform_3d
        )
    else:
        print("[WARN] No Conv3d found in resnet; falling back to conv0.")
    
        # conv0 출력: [B, 512, 16] -> 4x4로 복원
        target_layers = [wrapped_model.model.conv0]
    
        def reshape_transform_conv0(t):
            h = int(t.size(2) ** 0.5)
            return t.reshape(t.size(0), t.size(1), h, h)
    
        cam = GradCAMForVideo(
            model=wrapped_model,
            target_layers=target_layers,
            reshape_transform=reshape_transform_conv0
        )

    # target_layers = [wrapped_model.model.resnet.layer4[-1]]
    # cam = GradCAMForVideo(model=wrapped_model,
    #                       target_layers=target_layers,
    #                       reshape_transform=reshape_transform)

     # --- test_epoch 실행 (CAM 객체와 저장 경로 전달) ---
    test_epoch(1, val_loader, model, criterion, opt, writer, cam, cam_save_dir, wrapped_model, silence_audio_for_vis_auc=False)

    writer.close()
    print("Evaluation and visualization finished.")


if __name__ == "__main__":
    main()

"""
python main.py --expr_name demo
"""