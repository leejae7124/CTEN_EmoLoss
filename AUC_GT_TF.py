from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy
# 오디오 사용, GT
import os
import time
import torch
import cv2
import numpy as np
# --- CAM 관련 라이브러리 추가 ---
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

import torchvision.utils as vutils

def _denorm_frame(x_chw, mean=None, std=None):
    """x_chw: [C,H,W] in model space → [0,1]로 복원(선택)"""
    if (mean is not None) and (std is not None):
        device = x_chw.device
        mean_t = torch.tensor(mean, device=device).view(-1,1,1)
        std_t  = torch.tensor(std, device=device).view(-1,1,1)
        x_chw = x_chw * std_t + mean_t
    return x_chw.clamp(0, 1)

def save_base_ins_debug(clip_5d, base_ins_5d, out_dir, tag,
                        mean=None, std=None, step=4):
    """
    clip_5d / base_ins_5d: [1, C, T, H, W]
    out_dir/tag: 저장 경로와 파일 prefix
    mean/std: (선택) 정규화 복원용. 없으면 그대로 [0,1]로 가정
    step: 모든 프레임이 많으면 T에서 step 간격으로 샘플
    """
    os.makedirs(out_dir, exist_ok=True)
    clip_5d = clip_5d.detach().cpu()
    base_5d = base_ins_5d.detach().cpu()

    _, C, T, H, W = clip_5d.shape
    t_idx = list(range(0, T, max(1, step)))

    orig_list, blur_list, diff_list = [], [], []
    for t in t_idx:
        x = clip_5d[0, :, t]        # [C,H,W]
        b = base_5d[0, :, t]        # [C,H,W]
        x = _denorm_frame(x, mean, std)
        b = _denorm_frame(b, mean, std)
        d = (x - b).abs()

        orig_list.append(x)
        blur_list.append(b)
        diff_list.append(d)

    # 각각 가로 그리드로 저장
    grid_orig = vutils.make_grid(orig_list, nrow=len(t_idx))
    grid_blur = vutils.make_grid(blur_list, nrow=len(t_idx))
    grid_diff = vutils.make_grid(diff_list, nrow=len(t_idx))

    vutils.save_image(grid_orig, os.path.join(out_dir, f"{tag}_orig.png"))
    vutils.save_image(grid_blur, os.path.join(out_dir, f"{tag}_base_ins.png"))
    vutils.save_image(grid_diff, os.path.join(out_dir, f"{tag}_absdiff.png"))

# ------------------------------------------------------------
# util 함수: 2-D CAM → (1, C, T, H, W) 마스크로 변환
# ------------------------------------------------------------
# AUC_test.py

# AUC_graph.py 상단 어딘가에 헬퍼 추가
def tbx_add_figure(writer, tag, fig, step):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(h, w, 3)              # HWC (RGB)
    img = np.transpose(img, (2, 0, 1))      # CHW
    writer.add_image(tag, img, step, dataformats='CHW')
    plt.close(fig)

def cam_to_mask(cam_2d: np.ndarray, input_tensor: torch.Tensor):
    """
    cam_2d : [H_cam, W_cam] numpy, 값 0~1
    input_tensor : 현재 클립 입력, shape [1, C, T, H_in, W_in]
    반환 : mask_5d, torch.float32, shape 동일, 값 0~1
    """
    _, _, T, H_in, W_in = input_tensor.shape
    # 1. 원본 텐서가 있는 디바이스 정보를 가져옵니다 (GPU).
    device = input_tensor.device

    # (1,1,H_cam,W_cam) → resize → (1,1,H_in,W_in)
    cam_t = torch.from_numpy(cam_2d).unsqueeze(0).unsqueeze(0).float() # [1,1,Hc,Wc]
    
    # 2. CPU에 생성된 cam_t 텐서를 GPU로 보냅니다.
    cam_t = cam_t.to(device)
    
    cam_t = F.interpolate(cam_t, size=(H_in, W_in), mode="bilinear", align_corners=False)

    # 브로드캐스트: (1,1,1,H,W) → (1,1,T,H,W)
    mask = cam_t.unsqueeze(2).repeat(1, 1, T, 1, 1)
    return mask       # 값 0~1          # 값 0~1

import torch.nn.functional as F

def strong_blur_5d(x5d: torch.Tensor, spatial_scale: float = 0.125) -> torch.Tensor:
    """
    x5d: [1, C, T, H, W]
    spatial_scale: 0.125이면 H/8, W/8로 다운샘플 후 업샘플 → 강한 블러
    """
    B, C, T, H, W = x5d.shape
    # 시간축 T는 그대로 두고, 공간(H, W)만 강하게 블러
    low = F.interpolate(
        x5d, size=(T, max(1, int(H * spatial_scale)), max(1, int(W * spatial_scale))),
        mode="trilinear", align_corners=False
    )
    blur = F.interpolate(low, size=(T, H, W), mode="trilinear", align_corners=False)
    return blur


class AUCMeter:
    """누적 평균 + 리스트 저장(원하면)"""
    def __init__(self, keep_all=False):
        self.sum = 0.0
        self.n   = 0
        self.keep = [] if keep_all else None

    def update(self, val, k=1):
        self.sum += val * k
        self.n   += k
        if self.keep is not None:
            self.keep.append(val)

    @property
    def avg(self):
        return self.sum / max(1, self.n)

def test_epoch(epoch, data_loader, model, criterion, opt, writer, cam, cam_save_dir, wrapped_model, silence_audio_for_vis_auc=True):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()
    top1_correct = 0
    total = 0

    # --- meters
    class M:
        def __init__(self): self.del_auc, self.ins_auc = AUCMeter(), AUCMeter()
    meter_all = M()     # 전체
    meter_tp  = M()     # TP만
    meter_fn  = M()     # FN만

    steps = torch.linspace(0, 1, 5, device=opt.device)   # 0~1, 5% 간격
    baseline_val = 0.392156

    # --- 시각화/평균곡선 준비
    save_root = Path(cam_save_dir) / f"auc_epoch_{epoch:04d}"
    save_root.mkdir(parents=True, exist_ok=True)
    p_vals = steps.detach().cpu().numpy()
    def zeros_like_p(): return np.zeros_like(p_vals, dtype=np.float64)

    mean_curve_sum_all = {"del": zeros_like_p(), "ins": zeros_like_p()}
    mean_curve_sum_tp  = {"del": zeros_like_p(), "ins": zeros_like_p()}
    mean_curve_sum_fn  = {"del": zeros_like_p(), "ins": zeros_like_p()}
    n_all = 0; n_tp = 0; n_fn = 0

    SAVE_PER_CLIP = getattr(opt, "save_auc_per_clip", False)

    for i, data_item in enumerate(data_loader):
        (visual, saliency_map,target,audio,visualization_item,batch_size,video_item,sal_path) = process_data_item(opt, data_item)

        with torch.no_grad():
            output, loss, _ = run_model(opt, [visual, target, audio, saliency_map], model, criterion, i)
        acc_batch = calculate_accuracy(output, target)
        print(f"[DEBUG] batch {i} acc1_like = {acc_batch:.4f}")

        # 입력 텐서 배치/시퀀스 정렬 확인
        if visual.size(0) == opt.seq_len:   # [seq_len, batch, C,T,H,W]
            seq_first = True
            bs = visual.size(1)
        else:                                # [batch, seq_len, C,T,H,W]
            seq_first = False
            bs = visual.size(0)

        for sample_idx in range(bs):
            # 현재 샘플의 시퀀스 텐서
            video_sample = visual[:, sample_idx] if seq_first else visual[sample_idx]  # [seq_len, C,T,H,W]

            # GT 클래스(one-hot/정수 모두 대응)
            if target.dim() >= 2 and target.size(-1) > 1:
                gt_class = int(target[sample_idx].argmax(dim=-1).item())
            else:
                gt_class = int(target[sample_idx].item())

            # 시퀀스 단위 정확도 집계
            pred_full = int(output.argmax(1)[sample_idx].item())
            is_tp = (pred_full == gt_class)
            top1_correct += int(is_tp)
            total += 1

            audio_sample = audio[sample_idx].unsqueeze(0)

            # CAM용 오디오 세팅
            if silence_audio_for_vis_auc:
                base_audio = torch.zeros_like(audio_sample)
                cam.model.set_audio(base_audio)
            else:
                cam.model.set_audio(audio_sample)

            # --- 시퀀스의 각 클립에 대해 GT CAM 계산
            for clip_idx in range(opt.seq_len):
                clip_tensor_5d = video_sample[clip_idx].unsqueeze(0)  # [1,C,T,H,W]
                
                if seq_first:
                    # saliency_map: [seq_len, batch, 1,T,H,W] 인 경우
                    sal_clip_5d = saliency_map[clip_idx, sample_idx]   # [1, T,H,W]
                else:
                    # saliency_map: [batch, seq_len, 1,T,H,W] 인 경우
                    sal_clip_5d = saliency_map[sample_idx, clip_idx]   # [1, T,H,W]     # [1, T, H, W]
                sal_clip_5d = sal_clip_5d.unsqueeze(0)                 # [1, 1, T, H, W]
                sal_clip_5d = sal_clip_5d.to(clip_tensor_5d.device, dtype=clip_tensor_5d.dtype)

                cam.model.set_saliency(sal_clip_5d)

                # baseline 준비
                base_del = torch.full_like(clip_tensor_5d, baseline_val)        # 삭제용 상수
                base_ins = strong_blur_5d(clip_tensor_5d, spatial_scale=0.125)  # 삽입용 강블러
                # base_ins = torch.full_like(clip_tensor_5d, baseline_val)

                # 디버그로 첫 배치/첫 샘플 몇 클립만 저장
                if i == 0 and sample_idx == 0 and clip_idx in (0, opt.seq_len//2, opt.seq_len-1):
                    # 전처리 복원을 원하면 mean/std 채워 넣기 (없으면 None, None)
                    rgb_mean = None  # 예: [0.485, 0.456, 0.406]
                    rgb_std  = None  # 예: [0.229, 0.224, 0.225]
                    save_base_ins_debug(
                        clip_tensor_5d, base_ins,
                        out_dir=os.path.join(save_root, "debug_base_ins"),
                        tag=f"s{sample_idx:03d}_c{clip_idx:03d}",
                        mean=rgb_mean, std=rgb_std, step=1  # step=1이면 해당 클립의 모든 프레임
                    )


                # CAM & mask (타깃 = GT)
                targets = [ClassifierOutputTarget(gt_class)]
                grayscale_cam_clip = cam(input_tensor=clip_tensor_5d, targets=targets)  # [1,Hc,Wc]
                grayscale_cam_2d = grayscale_cam_clip[0]
                mask_5d = cam_to_mask(grayscale_cam_2d, clip_tensor_5d)  # [1,1,T,H,W]

                # 중요도 순 인덱스 준비
                flat = mask_5d.reshape(-1)
                _, idx_sorted = torch.sort(flat, descending=True)
                N = flat.numel()
                cum_mask_flat = torch.zeros_like(flat)

                del_scores, ins_scores = [], []
                with torch.no_grad():
                    for p in steps:
                        K = int(round(p.item() * N))
                        already = int(cum_mask_flat.sum().item())
                        if K > already:
                            cum_mask_flat[idx_sorted[already:K]] = 1.0
                        del_mask = cum_mask_flat.view_as(mask_5d)

                        # deletion: 원본→baseline로 대체
                        x_del = clip_tensor_5d * (1 - del_mask) + base_del * del_mask
                        logits_del = wrapped_model(x_del)
                        prob_del = F.softmax(logits_del, dim=-1)
                        del_scores.append(prob_del[0, gt_class].item())

                        # insertion: baseline→원본으로 채움
                        x_ins = base_ins * (1 - del_mask) + clip_tensor_5d * del_mask
                        logits_ins = wrapped_model(x_ins)
                        prob_ins = F.softmax(logits_ins, dim=-1)
                        ins_scores.append(prob_ins[0, gt_class].item())

                # --- AUC 계산
                del_curve = np.asarray(del_scores, dtype=np.float64)
                ins_curve = np.asarray(ins_scores, dtype=np.float64)
                auc_del = np.trapz(del_curve, p_vals)
                auc_ins = np.trapz(ins_curve, p_vals)

                # 전체 누적
                meter_all.del_auc.update(auc_del)
                meter_all.ins_auc.update(auc_ins)
                mean_curve_sum_all["del"] += del_curve
                mean_curve_sum_all["ins"] += ins_curve
                n_all += 1

                # TP/FN 분기 누적
                if is_tp:
                    meter_tp.del_auc.update(auc_del)
                    meter_tp.ins_auc.update(auc_ins)
                    mean_curve_sum_tp["del"] += del_curve
                    mean_curve_sum_tp["ins"] += ins_curve
                    n_tp += 1
                else:
                    meter_fn.del_auc.update(auc_del)
                    meter_fn.ins_auc.update(auc_ins)
                    mean_curve_sum_fn["del"] += del_curve
                    mean_curve_sum_fn["ins"] += ins_curve
                    n_fn += 1

                # per-clip PNG (옵션)
                if SAVE_PER_CLIP:
                    grp = "tp" if is_tp else "fn"
                    fig = plt.figure(figsize=(6, 4), dpi=160)
                    plt.plot(p_vals, del_curve, label=f"Deletion (AUC={auc_del:.4f})")
                    plt.plot(p_vals, ins_curve, label=f"Insertion (AUC={auc_ins:.4f})")
                    plt.fill_between(p_vals, del_curve, alpha=0.15)
                    plt.fill_between(p_vals, ins_curve, alpha=0.15)
                    plt.title(f"[GT/{grp}] Sample {sample_idx}, Clip {clip_idx}, cls={gt_class}")
                    plt.xlabel("p (top-k ratio)"); plt.ylabel("p(class)")
                    plt.grid(True, alpha=0.3); plt.legend()
                    out_png = save_root / f"gt_{grp}_auc_sample{sample_idx:03d}_clip{clip_idx:03d}.png"
                    fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)

        if (i + 1) % 5 == 0:
            print(f"-- batch {i+1}/{len(data_loader)} "
                  f"ALL(del={meter_all.del_auc.avg:.4f}, ins={meter_all.ins_auc.avg:.4f}) "
                  f"TP(del={meter_tp.del_auc.avg:.4f}, ins={meter_tp.ins_auc.avg:.4f}) "
                  f"FN(del={meter_fn.del_auc.avg:.4f}, ins={meter_fn.ins_auc.avg:.4f})")

    # --- 에폭 요약 & TensorBoard (정확도)
    if total > 0:
        acc = top1_correct / total
        print(f"[VAL] top1 acc = {acc:.4f}")
        writer.add_scalar("val/top1_acc", acc, epoch)

    # --- 스칼라 로깅
    print(f"[VAL-AUC][ALL] deletion={meter_all.del_auc.avg:.4f}  insertion={meter_all.ins_auc.avg:.4f}  (N={n_all})")
    writer.add_scalar('val_auc_gt_all/deletion',  meter_all.del_auc.avg, epoch)
    writer.add_scalar('val_auc_gt_all/insertion', meter_all.ins_auc.avg, epoch)

    if n_tp > 0:
        print(f"[VAL-AUC][TP ] deletion={meter_tp.del_auc.avg:.4f}  insertion={meter_tp.ins_auc.avg:.4f}  (N={n_tp})")
        writer.add_scalar('val_auc_gt_tp/deletion',  meter_tp.del_auc.avg, epoch)
        writer.add_scalar('val_auc_gt_tp/insertion', meter_tp.ins_auc.avg, epoch)
    else:
        print("[VAL-AUC][TP ] no samples")

    if n_fn > 0:
        print(f"[VAL-AUC][FN ] deletion={meter_fn.del_auc.avg:.4f}  insertion={meter_fn.ins_auc.avg:.4f}  (N={n_fn})")
        writer.add_scalar('val_auc_gt_fn/deletion',  meter_fn.del_auc.avg, epoch)
        writer.add_scalar('val_auc_gt_fn/insertion', meter_fn.ins_auc.avg, epoch)
    else:
        print("[VAL-AUC][FN ] no samples")

    # --- 에폭 평균 곡선(ALL/TP/FN)
    def plot_and_save(mean_sum, N, meter, tag_prefix, fname_prefix, title_suffix):
        if N <= 0: return
        mean_del = mean_sum["del"] / N
        mean_ins = mean_sum["ins"] / N
        auc_del_curve = np.trapz(mean_del, p_vals)
        auc_ins_curve = np.trapz(mean_ins, p_vals)

        fig = plt.figure(figsize=(7, 5), dpi=160)
        plt.plot(p_vals, mean_del,
                 label=f"Deletion  ⟨AUC⟩={meter.del_auc.avg:.4f} | curveAUC={auc_del_curve:.4f}")
        plt.plot(p_vals, mean_ins,
                 label=f"Insertion ⟨AUC⟩={meter.ins_auc.avg:.4f} | curveAUC={auc_ins_curve:.4f}")
        plt.fill_between(p_vals, mean_del, alpha=0.15)
        plt.fill_between(p_vals, mean_ins, alpha=0.15)
        plt.title(f"[GT]{title_suffix} — epoch {epoch}")
        plt.xlabel("p (top-k ratio)"); plt.ylabel("p(class)")
        plt.grid(True, alpha=0.3); plt.legend()

        out_png = save_root / f"{fname_prefix}_auc_curves_epoch{epoch:04d}.png"
        fig.savefig(out_png, bbox_inches='tight')
        tbx_add_figure(writer, f'{tag_prefix}/mean_curves', fig, epoch)  # 내부에서 fig close

        np.savez(save_root / f"{fname_prefix}_auc_curves_epoch{epoch:04d}.npz",
                 p=p_vals, mean_del=mean_del, mean_ins=mean_ins,
                 mean_del_auc_scalar=meter.del_auc.avg, mean_ins_auc_scalar=meter.ins_auc.avg,
                 mean_del_curve_auc=auc_del_curve, mean_ins_curve_auc=auc_ins_curve)

    plot_and_save(mean_curve_sum_all, n_all, meter_all, 'val_auc_gt_all', 'gt_all', ' Mean Deletion/Insertion (ALL)')
    plot_and_save(mean_curve_sum_tp,  n_tp,  meter_tp,  'val_auc_gt_tp',  'gt_tp',  ' Mean Deletion/Insertion (TP)')
    plot_and_save(mean_curve_sum_fn,  n_fn,  meter_fn,  'val_auc_gt_fn',  'gt_fn',  ' Mean Deletion/Insertion (FN)')
