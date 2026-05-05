from core.utils import AverageMeter, process_data_item, calculate_accuracy
import matplotlib.pyplot as plt
import os
import time
import torch

import time

def visualize_cam_overlay(visual, cam_map, step=0, video_id="unknown",
                          save_root="cam_vis", epoch=0, batch_idx=0):
    os.makedirs(save_root, exist_ok=True)

    b = 0
    s = 0
    d = step

    frame = visual[b, s, :, d].detach().cpu()
    heat = cam_map[b, s].detach().cpu()

    frame = frame.permute(1, 2, 0).float()
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(frame.numpy())
    plt.title("Original frame")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heat.numpy(), cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(frame.numpy())
    plt.imshow(heat.numpy(), cmap="jet", alpha=0.45)
    plt.title("Overlay")
    plt.axis("off")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        save_root,
        f"{video_id}_ep{epoch:03d}_batch{batch_idx:03d}_seq{s}_frame{d:03d}_{stamp}.png"
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_saliency_effect(visual, saliency_map, step=0, video_id="unknown",
                              save_root="vis_results", epoch=0, batch_idx=0):
    seq_idx = 0
    depth_idx = step

    original_frame = visual[batch_idx, seq_idx, :, depth_idx].detach().cpu().float()
    sal_map = saliency_map[batch_idx, seq_idx, 0, depth_idx].detach().cpu().float()

    # [C, H, W] -> [H, W, C]
    original_frame = original_frame.permute(1, 2, 0)

    # 원본 프레임 정규화
    original_frame = (original_frame - original_frame.min()) / (original_frame.max() - original_frame.min() + 1e-6)

    # saliency도 보기 좋게 정규화
    sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-6)

    # overlay용 적용 이미지
    applied = original_frame * sal_map.unsqueeze(-1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original_frame.numpy())
    axs[0].set_title(f"Video: {video_id}\nOriginal Frame")

    axs[1].imshow(sal_map.numpy(), cmap='gray')
    axs[1].set_title("Saliency Map")

    axs[2].imshow(applied.numpy())
    axs[2].set_title("Applied (Visual * Saliency)")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()

    os.makedirs(save_root, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        save_root,
        f"{video_id}_ep{epoch:03d}_batch{batch_idx:03d}_frame{step:03d}_{stamp}.png"
    )
    plt.savefig(save_path)
    plt.close(fig)

# sklearn 없이 직접 macro-F1을 계산하는 함수. 클래스마다 F1을 구한 뒤 평균을 내는 방식.
@torch.no_grad()
def compute_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1_sum = 0.0
    eps = 1e-12
    for c in range(num_classes): #클래스별로 하나씩 F1 계산. 각 클래스에 대해 TP, FP, FN을 계산한다.
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        denom = (2 * tp + fp + fn)
        f1_c = (2 * tp) / (denom + eps) if denom > 0 else 0.0
        f1_sum += f1_c
    return f1_sum / max(num_classes, 1) #모든 클래스의 F1 평균 = macro-F1


#Grad-CAM heatmap 한 장에서 상위 top-k 비율에 해당하는 픽셀 영역을 감싸는 bounding box를 만드는 함수
def heatmap_to_bbox(hm, topk_ratio=0.1, min_max=1e-8):
    if hm.max() <= min_max: #CAM이 거의 다 0이면 의미 있는 영역이 없다고 보고 건너 뜀???????
        return None

    H, W = hm.shape
    flat = hm.flatten() #heat map을 1차원으로 편다.

    k = max(1, int(flat.numel() * topk_ratio)) #전체 픽셀 수의 일정 퍼센트만큼 선택함
    topk_idx = torch.topk(flat, k=k, largest=True).indices # 그 중 값이 큰 위치만 뽑음

    ys = topk_idx // W
    xs = topk_idx % W #1차원 인덱스를 2차원 좌표 값으로 복원

    if ys.numel() == 0:
        return None

    y1, y2 = ys.min().item(), ys.max().item() + 1
    x1, x2 = xs.min().item(), xs.max().item() + 1 #top-k 픽셀 전체를 감싸는 최소 bounding box 생성
    return y1, y2, x1, x2

# CAM이 중요하다고 본 영역(bbox)을 가우시안 노이즈로 덮고, 그것을 새로운 입력으로 만드는 함수
def cover_visual_with_gaussian_noise(visual, cam_map, topk_ratio=0.1):
    # visual: [B, Seq, C, D, H, W]
    # cam_map: [B, Seq, H, W]
    visual_cov = visual.clone()
    B, Seq, C, D, H, W = visual_cov.shape

    area_list = []
    area_ratio_list = []

    for b in range(B): #배치와 시퀀스에 대해 반복
        for s in range(Seq):
            bbox = heatmap_to_bbox(cam_map[b, s], topk_ratio=topk_ratio) #top k 영역에 대한 bounding box 얻음
            if bbox is None:
                continue

            y1, y2, x1, x2 = bbox
            area = (y2 - y1) * (x2 - x1)
            area_ratio = area / (H * W)
            
            area_list.append(area)
            area_ratio_list.append(area_ratio)

            region = visual_cov[b, s, :, :, y1:y2, x1:x2]
            mu = region.mean() #해당 영역의 평균
            std = region.std().clamp_min(1e-6) #해당 영역의 표준편차
            noise = torch.randn_like(region) * std + mu #같은 통계를 가진 가우시안 노이즈 생성
            visual_cov[b, s, :, :, y1:y2, x1:x2] = noise #bounding box 영역을 노이즈로 덮는다.
    
    stats = {
        "bbox_area_mean": float(sum(area_list) / len(area_list)) if area_list else 0.0,
        "bbox_area_ratio_mean": float(sum(area_ratio_list) / len(area_ratio_list)) if area_ratio_list else 0.0,
        "bbox_area_min": float(min(area_list)) if area_list else 0.0,
        "bbox_area_max": float(max(area_list)) if area_list else 0.0,
        "bbox_count": len(area_list),
        "topk_ratio": topk_ratio,
    }

    return visual_cov, stats

# bbox가 아니라 top-k pixel-wise로 선별
def heatmap_to_topk_mask(hm, topk_ratio=0.1, min_max=1e-8):
    """
    hm: [H, W]
    return: [H, W] bool mask
    """
    if hm.max() <= min_max:
        return None

    H, W = hm.shape
    flat = hm.flatten()

    k = max(1, int(flat.numel() * topk_ratio))
    topk_idx = torch.topk(flat, k=k, largest=True).indices

    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[topk_idx] = True
    mask = mask.view(H, W)
    return mask

#bbox가 아니라 pixel-wise 노이즈 덮음
def cover_visual_with_gaussian_noise_pixelwise(visual, cam_map, topk_ratio=0.1):
    """
    visual:  [B, Seq, C, D, H, W]
    cam_map: [B, Seq, H, W]
    top-k 픽셀 위치만 직접 Gaussian noise로 덮음
    """
    visual_cov = visual.clone()
    B, Seq, C, D, H, W = visual_cov.shape

    pixel_count_list = []
    pixel_ratio_list = []

    for b in range(B):
        for s in range(Seq):
            mask2d = heatmap_to_topk_mask(cam_map[b, s], topk_ratio=topk_ratio)
            if mask2d is None:
                continue

            pixel_count = mask2d.sum().item()
            pixel_ratio = pixel_count / (H * W)

            pixel_count_list.append(pixel_count)
            pixel_ratio_list.append(pixel_ratio)

            # [H, W] -> [1, 1, H, W] -> [C, D, H, W]
            mask4d = mask2d.unsqueeze(0).unsqueeze(0).expand(C, D, H, W)

            region = visual_cov[b, s]  # [C, D, H, W]

            masked_vals = region[mask4d]
            if masked_vals.numel() == 0:
                continue

            mu = masked_vals.mean()
            std = masked_vals.std().clamp_min(1e-6)

            noise = torch.randn_like(masked_vals) * std + mu
            region[mask4d] = noise

            visual_cov[b, s] = region

    stats = {
        "pixel_count_mean": float(sum(pixel_count_list) / len(pixel_count_list)) if pixel_count_list else 0.0,
        "pixel_ratio_mean": float(sum(pixel_ratio_list) / len(pixel_ratio_list)) if pixel_ratio_list else 0.0,
        "pixel_count_min": float(min(pixel_count_list)) if pixel_count_list else 0.0,
        "pixel_count_max": float(max(pixel_count_list)) if pixel_count_list else 0.0,
        "pixel_count_total": len(pixel_count_list),
        "topk_ratio": topk_ratio,
    }

    return visual_cov, stats


def val_epoch(epoch, data_loader, model, criterion, opt, writer, optimizer=None):
    print("# ---------------------------------------------------------------------- #")
    print(f"Validation at epoch {epoch}")
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    all_preds = []
    all_targets = []

    all_preds_cov = []
    all_targets_cov = []

    #bbox에 대한 변수
    # total_bbox_area = 0.0
    # total_bbox_area_ratio = 0.0
    # total_bbox_count = 0

    #픽셀에 대한 변수
    total_pixel_count = 0.0
    total_pixel_ratio = 0.0
    total_pixel_num = 0

    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        # WECL용 process_data_item은 뒤에 video_item, sal_path가 더 옴
        visual, saliency_map, target, audio, visualization_item, batch_size, _, _ = process_data_item(opt, data_item)


        data_time.update(time.time() - end_time)

        # Round 1: CAM 생성. CAM 생성을 위해 예외적으로 gradient를 켠다.
        # model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            # WECL DAC 모델이 [visual, audio, saliency_map] 입력을 받는 버전
            output, _, _, gamma, cam_map = model(
                [visual, audio, saliency_map],
                target_class=None,
                compute_gradcam=True
            )
            loss = criterion(output, target)

        acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)

        preds = torch.argmax(output, dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

        # CAM top-k bbox 영역 노이즈 덮음
        # '''
        # visual_cov, bbox_stats = cover_visual_with_gaussian_noise(
        #     visual, cam_map.detach(), topk_ratio=0.3
        # )

        # total_bbox_area += bbox_stats["bbox_area_mean"] * bbox_stats["bbox_count"]
        # total_bbox_area_ratio += bbox_stats["bbox_area_ratio_mean"] * bbox_stats["bbox_count"]
        # total_bbox_count += bbox_stats["bbox_count"]

        # '''
        # 픽셀 와이즈 노이즈 시작
        ratio = getattr(opt, "dac_topk_ratio", 0.1)

        visual_cov, pixel_stats = cover_visual_with_gaussian_noise_pixelwise(
            visual, cam_map.detach(), topk_ratio=ratio
        )

        total_pixel_count += pixel_stats["pixel_count_mean"] * pixel_stats["pixel_count_total"]
        total_pixel_ratio += pixel_stats["pixel_ratio_mean"] * pixel_stats["pixel_count_total"]
        total_pixel_num += pixel_stats["pixel_count_total"]

        if i == 0:
            video_id = visualization_item[0][0]
            visualize_saliency_effect(
                visual, saliency_map, step=5, video_id=video_id,
                epoch=epoch, batch_idx=i
            )
            visualize_cam_overlay(
                visual, cam_map, step=5, video_id=video_id,
                save_root="cam_vis", epoch=epoch, batch_idx=i
            )

        #여기까지 픽셀 와이즈

        # Round 2
        with torch.no_grad():
            output_cov, _, _, _ = model(
                [visual_cov, audio, saliency_map],
                compute_gradcam=False
            ) #노이즈로 덮은 입력으로 재추론

        preds_cov = torch.argmax(output_cov, dim=1)
        all_preds_cov.append(preds_cov.detach().cpu())
        all_targets_cov.append(target.detach().cpu())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    y_pred_cov = torch.cat(all_preds_cov, dim=0)
    y_true_cov = torch.cat(all_targets_cov, dim=0)

    num_classes = getattr(opt, "n_classes", output.size(1))

    # F1
    f1_round1 = compute_macro_f1(y_true, y_pred, num_classes)
    f1_round2 = compute_macro_f1(y_true_cov, y_pred_cov, num_classes)
    dac_f1 = f1_round1 - f1_round2

    # Acc
    acc_round1 = (y_pred == y_true).float().mean().item()
    acc_round2 = (y_pred_cov == y_true_cov).float().mean().item()
    dac_acc = acc_round1 - acc_round2

    #bbox 평균
    # mean_bbox_area = total_bbox_area / total_bbox_count if total_bbox_count > 0 else 0.0
    # mean_bbox_area_ratio = total_bbox_area_ratio / total_bbox_count if total_bbox_count > 0 else 0.0

    #pixel-wise mean
    mean_pixel_count = total_pixel_count / total_pixel_num if total_pixel_num > 0 else 0.0
    mean_pixel_ratio = total_pixel_ratio / total_pixel_num if total_pixel_num > 0 else 0.0
    
    print(f"Round1 macro-F1: {f1_round1:.4f}")
    print(f"Round2 macro-F1: {f1_round2:.4f}")
    print(f"DAC-F1: {dac_f1:.4f}")
    print(f"Round1 Acc: {acc_round1:.4f}")
    print(f"Round2 Acc: {acc_round2:.4f}")
    print(f"DAC-Acc: {dac_acc:.4f}")

    #bbox 출력문
    # print(f"Mean bbox area: {mean_bbox_area:.4f}")
    # print(f"Mean bbox area ratio: {mean_bbox_area_ratio:.4f}")

    #pixel-wise 출력문
    print(f"Mean masked pixel count: {mean_pixel_count:.4f}")
    print(f"Mean masked pixel ratio: {mean_pixel_ratio:.4f}")

    if writer is not None:
        writer.add_scalar('dac/macro_f1_round1', f1_round1, epoch)
        writer.add_scalar('dac/macro_f1_round2', f1_round2, epoch)
        writer.add_scalar('dac/dac_f1', dac_f1, epoch)

        writer.add_scalar('dac/acc_round1', acc_round1, epoch)
        writer.add_scalar('dac/acc_round2', acc_round2, epoch)
        writer.add_scalar('dac/dac_acc', dac_acc, epoch)

        writer.add_scalar('dac/loss_round1', losses.avg, epoch)

    return {
        "f1_round1": f1_round1,
        "f1_round2": f1_round2,
        "dac_f1": dac_f1,
        "acc_round1": acc_round1,
        "acc_round2": acc_round2,
        "dac_acc": dac_acc,
        "loss_round1": losses.avg,
        "pixel_count_mean": mean_pixel_count,
        "pixel_ratio_mean": mean_pixel_ratio
        # "bbox_area_mean": mean_bbox_area,
        # "bbox_area_ratio_mean": mean_bbox_area_ratio
        
    }