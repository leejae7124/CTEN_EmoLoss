import torch
from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy

@torch.no_grad()
def compute_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1_sum = 0.0
    eps = 1e-12

    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()

        denom = 2 * tp + fp + fn
        f1_c = (2 * tp) / (denom + eps) if denom > 0 else 0.0
        f1_sum += f1_c

    return f1_sum / max(num_classes, 1)

def test_epoch(data_loader, model, criterion, opt):
    print("# -------------------------------------------------- #")
    print("Test model")
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    all_preds = []
    all_targets = []
    all_video_ids = []

    with torch.no_grad():
        for i, data_item in enumerate(data_loader):
            visual, saliency_map, target, audio, visualization_item, batch_size, video_item, sal_path = process_data_item(opt, data_item)

            outputs, loss, gamma = run_model(
                opt,
                [visual, target, audio, saliency_map],
                model,
                criterion,
                i,
                print_attention=False
            )

            acc = calculate_accuracy(outputs, target)

            losses.update(loss.item(), batch_size)
            accuracies.update(acc, batch_size)

            preds = outputs.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(target.detach().cpu())
            all_video_ids.extend(visualization_item)
    
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    num_classes = getattr(opt, "n_classes", outputs.size(1))
    macro_f1 = compute_macro_f1(y_true, y_pred, num_classes)

    print(f"Test loss    : {losses.avg:.4f}")
    print(f"Test acc     : {accuracies.avg:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")

    return accuracies.avg, macro_f1, y_pred.tolist(), y_true.tolist(), all_video_ids