from core.utils import AverageMeter, process_data_item, run_model_loss, calculate_accuracy,batch_augment
import os
import time
import torch
from tqdm import tqdm

# ✅ 전역 변수 추가
best_val_acc = 0.0
best_val_loss = float("inf")
best_macro_f1 = 0.0   # ✅ Macro-F1 베스트 트래킹(선택)

def val_epoch(epoch, data_loader, model, criterion, opt, writer, optimizer):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    global best_val_acc, best_val_loss, best_macro_f1   # ← 함수 안에서 갱신 가능하도록
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    accuracies1 = AverageMeter()
    accuracies2 = AverageMeter()
    # ✅ 혼동행렬 초기화 (CPU에 두는 게 안전)
    confmat = torch.zeros(opt.n_classes, opt.n_classes, dtype=torch.long)
    confmat1 = torch.zeros(opt.n_classes, opt.n_classes, dtype=torch.long)
    end_time = time.time()


    for i, data_item in tqdm(enumerate(data_loader)):
        visual, saliency_map, target, audio, visualization_item, batch_size,video_item, sal_path = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            output1, loss1, gamma1 = run_model_loss(opt, [visual, target, audio, saliency_map], model, criterion, i, print_attention=False, use_intensity=False,)
    
            gamma_row_max = torch.max(gamma1,dim=1)[0]*0.7 + torch.min(gamma1,dim=1)[0]*0.3
            gamma_row_max = gamma_row_max.unsqueeze(0).transpose(1, 0)
            gamma_thre = gamma_row_max.expand(gamma1.shape)
            high_index = gamma1 < gamma_thre
            # output2,loss2,gamma2=output1,loss1,gamma1
            visual_erase1, sal_erase1 = batch_augment(video_item, high_index, opt, visual, saliency_map, sal_path)
            output2, loss2, gamma2 = run_model_loss(opt, [visual_erase1, target, audio, sal_erase1], model, criterion, i,print_attention=False, use_intensity=False,)
        output=(output1+output2)/2.
        loss=loss1/2.+loss2/2.
        acc = calculate_accuracy(output, target) #앙상블 정확도
        acc1=calculate_accuracy(output1,target) #논문 기준 정확도
        acc2=calculate_accuracy(output2,target)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        accuracies1.update(acc1, batch_size)
        accuracies2.update(acc2, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # ✅ Macro-F1용 예측 누적 (앙상블 출력 기준)
        with torch.no_grad():
            pred = output.argmax(dim=1).detach().cpu()
            # output1, 즉 CTEN 논문 기준 원본 forward
            pred1 = output1.argmax(dim=1).detach().cpu()
            tgt  = target.detach().cpu()
            # confmat[true, pred]++
            for t_i, p_i in zip(tgt.view(-1), pred.view(-1)):
                confmat[t_i.long(), p_i.long()] += 1
            for t_i, p_i in zip(tgt.view(-1), pred1.view(-1)):
                confmat1[t_i.long(), p_i.long()] += 1

    Acc = max(accuracies.avg,accuracies1.avg,accuracies2.avg)
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc', accuracies.avg, epoch)
    writer.add_scalar('val/acc1', accuracies1.avg, epoch)    # ✅ CTEN 기준(xo)

    # ✅ Macro-F1 계산
    TP = torch.diag(confmat).to(torch.float32)
    FP = confmat.sum(0).to(torch.float32) - TP   # 열 합 - TP
    FN = confmat.sum(1).to(torch.float32) - TP   # 행 합 - TP
    prec = TP / torch.clamp(TP + FP, min=1.0)
    reca = TP / torch.clamp(TP + FN, min=1.0)
    f1_per_class = (2 * prec * reca) / torch.clamp(prec + reca, min=1e-12)
    macro_f1 = f1_per_class.mean().item()
    writer.add_scalar('val/macro_f1', macro_f1, epoch)

    # output1 기준 macro-F1
    TP1 = torch.diag(confmat1).to(torch.float32)
    FP1 = confmat1.sum(0).to(torch.float32) - TP1
    FN1 = confmat1.sum(1).to(torch.float32) - TP1
    prec1 = TP1 / torch.clamp(TP1 + FP1, min=1.0)
    reca1 = TP1 / torch.clamp(TP1 + FN1, min=1.0)
    f1_per_class1 = (2 * prec1 * reca1) / torch.clamp(prec1 + reca1, min=1e-12)
    macro_f1_acc1 = f1_per_class1.mean().item()
    writer.add_scalar('val/macro_f1_acc1', macro_f1_acc1, epoch)

    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))
    print("Val acc1: {:.4f}".format(accuracies1.avg))
    print("Val acc2: {:.4f}".format(accuracies2.avg))
    print("Val Macro-F1: {:.4f}".format(macro_f1))
    print("Val Macro-F1 acc1/xo: {:.4f}".format(macro_f1_acc1))


    # best acc1 (논문 기준) 갱신 시 저장
    if accuracies1.avg > best_val_acc:
        best_val_acc = accuracies1.avg
        save_file_path = os.path.join(opt.ckpt_path, f'save_{epoch}_best-acc1{best_val_acc:.4f}.pth')
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
        print(f"✅ New best-acc1 model saved: {save_file_path}")

    # best loss(앙상블 loss) 갱신 시 저장
    if losses.avg < best_val_loss:
        best_val_loss = losses.avg
        save_file_path = os.path.join(opt.ckpt_path, f'save_{epoch}_best-loss{best_val_loss:.4f}.pth')
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
        print(f"✅ New best-loss model saved: {save_file_path}")

    # (선택) Macro-F1 기준 베스트 모델도 저장하고 싶다면:
    
    if macro_f1_acc1 > best_macro_f1:
        best_macro_f1 = macro_f1_acc1
        save_file_path = os.path.join(opt.ckpt_path, f'save_{epoch}_best-macroF1-acc1-{best_macro_f1:.4f}.pth')
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
        print(f"✅ New best-macroF1-acc1 model saved: {save_file_path}")

    return Acc
