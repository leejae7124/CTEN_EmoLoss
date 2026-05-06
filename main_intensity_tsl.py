import os
# (선택) 장치 정렬을 PCI 순서로 고정
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# print("CVD in python =", os.environ.get("CUDA_VISIBLE_DEVICES"))
import argparse
import torch
import sys
# print("device_count =", torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print("logical", i, "-", torch.cuda.get_device_name(i))
from torch.cuda import device_count
from tensorboardX import SummaryWriter

# from opts_tsl_school import parse_opts
# from opts_tsl import parse_opts
from core.model import generate_vaaerase_intensity_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform, get_saliency_transform
from core.dataset2 import get_training_set, get_validation_set, get_test_set, get_data_loader
from transforms.temporal import TSN
from transforms.target import ClassLabel
from train_new import train_epoch
from validation import val_epoch


# def get_audio_stats(data_loader):
#     sum_val = 0.0
#     sum_sq_val = 0.0
#     count = 0

#     print("Calculating audio stats from training data...")
#     for data_item in data_loader:
#         # CTEN+Saliency dataset return:
#         # snippets, saliency_snippets, target, audios, visualization_item, video, n_frames, sal_path
#         audio_batch = data_item[3]

#         sum_val += torch.sum(audio_batch).item()
#         sum_sq_val += torch.sum(audio_batch.pow(2)).item()
#         count += audio_batch.numel()

#     mean = sum_val / count
#     std = ((sum_sq_val / count) - (mean ** 2)) ** 0.5

#     print(f"[Audio Stats] mean: {mean}")
#     print(f"[Audio Stats] std : {std}")

#     return mean, std

def load_parse_opts():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env", choices=["lab", "school"], default="lab")

    args, remaining = bootstrap.parse_known_args()

    # 나머지 인자들은 실제 opts parser로 넘기기 위해 sys.argv 재구성
    sys.argv = [sys.argv[0]] + remaining

    if args.env == "school":
        print("school")
        from opts_tsl_school import parse_opts
    else:
        print("lab")
        from opts_tsl import parse_opts

    return parse_opts

def main():
    parse_opts = load_parse_opts()
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)
    opt.saliency_level = 'feature_map'
    model, parameters = generate_vaaerase_intensity_model(opt)
    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)
    writer = SummaryWriter(logdir=opt.log_path)
    # train
    spatial_transform = get_spatial_transform(opt, 'train')
    saliency_transform = get_saliency_transform(opt, 'train', spatial_transform)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform)
    train_loader = get_data_loader(opt, training_data, shuffle=True)

    # print("dataset norm_mean:", training_data.norm_mean)
    # print("dataset norm_std :", training_data.norm_std)

    # audio_mean, audio_std = get_audio_stats(train_loader)
    # print(f"Calculated Audio Stats -> Mean: {audio_mean:.6f}, Std: {audio_std:.6f}")

    # validation
    spatial_transform = get_spatial_transform(opt, 'val')
    saliency_transform = get_saliency_transform(opt, 'val', spatial_transform)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)
    his = -1
    for i in range(1, opt.n_epochs + 1):
        train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names, writer)
        acc = val_epoch(i, val_loader, model, criterion, opt, writer, optimizer)
        his = max(his, acc)
        print('History Acc:', his)
    writer.close()

if __name__ == "__main__":
    main()
