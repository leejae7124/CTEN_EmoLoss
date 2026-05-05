import os
# (선택) 장치 정렬을 PCI 순서로 고정
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# print("CVD in python =", os.environ.get("CUDA_VISIBLE_DEVICES"))

import torch
# print("device_count =", torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print("logical", i, "-", torch.cuda.get_device_name(i))
from torch.cuda import device_count
from tensorboardX import SummaryWriter

from opts import parse_opts
from core.model import generate_model,generate_visual_Erase_model, generate_vaaerase_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform, get_saliency_transform
from core.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader
from transforms.temporal import TSN
from transforms.target import ClassLabel
from train_new import train_epoch
from validation import val_epoch




def main():
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)
    opt.saliency_level = 'feature_map'
    model, parameters = generate_vaaerase_model(opt)
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
    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    saliency_transform = get_saliency_transform(opt, 'test', spatial_transform)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
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