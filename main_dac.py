import os
import torch
from torch.cuda import device_count
from tensorboardX import SummaryWriter

import numpy as np
import random

from opts_dac import parse_opts
from core.model import generate_vaaerase_saliency_dac_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform, get_saliency_transform
from core.dataset import get_validation_set, get_data_loader
from transforms.temporal import TSN
from transforms.target import ClassLabel
from validation_dac2 import val_epoch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)
    set_seed(42)

    opt.saliency_level = 'feature_map'

    model, parameters = generate_vaaerase_saliency_dac_model(opt)

    if opt.checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path, map_location='cuda:0')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {opt.checkpoint_path}")
    else:
        print("Warning: No checkpoint path provided.")

    criterion = get_loss(opt).cuda()
    optimizer = get_optim(opt, parameters)
    writer = SummaryWriter(logdir=opt.log_path)

    spatial_transform = get_spatial_transform(opt, 'test')
    saliency_transform = get_saliency_transform(opt, 'test', spatial_transform)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()

    validation_data = get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform, saliency_transform
    )
    val_loader = get_data_loader(opt, validation_data, shuffle=False)

    result = val_epoch(0, val_loader, model, criterion, opt, writer, optimizer)
    print(result)

    writer.close()


if __name__ == "__main__":
    main()