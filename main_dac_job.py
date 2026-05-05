import os
from copy import deepcopy
import torch
from torch.cuda import device_count
from tensorboardX import SummaryWriter

import numpy as np
import random

from opts_dac_job import parse_opts
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

def build_eval_jobs(opt):
    if not opt.test_jobs:
        return [
            {
                "annotation_path": opt.annotation_path,
                "checkpoint_path": opt.checkpoint_path,
                "role": "single",
            }
        ]

    jobs = []
    for job_str in opt.test_jobs:
        try:
            annotation_path, checkpoint_path, role = job_str.split("::", 2)
        except ValueError:
            raise ValueError(
                f"Invalid --test_jobs item: {job_str}\n"
                "Expected format: annotation::checkpoint::role"
            )

        role = role.strip().lower()
        if role not in ("acc", "f1"):
            raise ValueError(f"Invalid role: {role}. Use 'acc' or 'f1'.")

        jobs.append(
            {
                "annotation_path": annotation_path,
                "checkpoint_path": checkpoint_path,
                "role": role,
            }
        )

    return jobs


def main():
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)

    set_seed(42)

    opt.saliency_level = 'feature_map'

    jobs = build_eval_jobs(opt)
    all_results = []
    base_log_path = opt.log_path

    for job in jobs:
        job_opt = deepcopy(opt)
        job_opt.annotation_path = job["annotation_path"]
        job_opt.checkpoint_path = job["checkpoint_path"]
        job_opt.metric_role = job["role"]

        if not os.path.isabs(job_opt.annotation_path):
            job_opt.annotation_path = os.path.join(job_opt.root_path, job_opt.annotation_path)

        if not os.path.isabs(job_opt.checkpoint_path):
            job_opt.checkpoint_path = os.path.join(job_opt.root_path, job_opt.checkpoint_path)

        ann_name = os.path.splitext(os.path.basename(job_opt.annotation_path))[0]
        ckpt_name = os.path.splitext(os.path.basename(job_opt.checkpoint_path))[0]

        job_opt.log_path = os.path.join(
            base_log_path,
            f"{ann_name}__{job_opt.metric_role}__{ckpt_name}"
        )

        print(f"\n===== Evaluating {job_opt.metric_role} =====")
        print(f"annotation: {job_opt.annotation_path}")
        print(f"checkpoint: {job_opt.checkpoint_path}")

        model, parameters = generate_vaaerase_saliency_dac_model(job_opt)

        if job_opt.checkpoint_path:
            checkpoint = torch.load(job_opt.checkpoint_path, map_location='cuda:0')
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded checkpoint from {job_opt.checkpoint_path}")
        else:
            print("Warning: No checkpoint path provided.")

        criterion = get_loss(job_opt).cuda()
        optimizer = get_optim(job_opt, parameters)
        writer = SummaryWriter(logdir=job_opt.log_path)

        spatial_transform = get_spatial_transform(job_opt, 'test')
        saliency_transform = get_saliency_transform(job_opt, 'test', spatial_transform)
        temporal_transform = TSN(
            seq_len=job_opt.seq_len,
            snippet_duration=job_opt.snippet_duration,
            center=False
        )
        target_transform = ClassLabel()

        validation_data = get_validation_set(
            job_opt, spatial_transform, temporal_transform, target_transform, saliency_transform
        )
        val_loader = get_data_loader(job_opt, validation_data, shuffle=False)

        result = val_epoch(0, val_loader, model, criterion, job_opt, writer, optimizer)
        print(result)

        all_results.append({
            "role": job_opt.metric_role,
            "annotation_path": job_opt.annotation_path,
            "checkpoint_path": job_opt.checkpoint_path,
            "result": result
        })

        writer.close()

    print("\n===== All Results =====")
    for item in all_results:
        print(item)


if __name__ == "__main__":
    main()