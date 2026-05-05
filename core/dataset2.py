from datasets.ve8_saliency import VE8Dataset
from datasets.tsl import TSLDataset
from datasets.caer import CAERDataset
from torch.utils.data import DataLoader


def get_ve8(opt, subset, transforms, saliency_transform):
    spatial_transform, temporal_transform, target_transform = transforms
    return VE8Dataset(
        opt.video_path,
        opt.audio_path,
        opt.annotation_path,
        opt.saliency_path,
        subset,
        opt.fps,
        spatial_transform,
        temporal_transform,
        target_transform,
        saliency_transform,
        need_audio=True
    )


def get_tsl(opt, subset, transforms, saliency_transform):
    spatial_transform, temporal_transform, target_transform = transforms
    return TSLDataset(
        video_path=opt.video_path,
        audio_path=opt.audio_path,
        saliency_path=opt.saliency_path,
        subset=subset,
        fps=opt.fps,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        saliency_transform=saliency_transform,
        need_audio=True,
    )


def get_caer(opt, subset, transforms, saliency_transform):
    spatial_transform, temporal_transform, target_transform = transforms
    return CAERDataset(
        video_path=opt.video_path,
        audio_path=opt.audio_path,
        saliency_path=opt.saliency_path,
        subset=subset,
        fps=opt.fps,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        saliency_transform=saliency_transform,
        need_audio=True,
    )


def get_training_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]

    if opt.dataset == 've8':
        return get_ve8(opt, 'training', transforms, saliency_transform)
    elif opt.dataset == 'tsl':
        return get_tsl(opt, 'train', transforms, saliency_transform)
    elif opt.dataset == 'caer':
        return get_caer(opt, 'train', transforms, saliency_transform)
    else:
        raise Exception


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]

    if opt.dataset == 've8':
        return get_ve8(opt, 'validation', transforms, saliency_transform)
    elif opt.dataset == 'tsl':
        return get_tsl(opt, 'validation', transforms, saliency_transform)
    elif opt.dataset == 'caer':
        return get_caer(opt, 'validation', transforms, saliency_transform)
    else:
        raise Exception


def get_test_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]

    if opt.dataset == 've8':
        return get_ve8(opt, 'validation', transforms, saliency_transform)
    elif opt.dataset == 'tsl':
        return get_tsl(opt, 'test', transforms, saliency_transform)
    elif opt.dataset == 'caer':
        return get_caer(opt, 'test', transforms, saliency_transform)
    else:
        raise Exception


def get_data_loader(opt, dataset, shuffle, batch_size=0):
    batch_size = opt.batch_size if batch_size == 0 else batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.n_threads,
        pin_memory=True,
        drop_last=opt.dl
    )