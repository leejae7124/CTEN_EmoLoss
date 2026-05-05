import torch.nn as nn
from models.vaanet import VAANet
from models.vaanet_erase_saliency import VAANetErase
from models.vaanet_erase_saliency_mean_norm import VAANetErase_mean
from models.vaanet_erase_saliency_binary import VAANetErase_binary
from models.vaanet_erase_saliency_mean_norm_dac import VAANetErase_mean_dac
from models.vaanet_erase_saliency_test import VAANetEraseTest
from models.visual_stream import VisualStream
from models.visual_stream_w_Erase import VisualErase
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def generate_model(opt):
    model = VAANet(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    model = model.cuda()
    return model, model.parameters()

def generate_vaaerase_model(opt):
    model = VAANetErase(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        saliency_level=opt.saliency_level
    )
    # model = nn.DataParallel(model)
    model = model.cuda()
    return model, model.parameters()

def generate_vaaerase_model(opt):
    model = VAANetErase_mean(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        saliency_level=opt.saliency_level
    )
    # model = nn.DataParallel(model)
    model = model.cuda()
    return model, model.parameters()

def generate_vaaerase_saliency_binary_model(opt):
    model = VAANetErase_binary(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        saliency_level=opt.saliency_level
    )
    # model = nn.DataParallel(model)
    model = model.cuda()
    return model, model.parameters()

def generate_vaaerase_test_model(opt):
    model = VAANetEraseTest(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        saliency_level=opt.saliency_level
    )
    # model = nn.DataParallel(model)
    model = model.cuda()
    return model, model.parameters()

def generate_vaaerase_saliency_dac_model(opt):
    model = VAANetErase_mean_dac(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        saliency_level=opt.saliency_level
    )
    # model = nn.DataParallel(model)
    model = model.cuda()
    return model, model.parameters()

def generate_visual_model(opt):
    model=VisualStream(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    model = nn.DataParallel(model)
    model=model.cuda()
    return model,model.parameters()

def generate_visual_Erase_model(opt):
    model=VisualErase(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    model = nn.DataParallel(model)
    model=model.cuda()
    return model, model.parameters()
