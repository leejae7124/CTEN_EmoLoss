import os
import datetime
import shutil
import torch
from transforms.temporal import TSN
from transforms.spatial import Preprocessing, Preprocessing_saliency
# from datasets.ve8_saliency import get_default_video_loader, get_default_saliency_loader
from datasets.caer import get_default_video_loader, get_default_saliency_loader
import numpy as np



def local2global_path(opt):
    if opt.root_path != '':
        # opt.video_path = os.path.join(opt.root_path, opt.video_path)
        # opt.audio_path = os.path.join(opt.root_path, opt.audio_path)
        # opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        if opt.debug:
            opt.result_path = "debug"
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.expr_name == '':
            now = datetime.datetime.now()
            now = now.strftime('result_%Y%m%d_%H%M%S')
            opt.result_path = os.path.join(opt.result_path, now)
        else:
            opt.result_path = os.path.join(opt.result_path, opt.expr_name)

            if os.path.exists(opt.result_path):
                shutil.rmtree(opt.result_path)
            os.mkdir(opt.result_path)
        opt.log_path = os.path.join(opt.result_path, "tensorboard")
        opt.ckpt_path = os.path.join(opt.result_path, "checkpoints")
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        if not os.path.exists(opt.ckpt_path):
            os.mkdir(opt.ckpt_path)
    else:
        raise Exception

def get_spatial_transform(opt, mode):
    if mode == "train":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=False)
    elif mode == "val":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    elif mode == "test":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    else:
        raise Exception
def get_saliency_transform(opt, mode, spatial_transform):
    if mode == "train":
        return Preprocessing_saliency(original_preprocessing_instance=spatial_transform)
    elif mode == "val":
        return Preprocessing_saliency(original_preprocessing_instance=spatial_transform)
    elif mode == "test":
        return Preprocessing_saliency(original_preprocessing_instance=spatial_transform)
    else:
        raise Exception

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def process_data_item(opt, data_item):
    visual, saliency, target, audio, visualization_item, video, n_frames, sal_path = data_item
    
    target = target.cuda()
    visual = visual.cuda()
    saliency = saliency.cuda()
    audio = audio.cuda()
    batch = visual.size(0)
    return visual, saliency, target, audio, visualization_item, batch, {'video':video, 'n_frames':n_frames}, sal_path

def run_model(opt, inputs, model, criterion, i=0, print_attention=False, period=30, return_attention=False):
    visual, target, audio, saliency_map = inputs
    outputs, gamma = model([visual, audio, saliency_map])
    loss = criterion(outputs, target)
    return outputs,loss,gamma

def run_model_loss(opt, inputs, model, criterion, i=0, print_attention=True, period=30, return_attention=False, use_intensity=True,
):
    """
    CTEN + intensity loss용 run_model_loss.

    inputs:
        [visual, target, audio, saliency_map]

    intensity train:
        output, gamma, cam_map = model([visual, audio], target_class=target, compute_gradcam=True)
        loss = CE + lambda * Align

    val/test 또는 CE-only:
        output, gamma = model([visual, audio], compute_gradcam=False)
        loss = CE
    """
    visual, target, audio, saliency_map = inputs

    need_align = (
        hasattr(opt, "loss_func")
        and str(opt.loss_func).startswith("ce_intensity")
    )

    do_cam = (
        need_align and use_intensity and model.training and torch.is_grad_enabled()
    )

    if do_cam:
        # CTEN intensity model:
        # return output, gamma, cam_map
        y_pred, gamma, cam_map = model(
            [visual, audio], target_class=target, compute_gradcam=True,
        )

        # CE와 Align을 분리 계산
        cls = criterion.cls_loss(y_pred, target)
        align = criterion.intensity_loss(cam_map, saliency_map)

        lam = float(getattr(criterion, "lambda_intensity", 1.0))
        loss = cls + lam * align

        if (i % period) == 0:
            ratio = (lam * align / (cls + 1e-12)).detach().item()
            print(
                f"[loss] cls={cls.item():.4f}  "
                f"align={align.item():.4f}  "
                f"lambda={lam:.3f}  "
                f"total={loss.item():.4f}  "
                f"(lam*align/cls={ratio:.3f})"
            )

            if hasattr(criterion.intensity_loss, "last_terms"):
                terms = criterion.intensity_loss.last_terms
                if len(terms) > 0:
                    print(
                        "[align terms] "
                        f"rmse={terms.get('rmse', None)}  "
                        f"grad={terms.get('grad', None)}  "
                        f"normal={terms.get('normal', None)}  "
                        f"total={terms.get('total', None)}"
                    )

    else:
        # val/test 또는 CE-only 상황
        y_pred, gamma = model(
            [visual, audio],
            compute_gradcam=False,
        )

        if need_align:
            # ce_intensity 옵션이어도 val/test에서는 CE만 계산
            loss = criterion.cls_loss(y_pred, target)
        else:
            loss = criterion(y_pred, target)

    if i % period == 0 and print_attention:
        print("====gamma====")
        print(gamma)

    if not return_attention:
        return y_pred, loss, gamma
    else:
        return y_pred, loss, [gamma]

def run_model_inf(opt, inputs, model, i=0, print_attention=False, period=30, return_attention=False):
    visual, _, audio, saliency_map = inputs
    outputs, gamma = model([visual, audio, saliency_map])
    return outputs,gamma

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    values, indices = outputs.topk(k=1, dim=1, largest=True)
    pred = indices
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elements = correct.float()
    n_correct_elements = n_correct_elements.sum()
    n_correct_elements = n_correct_elements.item()
    return n_correct_elements / batch_size

def get_new_indices(frame_indices):
    pass

# core/utils.py
def _batch_augment_impl(video_item, erase_index, opt, visual, saliency_map, sal_path):
    # 공통 구현 (증강 OFF + OOR 방지 + 폴백)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    spatial_transform = get_spatial_transform(opt, 'val')
    saliency_transform = get_saliency_transform(opt, 'val', spatial_transform)
    loadder = get_default_video_loader()
    s_loadder = get_default_saliency_loader()
    seq_len = opt.seq_len

    for i in range(len(video_item['video'])):
        frame_indices = []
        video_path = video_item['video'][i]
        n_frames   = int(video_item['n_frames'][i])
        saliency_path = sal_path[i]
        n_frames = video_item['n_frames'][i]
        segment_duration = max(1, int(n_frames // seq_len))

        index = (torch.where(erase_index[i] == True)[0])
        for ind in range(len(index)):
            t = int(index[ind].detach().cpu())
            sd = int(segment_duration)
            # 끝 경계 n_frames까지만
            max_len = min(sd * (t + 1), n_frames)
            start = t * sd + 1
            if start > max_len:
                continue
            # 1-based, end 포함
            test = list(range(start, max_len + 1))
            frame_indices.extend(test)

        # 빈 선택 폴백: 전체 프레임
        if len(index) == 0 or len(frame_indices) == 0:
            frame_indices = list(range(1, n_frames + 1))

        try:
            snippets_frame_idx = temporal_transform(frame_indices)
        except Exception as e:
            print(video_path, n_frames, e)
            # 마지막 창 폴백
            s = max(1, n_frames - opt.snippet_duration + 1)
            snippets_frame_idx = [list(range(s, s + opt.snippet_duration))]

        snippets = []
        saliency_snippets = [] #saliency map을 추가로 로드해야 함.
        for snippet_frame_idx in snippets_frame_idx:
            # 로더 직전 “창 이동”으로 경계 보정
            if len(snippet_frame_idx) > 0:
                k = len(snippet_frame_idx)
                if snippet_frame_idx[-1] > n_frames:
                    s = max(1, n_frames - k + 1)
                    snippet_frame_idx = list(range(s, s + k))
                if snippet_frame_idx[0] < 1:
                    s = 1
                    snippet_frame_idx = list(range(s, s + k))

            snippet = loadder(video_path, snippet_frame_idx)
            snippets.append(snippet)
            
            #saliency map 로드 추가
            saliency_snippet = s_loadder(saliency_path, snippet_frame_idx, n_frames)
            # print("sal snippet: ", saliency_snippet)
            saliency_snippets.append(saliency_snippet)

        # 랜덤 파라미터 호출 제거(증강 OFF)
        spatial_transform.randomize_parameters()
        saliency_transform.randomize_parameters()

        snippets_transformed = []
        saliency_snippets_transformed = []
        for snippet in snippets:
            snippet = [spatial_transform(img) for img in snippet]
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        snippets = torch.stack(snippets, 0)
        visual[i] = snippets
        
        for saliency_snippet in saliency_snippets:
            # print("snippet: ", saliency_snippet)
            saliency_snippet = [saliency_transform(saliency) for saliency in saliency_snippet] #saliency에 맞게 transform
            # print("transformed_saliency_snippet: ", saliency_snippet)
            saliency_snippet = torch.stack(saliency_snippet, 0).permute(1, 0, 2, 3)
            saliency_snippets_transformed.append(saliency_snippet)
        saliency_snippets = saliency_snippets_transformed
        saliency_snippets = torch.stack(saliency_snippets, 0)
        
        saliency_map[i] = saliency_snippets

    return visual, saliency_map

# 기존 시그니처 유지: 두 함수 모두 공통 코어만 호출
def batch_augment(video_item, erase_index, opt, visual, saliency_map, sal_path):
    return _batch_augment_impl(video_item, erase_index, opt, visual, saliency_map, sal_path)

def batch_augment2(video_item, erase_index, opt, visual, saliency_map, sal_path):
    return _batch_augment_impl(video_item, erase_index, opt, visual, saliency_map, sal_path)


# def batch_augment(video_item,erase_index,opt,visual):
#     temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
#     spatial_transform = get_spatial_transform(opt, 'train')
#     loadder=get_default_video_loader()
#     seq_len=opt.seq_len
#     for i in range(len(video_item['video'])):
#         frame_indices = []
#         video_path=video_item['video'][i]
#         n_frames = video_item['n_frames'][i]
#         segment_duration = n_frames // seq_len
#         index=(torch.where(erase_index[i]==True)[0])
#         for ind in range(len(index)):
#             t=index[ind].detach().cpu()
#             segment_duration=segment_duration.detach().cpu()
#             max_len=min(segment_duration*(t+1),n_frames+1)
#             test=list(range(t*segment_duration+1, max_len+1))
#             frame_indices.extend(test)
#         # if len(index)==0 or len(frame_indices) == 0:
#         #     frame_indices=list(range(1,n_frames+1))
#         try:
#             snippets_frame_idx = temporal_transform(frame_indices)
#         except:
#             print(video_path,n_frames)
#         snippets = []
#         for snippet_frame_idx in snippets_frame_idx:
#             snippet =loadder(video_path, snippet_frame_idx)
#             snippets.append(snippet)
#         spatial_transform.randomize_parameters()
#         snippets_transformed = []
#         for snippet in snippets:
#             snippet = [spatial_transform(img) for img in snippet]
#             snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
#             snippets_transformed.append(snippet)
#         snippets = snippets_transformed
#         snippets = torch.stack(snippets, 0)
#         visual[i]=snippets
#     return visual

# def batch_augment2(video_item,erase_index,opt,visual):
#     temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
#     spatial_transform = get_spatial_transform(opt, 'train')
#     loadder=get_default_video_loader()
#     seq_len=opt.seq_len
#     for i in range(len(video_item['video'])):
#         frame_indices = []
#         video_path=video_item['video'][i]
#         n_frames = video_item['n_frames'][i]
#         segment_duration = n_frames // seq_len

#         index=(torch.where(erase_index[i]==True)[0])
#         for ind in range(len(index)):
#             t=index[ind].detach().cpu()
#             segment_duration=segment_duration.detach().cpu()
#             max_len=min(segment_duration*(t+1),n_frames+1)
#             test=list(range(t*segment_duration+1, max_len+1))
#             frame_indices.extend(test)
#         if len(index)==0 or len(frame_indices) == 0:
#             frame_indices=list(range(1,n_frames+1))
#         snippets_frame_idx = temporal_transform(frame_indices)
#         snippets = []
#         for snippet_frame_idx in snippets_frame_idx:
#             snippet =loadder(video_path, snippet_frame_idx)
#             snippets.append(snippet)
#         spatial_transform.randomize_parameters()
#         snippets_transformed = []
#         for snippet in snippets:
#             snippet = [spatial_transform(img) for img in snippet]
#             snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
#             snippets_transformed.append(snippet)
#         snippets = snippets_transformed
#         snippets = torch.stack(snippets, 0)
#         visual[i]=snippets
#     return visual

def batch_random_erase(video_item,opt,visual):
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    spatial_transform = get_spatial_transform(opt, 'train')
    loadder = get_default_video_loader()
    seq_len = opt.seq_len
    for i in range(len(video_item['video'])):
        frame_indices = []
        video_path = video_item['video'][i]
        n_frames = video_item['n_frames'][i]
        segment_duration = n_frames // seq_len
        erase_index = torch.from_numpy(np.random.randint(2,size=(32,16)))
        index = (torch.where(erase_index[i] == True)[0])
        for ind in range(len(index)):
            t = index[ind].detach().cpu()
            segment_duration = segment_duration.detach().cpu()
            max_len = min(segment_duration * (t + 1), n_frames + 1)
            test = list(range(t * segment_duration + 1, max_len + 1))
            frame_indices.extend(test)
        if len(index) == 0 or len(frame_indices) == 0:
            frame_indices = list(range(1, n_frames + 1))
        snippets_frame_idx = temporal_transform(frame_indices)
        snippets = []
        for snippet_frame_idx in snippets_frame_idx:
            snippet = loadder(video_path, snippet_frame_idx)
            snippets.append(snippet)
        spatial_transform.randomize_parameters()
        snippets_transformed = []
        for snippet in snippets:
            snippet = [spatial_transform(img) for img in snippet]
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        snippets = torch.stack(snippets, 0)
        visual[i] = snippets
    return visual
