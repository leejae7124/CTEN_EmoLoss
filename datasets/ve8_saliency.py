import torch
import torch.utils.data as data

from torchvision import get_image_backend

from PIL import Image
import random
import json
import os
import functools
import librosa
import numpy as np
import torchaudio
torchaudio.set_audio_backend("ffmpeg")


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        return float(input_file.read().rstrip('\n\r'))


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    #data keys: 'labels', 'database'
    #data values: ['Anger', 'Anticipation', ...], {'비디오 이름1', '비디오 이름2', ...}
    for key, value in data['database'].items():
        if value['subset'] == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
    return video_names, annotations


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_saliency_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def get_default_saliency_image_loader():
    return pil_saliency_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
        assert os.path.exists(image_path), "image does not exists"
        video.append(image_loader(image_path))
    return video

###추가한 함수: Saliency loader
def saliency_loader(saliency_dir_path, frame_indices, n_frames, saliency_image_loader):
    #n_frames는 마지막 프레임
    video = []
    for i in frame_indices:
        # print("i: ", i)
        # print("n_frames: ", n_frames)
        saliency_path = os.path.join(saliency_dir_path, '{:06d}.jpg'.format(i))
        assert os.path.exists(saliency_path), "image does not exists"
        video.append(saliency_image_loader(saliency_path))
    return video
# saliency_loader 함수 (수정)
# def saliency_loader(saliency_dir_path, frame_indices, saliency_image_loader):
#     video = []
#     for i in frame_indices:
#         print(f"Loading saliency for frame index: {i}") # 디버깅 추가
#         saliency_path = os.path.join(saliency_dir_path, 'pred_sal_img{:05d}-{:06d}.jpg'.format(i, i+16))
#         print(f"  Saliency path: {saliency_path}") # 디버깅 추가
        
#         # 파일 존재 여부 다시 확인
#         if not os.path.exists(saliency_path):
#             # 파일이 없으면 이 단계에서 명확한 오류 발생
#             raise FileNotFoundError(f"Missing saliency image for frame {i}: {saliency_path}")

#         try:
#             loaded_img = saliency_image_loader(saliency_path)
            
#             # 로드된 객체가 PIL.Image.Image 인스턴스인지 명확히 확인
#             if not isinstance(loaded_img, Image.Image):
#                 # PIL.Image 객체가 아니면 유효하지 않은 것으로 간주하고 오류 발생
#                 raise ValueError(
#                     f"saliency_image_loader did not return a PIL.Image object for path: {saliency_path}. "
#                     f"Instead got type: {type(loaded_img)}"
#                 )
            
#             # 이미지 모드 확인 (L 모드가 아니면 경고 또는 오류)
#             if loaded_img.mode != 'L':
#                 print(f"Warning: Saliency image {saliency_path} has mode {loaded_img.mode}, expected 'L'. Converting.")
#                 loaded_img = loaded_img.convert('L') # 강제로 L 모드로 변환

#             video.append(loaded_img)
#         except Exception as e:
#             # 로딩/변환 중 발생한 예외를 명확히 보고
#             print(f"CRITICAL ERROR loading saliency image {saliency_path} (frame {i}): {e}")
#             # 이 오류는 데이터 로딩 파이프라인 전체를 중단시켜야 합니다.
#             # `__getitem__`이 `None`을 받지 않도록 명확히 오류를 전파합니다.
#             raise # 오류를 다시 발생시켜 프로그램 중단
            
#     return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def get_default_saliency_loader():
    saliency_image_loader = get_default_saliency_image_loader()
    return functools.partial(saliency_loader, saliency_image_loader=saliency_image_loader)

def preprocess_audio(audio_path):
    "Extract audio features from an audio file"
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    return mfccs


class VE8Dataset(data.Dataset):
    def __init__(self,
                 video_path,
                 audio_path,
                 annotation_path,
                 saliency_path,
                 subset,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 saliency_transform=None,
                 get_loader=get_default_video_loader,
                 get_saliency_loader=get_default_saliency_loader,
                 need_audio=True):
        self.data, self.class_names = make_dataset(
            video_root_path=video_path,
            annotation_path=annotation_path,
            audio_root_path=audio_path,
            saliency_root_path=saliency_path,
            subset=subset,
            fps=fps,
            need_audio=need_audio
        )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.saliency_transform = saliency_transform
        self.loader = get_loader()
        self.saliency_loader = get_saliency_loader()
        self.fps = fps
        self.ORIGINAL_FPS = 30
        self.need_audio = need_audio
        self.norm_mean = -6.6268077#이 부분 다름
        self.norm_std = 5.358466#이 부분 다름
        self.audio_n_segments = 16 #이 부분 다름

    def __getitem__(self, index):
        data_item = self.data[index]
        # print("keys: ", data_item.keys()) #['video', 'segment', 'n_frames', 'video_id', 'label', 'frame_indices']
        video_path = data_item['video']
        saliency_path = data_item['saliency']
        # print("video path: ", video_path)
        frame_indices = data_item['frame_indices']#frames 인덱스 값들이 저장된다.
        n_frames = data_item['n_frames']
        snippets_frame_idx = self.temporal_transform(frame_indices)
        

        # if self.need_audio:
        #     timeseries_length = 4096
        #     audio_path = data_item['audio']
        #     feature = preprocess_audio(audio_path).T
        #     k = timeseries_length // feature.shape[0] + 1
        #     feature = np.tile(feature, reps=(k, 1))
        #     audios = feature[:timeseries_length, :]
        #     audios = torch.FloatTensor(audios)
        # else:
        #     audios = []
        if self.need_audio: #내부가 다름
            timeseries_length = 100*self.audio_n_segments
            waveform, sr = torchaudio.load(data_item['audio'])
            
            # 1) 로드 직후 바로 검사
            if not torch.isfinite(waveform).all():
                print("[BAD AUDIO] non-finite in raw waveform:", data_item['audio'])
            
            
            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(waveform, 
                                                htk_compat=True,
                                                sample_frequency=sr,
                                                use_energy=False,
                                                window_type='hanning',
                                                num_mel_bins=128,
                                                dither=0.0,
                                                frame_shift=10)
            if fbank.shape[0]<=timeseries_length:
                k = timeseries_length // fbank.shape[0] + 1
                fbank = np.tile(fbank, reps=(k, 1))
                audios = fbank[:timeseries_length, :]
            else:
                blk = int(fbank.shape[0]/self.audio_n_segments)
                aud = []
                for i in list(range(0,fbank.shape[0],blk))[:self.audio_n_segments]:
                    ind = i+int(random.random()*(blk-100))
                    aud.append(fbank[ind:ind+100])
                audios = torch.cat(aud)
            if audios.shape[0]!=timeseries_length:
                print(audios.shape)
            audios = torch.FloatTensor(audios)
            audios = (audios - self.norm_mean) / (self.norm_std * 2)
        else:
            audios = []

        snippets = []
        saliency_snippets = [] #saliency map을 추가로 로드해야 함.
        for snippet_frame_idx in snippets_frame_idx:
            # print("snippets frame idx: ", snippet_frame_idx) #하나의 스니펫에 저장된 인덱스 번호들.
            snippet = self.loader(video_path, snippet_frame_idx)
            # print("snippet: ", snippet)
            snippets.append(snippet)

            #saliency map 로드 추가
            saliency_snippet = self.saliency_loader(saliency_path, snippet_frame_idx, n_frames)
            # print("sal snippet: ", saliency_snippet)
            saliency_snippets.append(saliency_snippet)

        self.spatial_transform.randomize_parameters()
        self.saliency_transform.randomize_parameters()

        # [추가] 동기화가 잘 되었는지 여기서 직접 확인!
        spatial_crop_pos = self.spatial_transform.get_current_crop_params()
        saliency_crop_pos = self.saliency_transform.f1_1.crop_position # Saliency의 crop 위치
        # print(f"✅ [SYNC CHECK] Video Crop: {spatial_crop_pos} | Saliency Crop: {saliency_crop_pos}")
        
        # [추가] 동기화 실패 시 에러 발생
        assert spatial_crop_pos == saliency_crop_pos, "Crop positions do not match!"

        #print(f"[디버깅] spatial crop: {self.spatial_transform.get_current_crop_params()} | saliency crop: {self.saliency_transform.original_preprocessing.get_current_crop_params()}")

        snippets_transformed = []
        saliency_snippets_transformed = []
        for snippet in snippets:
            snippet = [self.spatial_transform(img) for img in snippet]
            # print("transformed_snippet: ", snippet)
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets_transformed.append(snippet)

        
        snippets = snippets_transformed
        snippets = torch.stack(snippets, 0)

        # [DEBUG] visual 범위/형태 (첫 샘플 한 번만)
        if index == 0 and not hasattr(self, "_dbg_printed"):
            print("rgb shape:", tuple(snippets.shape),
                  "range:", float(snippets.min()), float(snippets.max()),
                  "dtype:", snippets.dtype, "device:", snippets.device)

        for saliency_snippet in saliency_snippets:
            # print("snippet: ", saliency_snippet)
            saliency_snippet = [self.saliency_transform(saliency) for saliency in saliency_snippet] #saliency에 맞게 transform
            # print("transformed_saliency_snippet: ", saliency_snippet)
            saliency_snippet = torch.stack(saliency_snippet, 0).permute(1, 0, 2, 3)
            saliency_snippets_transformed.append(saliency_snippet)
        saliency_snippets = saliency_snippets_transformed
        saliency_snippets = torch.stack(saliency_snippets, 0)
        # saliency_snippets = torch.stack(saliency_snippets, 0)
        # [DEBUG] saliency 범위/형태 (첫 샘플 한 번만)
        if index == 0 and not hasattr(self, "_dbg_printed"):
            print("sal shape:", tuple(saliency_snippets.shape),
                  "range:", float(saliency_snippets.min()), float(saliency_snippets.max()),
                  "dtype:", saliency_snippets.dtype, "device:", saliency_snippets.device)
            self._dbg_printed = True
        
        # ... snippets, saliency_snippets 를 텐서로 만든 직후
        # snippets: [num_snip, C, T, H, W]
        # saliency_snippets: [num_snip, 1, T, H, W]

        if index == 0:  # 첫 샘플만 저장 (원하면 글로벌 카운터나 env flag로 제어)
            import os
            from torchvision.utils import save_image
            os.makedirs("./debug_crop", exist_ok=True)

            vid = data_item['video_id']
            sn = 0   # 첫 스니펫
            t_list = [0, min(5, snippets.size(2)-1), snippets.size(2)-1]  # 처음/중간/끝 프레임

            # 역정규화 도우미 (네 파이프에서 visual은 0~1 범위일 수도 있어요. 필요시 조정)
            def _maybe_denorm(x):
                # x: [C,H,W], 이미 ToTensor(div 255)만 했다면 생략 가능
                return x.clamp(0,1)

            for t in t_list:
                abs_f = int(snippets_frame_idx[sn][t])  # 예: 1, 17, 33, ...
                rgb = snippets[sn, :, t]                 # [C,H,W]
                sm  = saliency_snippets[sn, 0, t]        # [H,W]

                rgb = _maybe_denorm(rgb)
                # 마스크를 빨간 채널로 오버레이
                alpha = 0.5
                sm3 = torch.stack([sm, torch.zeros_like(sm), torch.zeros_like(sm)], dim=0)  # [3,H,W]
                overlay = (1 - alpha) * rgb + alpha * sm3
                overlay = overlay.clamp(0, 1)
                base = f"{vid}_sn{sn:02d}_t{t:02d}_f{abs_f:06d}"
                # print("!!!!!!!!!!!!!!", base)

                save_image(rgb.cpu(),      f"./debug_crop/{base}_frame.png")
                save_image(sm.unsqueeze(0).cpu(), f"./debug_crop/{base}_mask.png")   # 그레이스케일
                save_image(overlay.cpu(),  f"./debug_crop/{base}_overlay.png")

                # print(f"[DEBUG SAVE] ./debug_crop/{base}_overlay.png")


        # print("==== Saliency Snippet 디버깅 ====")
        # for i, snippet in enumerate(saliency_snippets_transformed):
        #     print(f"[Snippet {i}] type: {type(snippet)}, device: {getattr(snippet, 'device', 'N/A')}, dtype: {getattr(snippet, 'dtype', 'N/A')}")
        #     if isinstance(snippet, torch.Tensor):
        #         print(f"  shape: {snippet.shape}")
        # print("=================================")


        target = self.target_transform(data_item)
        visualization_item = [data_item['video_id']]
        # print("=2=2=2=2=2=2=2=   ", visualization_item)

        return snippets, saliency_snippets, target, audios, visualization_item,  data_item['video'], data_item['n_frames'], data_item['saliency']

    def __len__(self):
        return len(self.data)


def make_dataset(video_root_path, annotation_path, audio_root_path, saliency_root_path, subset, fps=30, need_audio=False):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    # print("idx_to_class", idx_to_class)
    dataset = []
    for i in range(len(video_names)):
        # if i % 100 == 0:
        #     print("Dataset loading [{}/{}]".format(i, len(video_names)))
        video_path = os.path.join(video_root_path, video_names[i])
        # print("video path: ", video_path)
        saliency_path = os.path.join(saliency_root_path, video_names[i])
        # print("saliency path:", saliency_path)

        if need_audio:
            audio_path = os.path.join(audio_root_path, video_names[i] + '.mp3')
            # print("-- audio -- : ", audio_path)
        else:
            audio_path = None
        # print(saliency_path)

        # assert os.path.exists(audio_path), audio_path
        assert os.path.exists(video_path), video_path
        assert os.path.exists(saliency_path), saliency_path

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            # print(video_path)
            continue

        begin_t = 1
        n_frames = n_frames
        end_t = n_frames

        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1],
            'saliency': saliency_path #saliency path 추가
        }
        if need_audio: sample['audio'] = audio_path
        assert len(annotations) != 0
        sample['label'] = class_to_idx[annotations[i]['label']]
        # print("(((((((((((((((((())))))))))))))))))", sample)

        ORIGINAL_FPS = 30
        step = ORIGINAL_FPS // fps

        sample['frame_indices'] = list(range(1, n_frames + 1, step))#프레임 인덱스 값들이 저장된다.
        # print("dddddd ", sample['frame_indices']) 
        dataset.append(sample)
    return dataset, idx_to_class
