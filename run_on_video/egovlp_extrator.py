import torch
import numpy as np
import decord
import random
import transformers
from run_on_video.egovlp.model import FrozenInTime
from run_on_video.egovlp.model_utils import state_dict_data_parallel_fix
from torchvision import transforms
from pytorchvideo.transforms import Normalize as NormalizeVideo
import time
decord.bridge.set_bridge("torch")

config = {
    "video_params": {
        "model": "SpaceTimeTransformer",
        "arch_config": "base_patch16_224",
        "num_frames": 16,
        "pretrained": True,
        "time_init": "zeros"
    },
    "text_params": {
        "model": "distilbert-base-uncased",
        "pretrained": True,
        "input": "text"
    },
    "projection_dim": 256,
    "load_checkpoint": "./egovlp/egovlp.pth"
}


class VideoLoader:
    """Pytorch video loader.
    """
    def __init__(
            self,
            fps=1.875,
            input_res=224,
            center_crop=256,
            norm_mean=(0.485, 0.456, 0.406),
            norm_std=(0.229, 0.224, 0.225)
    ):
        self.fps = fps
        normalize = NormalizeVideo(mean=norm_mean, std=norm_std)
        self.transforms = transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ])

    def read_frames_decord(self, video_path, sample='rand', fix_start=None):
        load_video_start_time = time.time()
        video_reader = decord.VideoReader(video_path, num_threads=1)

        vlen = len(video_reader)
        num_frames = int(vlen / video_reader.get_avg_fps() * self.fps * 4)

        frame_idxs = self.sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
        video_reader.skip_frames(1)

        get_frame_start_time = time.time()
        frames = video_reader.get_batch(frame_idxs)

        transform_frame_start_time = time.time()
        #frames = torch.from_numpy(frames.asnumpy().astype(np.float32) / 255)
        frames = frames/255
        frames = frames.permute(0, 3, 1, 2) # [T, H, W, C]  ---> [T, C, H, W]

        frames = self.transform_frames(frames)
        #print("transform frame time: ",time.time()-transform_frame_start_time)

        return frames

    def transform_frames(self, imgs):

        imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
        imgs = self.transforms(imgs)
        imgs = imgs.transpose(0, 1)  # recover

        return imgs

    def sample_frames(self, num_frames, vlen, sample='rand', fix_start=None):
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif fix_start is not None:
            frame_idxs = [x[0] + fix_start for x in ranges]
        elif sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError
        return frame_idxs


class EgovlpFeatureExtractor:
    def __init__(self, load_checkpoint_path="ckpt/egovlp.pth", device="cuda"):
        print("Loading EgoVLP models")
        model = FrozenInTime(**config)
        checkpoint = torch.load(load_checkpoint_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.device = device
        self.egovlp_extractor = model.to(self.device)
        self.num_frame = 4
        self.dim = 256


    @torch.no_grad()
    def encode_video(self, video_path: str):
        video_loader = VideoLoader()
        video_imgs = video_loader.read_frames_decord(video_path)

        f, c, h, w = video_imgs.shape
        video_imgs = video_imgs[:(f // self.num_frame * self.num_frame), ]
        video_imgs = video_imgs.reshape(-1, self.num_frame, c, h, w)
        video_imgs = video_imgs.to(self.device)
        video_features = torch.zeros(video_imgs.shape[0], self.dim)

        batch = 4
        times = video_imgs.shape[0] // batch
        for j in range(times):
            start = j * batch
            if (j + 1) * batch > video_imgs.shape[0]:
                end = video_imgs.shape[0]
            else:
                end = (j + 1) * batch

            video_features[start:end, ] = \
                self.egovlp_extractor.compute_video(video_imgs[start:end, ])

        return video_features  # (T=#frames, d) torch tensor

    @torch.no_grad()
    def encode_text(self, text_query: str):
        text_query_dict = self.tokenizer(text_query, return_tensors='pt', padding=True, truncation=True)
        text_query_dict = {key: val.cuda() for key, val in text_query_dict.items()}

        text_token_features = self.egovlp_extractor.compute_text_tokens(text_query_dict)[0]
        num_words = text_query_dict['attention_mask'][0].sum()
        text_token_features = text_token_features[1: num_words - 1]

        text_cls_feature = self.egovlp_extractor.compute_text(text_query_dict).squeeze()

        return text_token_features, text_cls_feature  # List([L_j, d]) torch tensor
