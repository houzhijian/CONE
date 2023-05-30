import os.path
import torch
import time
import random
import numpy as np
from run_on_video.egovlp_extrator import EgovlpFeatureExtractor
from run_on_video.cone_localizator import CONELocalizator


class CONEPredictor:
    def __init__(self, extractor_ckpt_path, localizator_ckpt_path, save_path_dir, device="cuda"):
        self.device = device
        print("Loading feature extractors...")
        self.feature_extractor = EgovlpFeatureExtractor(extractor_ckpt_path, device=device)
        print("Loading trained Moment-DETR model...")
        self.save_path_dir = save_path_dir
        self.localizator = CONELocalizator(localizator_ckpt_path, device=device)

    @torch.no_grad()
    def load_video_feat(self, video_path):
        """
        Args:
            video_path: str, path to the video file
        """
        # construct model inputs
        video_name = os.path.basename(video_path).split(".")[0]
        print("video_name: ", video_name)
        cur_save_path = os.path.join(self.save_path_dir, video_name)
        os.makedirs(cur_save_path, exist_ok=True)
        feat_save_path = os.path.join(cur_save_path, video_name + ".pt")

        if os.path.exists(feat_save_path):
            video_feats = torch.load(feat_save_path)
        else:
            feature_extract_start_time = time.time()
            video_feats = self.feature_extractor.encode_video(video_path)
            torch.save(video_feats.detach().cpu(), feat_save_path)
            print("get feature time: ", time.time() - feature_extract_start_time)

        return video_feats

    def load_text_feat(self, text_input):
        """
        Args:
           text_input: str, user input text
        """
        print("text_query: ", text_input)
        text_feats = self.feature_extractor.encode_text(text_input)

        return text_feats

    def localize_moment(self, video_path, query_text):

        video_feats = self.load_video_feat(video_path).to(self.device)
        text_feats = self.load_text_feat(query_text)

        outputs = self.localizator.predict_moment(video_feats, text_feats)
        print("-----------------------------prediction------------------------------------")

        for idx, item in enumerate(outputs):
            print("Rank %d, moment boundary in seconds: %s %s, score: %s" % (idx+1, item[0], item[1], item[2]))


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def run_example():
    set_seed(2018)

    cone_ckpt_path = "./ckpt/model_best.ckpt"
    egovlp_ckpt_path = "./ckpt/egovlp.pth"

    # run predictions
    print("Build models...")
    cone_predictor = CONEPredictor(
        extractor_ckpt_path=egovlp_ckpt_path,
        localizator_ckpt_path=cone_ckpt_path,
        save_path_dir="save",
        device="cuda"
    )

    # load example data
    query_text = "Did I wash the green pepper?"
    video_path = "example/94cdabf3-c078-4ad4-a3a1-c42c8fc3f4ad.mp4"
    print("Run prediction...")
    cone_predictor.localize_moment(
        video_path=video_path, query_text=query_text)




if __name__ == "__main__":
    run_example()
