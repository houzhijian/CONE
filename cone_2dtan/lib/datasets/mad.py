""" Dataset loader for the MAD dataset """
import os

import json
import torch
import torch.nn.functional as F
import torch.utils.data as data
from core.eval import iou
from core.config import config
from core.utils import l2_normalize_np_array
import numpy as np
import lmdb
import io
import math
import random
import tqdm
from scipy.stats import norm


class MAD(data.Dataset):
    def __init__(self, split, is_eval=True):
        super(MAD, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        with open(os.path.join(self.data_dir, '{}.jsonl'.format(split)), "r") as f:
            self.annotations = [json.loads(l.strip("\n")) for l in f.readlines()]
            # if is_eval:
            #     self.annotations = self.annotations[:30]

        self.appearance_visual_env = lmdb.open(config.DATASET.APPEARANCE_FEAT_DIR, readonly=True, create=False,
                                               max_readers=4096 * 8, readahead=False)
        self.appearance_visual_txn = self.appearance_visual_env.begin(buffers=True)

        self.textual_env = lmdb.open(config.DATASET.Q_FEAT_DIR, readonly=True, create=False, max_readers=4096 * 8,
                                     readahead=False)
        self.textual_txn = self.textual_env.begin(buffers=True)

        self.slide_window_size = int(config.DATASET.NUM_SAMPLE_CLIPS / 2)

        self.max_v_l = config.DATASET.NUM_SAMPLE_CLIPS

        self.eval = is_eval

        self.clip_len = config.DATASET.CLIP_LEN

        self.query_id2windowidx = None

        self.topk_window = config.TEST.WINDOW_TOP_K

        self.use_videodict = True

        if self.use_videodict:
            self.videofeat = self.load_video_feat()
        # print("self.clip_len: ",self.clip_len )

    def load_video_feat(self):
        video_set = set([item['clip_id'] for item in self.annotations])
        video2feat = {}
        for video_id in tqdm.tqdm(video_set, desc="load video feat"):
            video_clip_feat = self.get_video_features(video_id)
            video2feat[video_id] = video_clip_feat
        return video2feat

    def __getitem__(self, index):

        _meta = self.annotations[index]
        query_id = _meta["query_id"]
        video_id = _meta["clip_id"]

        # gt_s_time, gt_e_time = _meta["timestamps"]
        # print(" gt_s_time, gt_e_time: ", gt_s_time, gt_e_time)

        query_feat, query_cls_feat = self._get_query_feat_by_qid(query_id)

        if self.use_videodict:
            video_motion_feat = self.videofeat[_meta["clip_id"]]
        else:
            video_motion_feat = self.get_video_features(video_id)

        video_clip_feat = video_motion_feat

        ctx_l = len(video_motion_feat)
        assert ctx_l > 0, ctx_l
        num_window = math.ceil(ctx_l / self.slide_window_size) + 1

        model_inputs = []
        model_clip_inputs = []
        if self.eval:
            # select top-k windows for inference
            window_index_list = range(num_window)
            if self.query_id2windowidx is not None:
                window_index_list = self.query_id2windowidx[query_id][:self.topk_window]
            for i in window_index_list:
                new_start = max((i - 1) * self.slide_window_size, 0)
                new_end = min((i - 1) * self.slide_window_size + self.max_v_l, ctx_l)
                tmp_video_motion_feat = video_motion_feat[new_start:new_end, :]
                tmp_video_appearance_feat = video_clip_feat[new_start:new_end, :]
                model_inputs.append({
                    'video_length': new_end - new_start,
                    'video_start': new_start,
                    'video_motion_feat': tmp_video_motion_feat,
                    'query_feat': query_feat})
                model_clip_inputs.append({
                    'video_appear_feat': tmp_video_appearance_feat,
                    'query_cls_feat': query_cls_feat, })
        else:
            # calculate the positive window list for training
            start = _meta["timestamps"][0] / self.clip_len
            end = _meta["timestamps"][1] / self.clip_len

            assert start < end, (end, start, _meta)
            start = min(ctx_l, start)
            end = min(ctx_l, end)
            pos_window_id_list = range(math.floor(start / self.slide_window_size),
                                       math.ceil(end / self.slide_window_size) + 1)
            assert len(pos_window_id_list), (_meta, ctx_l, _meta["timestamps"], pos_window_id_list)

            neg_window_pool = list(set(range(num_window)) - set(pos_window_id_list))
            assert len(neg_window_pool), (_meta, ctx_l, _meta["timestamps"], pos_window_id_list)

            ####
            # There are at least two positive windows for each query, we choose a strategy to choose more
            # middle-position window because it is more likely to cover the whole duration rather than partial coverage.
            ####
            pos_window_id_list = np.array(pos_window_id_list)
            temp_number = pos_window_id_list - pos_window_id_list.mean()
            temp_weight = norm.pdf(temp_number)
            weight = temp_weight / np.sum(temp_weight)
            idx = np.random.choice(pos_window_id_list, p=weight)

            new_start = max((idx - 1) * self.slide_window_size, 0)
            new_end = min((idx - 1) * self.slide_window_size + self.max_v_l, ctx_l)
            tmp_video_motion_feat = video_motion_feat[new_start:new_end, :]
            tmp_video_appearance_feat = video_clip_feat[new_start:new_end, :]
            tmp_model_inputs = {
                'video_length': new_end - new_start,
                'video_start': new_start,
                'video_motion_feat': tmp_video_motion_feat,
                'query_feat': query_feat}
            tmp_model_clip_inputs = {
                'video_appear_feat': tmp_video_appearance_feat,
                'query_cls_feat': query_cls_feat, }

            # span_proposal ground-truth
            start_pos = max((idx - 1) * self.slide_window_size, start) - tmp_model_inputs["video_start"]
            end_pos = min((idx - 1) * self.slide_window_size + self.max_v_l, end) - tmp_model_inputs["video_start"]
            normalized_start_pos = start_pos / config.DATASET.TARGET_STRIDE
            normalized_end_pos = end_pos / config.DATASET.TARGET_STRIDE
            num_clips = self.max_v_l // config.DATASET.TARGET_STRIDE
            s_times = torch.arange(0, num_clips).float()
            e_times = torch.arange(1, num_clips + 1).float()

            pos_overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                            e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                               torch.tensor([normalized_start_pos, normalized_end_pos]).tolist()).reshape(num_clips, num_clips)
            pos_overlaps = torch.from_numpy(pos_overlaps)
            tmp_model_inputs.update({'pos_overlaps': pos_overlaps})

            # torch.set_printoptions(profile="full")
            # if torch.any(torch.isnan(pos_overlaps)):
            #     print("_meta: ", _meta)
            #     print("pos_window_id_list: ", pos_window_id_list)
            #     print("idx: ", idx)
            #     print(" gt_s_time, gt_e_time: ", gt_s_time, gt_e_time)
            #     print("start end: ", start, end)
            #     print("start_pos end_pos: ", start_pos, end_pos)
            #     print("normalized_start_pos, normalized_end_pos: ", normalized_start_pos, normalized_end_pos)
            #     print("pos_overlaps: ", pos_overlaps)
            #     exit(1)
            # torch.set_printoptions(profile="default")
            #

            tmp_model_clip_inputs.update(
                {'span_proposal': torch.IntTensor([[math.floor(start_pos), math.ceil(end_pos)]])})

            # Randomly choose one negative window
            neg_window_id = random.choice(neg_window_pool)
            neg_start = max((neg_window_id - 1) * self.slide_window_size, 0)
            neg_end = min((neg_window_id - 1) * self.slide_window_size + self.max_v_l, ctx_l)
            tmp_model_inputs.update({'neg_overlaps': torch.zeros_like(pos_overlaps)})
            tmp_model_inputs.update(
                {"neg_window_motion_feat": video_motion_feat[neg_start:neg_end, :]})
            tmp_model_clip_inputs.update(
                {"neg_window_appear_feat": video_clip_feat[neg_start:neg_end, :]})

            model_clip_inputs.append(tmp_model_clip_inputs)
            model_inputs.append(tmp_model_inputs)

        meta = []
        for idx in range(len(model_inputs)):
            item = _meta.copy()
            item['duration'] = model_inputs[idx]['video_length']
            item['video_start'] = model_inputs[idx]['video_start']
            meta.append(item)

        return dict(meta=meta, model_inputs=model_inputs, model_clip_inputs=model_clip_inputs)

    def __len__(self):
        return len(self.annotations)

    def _get_query_feat_by_qid(self, qid):
        """
        qid: query_id
        returns both textual token feature and holistic text feature for each query
        """
        dump = self.textual_txn.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            q_feat = q_dump['token_features']
            try:
                cls_q_feat = q_dump['cls_features']
            except:
                cls_q_feat = q_dump['eot_features']
            if len(cls_q_feat.shape) == 2:
                cls_q_feat = cls_q_feat[0]

        if config.DATASET.NORMALIZE:
            q_feat = l2_normalize_np_array(q_feat)

        return torch.from_numpy(q_feat), torch.from_numpy(cls_q_feat)  # (Lq, D), (D, )

    def get_video_features(self, vid):
        dump = self.appearance_visual_txn.get(vid.encode())
        #print(vid)
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            features = torch.from_numpy(img_dump['features'])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)

        return features
