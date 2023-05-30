####
# The only difference between ego4d_mad_dataloader.py is
# the implementation of the multi-scale variable-length sliding window sampling strategy (starting from line 180)
# We use this strategy for the workshop leaderboard submission, but not in the main paper.
####
import torch
import tqdm
from torch.utils.data import Dataset
import numpy as np
import io
import time
import lmdb
import math
import random
import logging
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from cone.span_utils import span_xx_to_cxw
from scipy.stats import norm

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "query_id": "ca7e11a2-cd1e-40dd-9d2f-ea810ab6a99b_0",
      "query": "what did I put in the black dustbin?",
      "video_id": "38737402-19bd-4689-9e74-3af391b15feb",
      "clip_id": "93231c7e-1cf4-4a20-b1f8-9cc9428915b2",
      "timestamps": [425.0, 431.0],
      "duration": 480,
    }
    """

    def __init__(self, dset_name, data_path, motion_feat_dir, appearance_feat_dir, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=20, max_v_l=90, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True, query_id2windowidx=None,
                 topk_window=30, clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0, is_eval=False):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.motion_feat_dir = motion_feat_dir
        self.appearance_feat_dir = appearance_feat_dir
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.topk_window = topk_window
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0
        self.slide_window_size = int(max_v_l / 2)
        self.eval = is_eval
        timer_start = time.time()

        self.appearance_visual_env = lmdb.open(self.appearance_feat_dir, readonly=True, create=False,
                                               max_readers=4096 * 8, readahead=False)
        self.appearance_visual_txn = self.appearance_visual_env.begin(buffers=True)

        self.same_visual_path = self.motion_feat_dir == self.appearance_feat_dir
        if not self.same_visual_path:
            self.motion_visual_env = lmdb.open(self.motion_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                               readahead=False)
            self.motion_visual_txn = self.motion_visual_env.begin(buffers=True)

        self.textual_env = lmdb.open(q_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                     readahead=False)
        self.textual_txn = self.textual_env.begin(buffers=True)
        print("load lmdb time:", time.time() - timer_start)
        print("clip_len:", self.clip_len)

        # data
        self.data = self.load_data()
        # the window rank-list computed by the contrastive pre-trained model
        self.query_id2windowidx = query_id2windowidx
        self.videofeat = self.load_video_feat()

    def load_video_feat(self):
        video_set = set([item['clip_id'] for item in self.data])
        video2feat = {}
        for video_id in tqdm.tqdm(video_set, desc="load video feat"):
            video_clip_feat = self._get_video_appearance_feat_by_vid(video_id)
            video2feat[video_id] = video_clip_feat
        return video2feat

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _meta = self.data[index]
        query_id = _meta["query_id"]
        model_inputs = []
        model_clip_inputs = []
        query_feat, query_cls_feat = self._get_query_feat_by_qid(_meta["query_id"])

        assert self.use_video
        video_clip_feat = self.videofeat[_meta["clip_id"]]
        if self.same_visual_path:
            video_motion_feat = video_clip_feat
        else:
            video_motion_feat = self._get_video_motion_feat_by_vid(_meta["clip_id"])

        ctx_l = len(video_clip_feat)
        assert ctx_l > 0, ctx_l
        num_window = math.ceil(ctx_l / self.slide_window_size) + 1

        if self.eval:
            windowidx = self.query_id2windowidx[query_id][:self.topk_window]
            for i in windowidx:
                new_start = max((i - 1) * self.slide_window_size, 0)
                new_end = min((i - 1) * self.slide_window_size + self.max_v_l, ctx_l)
                tmp_video_motion_feat = video_motion_feat[new_start:new_end, :]
                tmp_video_appearance_feat = video_clip_feat[new_start:new_end, :]
                model_inputs.append({
                    'video_motion_feat': tmp_video_motion_feat,
                    'query_feat': query_feat})
                model_clip_inputs.append({
                    'video_length': new_end - new_start,
                    'video_start': new_start,
                    'video_appear_feat': tmp_video_appearance_feat,
                    'query_cls_feat': query_cls_feat, })
        else:
            start = _meta["timestamps"][0] / self.clip_len
            end = _meta["timestamps"][1] / self.clip_len
            assert start < end, (end, start, _meta)
            start = min(ctx_l, start)
            end = min(ctx_l, end)

            matched_window_id_list = range(math.floor(start / self.slide_window_size),
                                           math.ceil(end / self.slide_window_size) + 1)
            assert len(matched_window_id_list), (_meta, ctx_l, _meta["timestamps"], matched_window_id_list)
            neg_window_pool = list(set(range(num_window)) - set(matched_window_id_list))
            assert len(neg_window_pool), (_meta, ctx_l, _meta["timestamps"], matched_window_id_list)

            matched_window_id_list = np.array(matched_window_id_list)
            temp_number = matched_window_id_list - matched_window_id_list.mean()
            temp_weight = norm.pdf(temp_number)
            weight = temp_weight / np.sum(temp_weight)
            idx = np.random.choice(matched_window_id_list, p=weight)

            new_start = max((idx - 1) * self.slide_window_size, 0)
            new_end = min((idx - 1) * self.slide_window_size + self.max_v_l, ctx_l)
            tmp_video_appearance_feat = video_clip_feat[new_start:new_end, :]
            tmp_model_clip_inputs = {
                'video_length': new_end - new_start,
                'video_start': new_start,
                'video_appear_feat': tmp_video_appearance_feat,
                'query_cls_feat': query_cls_feat, }

            # span_proposal ground-truth
            start_pos = max((idx - 1) * self.slide_window_size, start) - tmp_model_clip_inputs["video_start"]
            end_pos = min((idx - 1) * self.slide_window_size + self.max_v_l, end) - tmp_model_clip_inputs["video_start"]
            assert 0 <= math.floor(start_pos) < math.ceil(end_pos), [start, end, idx,
                                                                     tmp_model_clip_inputs["video_start"],
                                                                     start_pos, end_pos, _meta]
            tmp_model_clip_inputs.update(
                {'span_proposal': torch.IntTensor([[math.floor(start_pos), math.ceil(end_pos)]])})
            model_clip_inputs.append(tmp_model_clip_inputs)

            #####
            # Adopt multi-scale variable-length sliding window sampling strategy
            #####
            minimum_ratio_list = [0.4, 0.6, 0.8]
            maximum_ratio_list = [0.6, 0.8, 1]

            for i in range(3):
                # randomly choose the window length within a pre-defined range
                gt_moment_len = math.ceil(end - start)
                gt_ratio = gt_moment_len / self.slide_window_size
                min_ratio = min(minimum_ratio_list[i], max(maximum_ratio_list[i], gt_ratio))
                max_ratio = max(maximum_ratio_list[i] * 2, min(minimum_ratio_list[i] * 2, 2 * gt_ratio))
                sw_len_ratio = random.uniform(min_ratio, max_ratio)
                window_length = int(self.slide_window_size * 2 * sw_len_ratio)

                # randomly choose the window start and end timestamp
                rand_start_choice = max(0, math.ceil(end) - window_length)
                rand_end_choice = min(math.floor(start), ctx_l - window_length)

                if rand_start_choice < rand_end_choice:
                    new_start = random.randrange(rand_start_choice, rand_end_choice)
                elif rand_start_choice > rand_end_choice:
                    new_start = random.randrange(rand_end_choice, rand_start_choice)
                else:
                    new_start = rand_end_choice
                new_end = min(new_start + window_length, ctx_l)

                tmp_video_motion_feat = video_motion_feat[new_start:new_end, :]
                tmp_model_inputs = {
                    'video_length': new_end - new_start,
                    'video_start': new_start,
                    'video_motion_feat': tmp_video_motion_feat,
                    'query_feat': query_feat}

                start_pos = max(start - new_start, 0)
                end_pos = min(end - new_start, window_length)
                assert 0 <= math.floor(start_pos) < math.ceil(end_pos), [start, end,
                                                                         tmp_model_inputs["video_start"],
                                                                         start_pos, end_pos, _meta]
                tmp_span_labels = self.get_span_labels([[start_pos, end_pos]], tmp_model_inputs['video_length'])
                tmp_model_inputs.update({'span_labels': tmp_span_labels})

                rel_clip_ids = list(range(math.floor(start_pos), math.ceil(end_pos)))
                if not len(rel_clip_ids):
                    rel_clip_ids = [math.floor(start_pos)]
                easy_neg_pool = list(set(range(tmp_model_inputs['video_length'])) - set(rel_clip_ids))
                if not len(easy_neg_pool):
                    easy_neg_pool = [0]
                tmp_model_inputs.update({"saliency_pos_labels": random.sample(rel_clip_ids, k=1)})
                tmp_model_inputs.update({"saliency_neg_labels": random.sample(easy_neg_pool, k=1)})

                neg_window_id = random.choice(neg_window_pool)
                neg_start = max((neg_window_id - 1) * self.slide_window_size, 0)
                neg_end = min((neg_window_id - 1) * self.slide_window_size + self.max_v_l, ctx_l)
                tmp_model_inputs.update(
                    {"neg_window_motion_feat": video_motion_feat[neg_start:neg_end, :]})
                model_inputs.append(tmp_model_inputs)

        meta = []
        for idx in range(len(model_clip_inputs)):
            item = _meta.copy()
            item['duration'] = model_clip_inputs[idx]['video_length']
            item['video_start'] = model_clip_inputs[idx]['video_start']
            meta.append(item)

        return dict(meta=meta, model_inputs=model_inputs, model_clip_inputs=model_clip_inputs)

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / ctx_l  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        """
        qid: query_id
        returns both textual token feature and holistic text feature for each query
        """
        dump = self.textual_txn.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            q_feat = q_dump['token_features']
            cls_q_feat = q_dump['cls_features']
            if len(cls_q_feat.shape) == 2:
                cls_q_feat = cls_q_feat[0]

        if self.q_feat_type == "last_hidden_state":
            q_feat = q_feat[:self.max_q_l]

        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)

        cls_q_feat = l2_normalize_np_array(cls_q_feat)

        return torch.from_numpy(q_feat), cls_q_feat  # (Lq, D), (D, )

    def _get_video_motion_feat_by_vid(self, vid):
        dump = self.motion_visual_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump['features'].astype(np.float32)

        if self.normalize_v:
            _v_feat = l2_normalize_np_array(v_feat)
        return torch.from_numpy(_v_feat)  # (Lv, D)

    def _get_video_appearance_feat_by_vid(self, vid):
        dump = self.appearance_visual_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump['features']

        if self.normalize_v:
            _v_feat = l2_normalize_np_array(v_feat)
        return torch.from_numpy(v_feat)  # (Lv, D)


def start_end_collate(batch):
    batch_meta = [item for e in batch for item in e["meta"]]  # seems no need to collate ?
    for e in batch:
        if len(e["model_inputs"]):
            model_inputs_keys = e["model_inputs"][0].keys()
            break
    for e in batch:
        if len(e["model_clip_inputs"]):
            model_clip_inputs_keys = e["model_clip_inputs"][0].keys()
            break

    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=item["span_labels"]) for e in batch for item in e['model_inputs']]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([item[k] for e in batch for item in e['model_inputs']])
            continue
        if k in ["video_start", "video_length"]:
            continue

        seq = [item[k] for e in batch for item in e['model_inputs']]
        batched_data[k] = pad_sequences_1d(
            seq, dtype=torch.float32, fixed_length=None)

    batched_clip_data = dict()
    for k in model_clip_inputs_keys:
        if k in ["span_proposal"]:
            batched_clip_data[k] = [dict(proposal=item["span_proposal"]) for e in batch for item in
                                    e['model_clip_inputs']]
            continue
        if k in ["query_cls_feat"]:
            batched_clip_data[k] = torch.FloatTensor([item[k] for e in batch for item in e['model_clip_inputs']])
            continue

        if k in ["video_start", "video_length"]:
            continue

        seq = [item[k] for e in batch for item in e['model_clip_inputs']]
        batched_clip_data[k] = pad_sequences_1d(
            seq, dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data, batched_clip_data


def prepare_batch_inputs(batched_model_inputs, batched_clip_model_inputs, device, non_blocking=False):
    pos_model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid_motion=batched_model_inputs["video_motion_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_motion_mask=batched_model_inputs["video_motion_feat"][1].to(device, non_blocking=non_blocking),
    )
    pos_clip_model_inputs = dict(
        src_cls_txt=batched_clip_model_inputs["query_cls_feat"].to(device, non_blocking=non_blocking),
        src_vid_appear=batched_clip_model_inputs["video_appear_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_appear_mask=batched_clip_model_inputs["video_appear_feat"][1].to(device, non_blocking=non_blocking),
    )
    if "neg_window_motion_feat" in batched_model_inputs:
        neg_model_inputs = dict(
            src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
            src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
            src_vid_motion=batched_model_inputs["neg_window_motion_feat"][0].to(device, non_blocking=non_blocking),
            src_vid_motion_mask=batched_model_inputs["neg_window_motion_feat"][1].to(device,
                                                                                        non_blocking=non_blocking),
        )
    else:
        neg_model_inputs = None

    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)
    if "span_proposal" in batched_clip_model_inputs:
        targets["span_proposal"] = [
            dict(proposal=e["proposal"].to(device, non_blocking=non_blocking))
            for e in batched_clip_model_inputs["span_proposal"]
        ]

    targets = None if len(targets) == 0 else targets
    return pos_model_inputs, pos_clip_model_inputs, (neg_model_inputs, None), targets


class PreFilteringDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "query_id": "ca7e11a2-cd1e-40dd-9d2f-ea810ab6a99b_0",
      "query": "what did I put in the black dustbin?",
      "video_id": "38737402-19bd-4689-9e74-3af391b15feb",
      "clip_id": "93231c7e-1cf4-4a20-b1f8-9cc9428915b2",
      "timestamps": [425.0, 431.0],
      "duration": 480,
    }
    """

    def __init__(self, dset_name, data_path, appearance_feat_dir, q_feat_dir,
                 ctx_mode="video", data_mode="context", data_ratio=1):
        self.dset_name = dset_name
        self.data_ratio = data_ratio
        self.data_path = data_path
        self.appearance_feat_dir = appearance_feat_dir
        self.q_feat_dir = q_feat_dir
        self.data_mode = data_mode
        self.ctx_mode = ctx_mode
        self.use_video = "video" in ctx_mode

        timer_start = time.time()
        self.appearance_feat_dir = lmdb.open(self.appearance_feat_dir, readonly=True, create=False,
                                             max_readers=4096 * 8, readahead=False)
        self.appearance_visual_txn = self.appearance_feat_dir.begin(buffers=True)
        self.textual_env = lmdb.open(q_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                     readahead=False)
        self.textual_txn = self.textual_env.begin(buffers=True)
        print("load lmdb time:", time.time() - timer_start)
        # data
        self.query_data = self.load_data()
        self.video_data = list(set([item["clip_id"] for item in self.query_data]))
        self.video2idx = {v: idx for idx, v in enumerate(self.video_data)}

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def set_data_mode(self, data_mode):
        """context or query"""
        assert data_mode in ["context", "query"]
        self.data_mode = data_mode

    def __len__(self):
        if self.data_mode == "context":
            return len(self.video_data)
        else:
            return len(self.query_data)

    def _get_video_appearance_feat_by_vid(self, vid):
        dump = self.appearance_visual_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump['features']

        v_feat = l2_normalize_np_array(v_feat)
        return torch.from_numpy(v_feat)  # (Lv, D)

    def _get_query_feat_by_qid(self, qid):
        dump = self.textual_txn.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            cls_q_feat = q_dump['cls_features']
            if len(cls_q_feat.shape) == 2:
                cls_q_feat = cls_q_feat[0]
        cls_q_feat = l2_normalize_np_array(cls_q_feat)
        return cls_q_feat  # (D, )

    def __getitem__(self, index):
        if self.data_mode == "context":
            return self._get_item_context(index)
        else:
            return self._get_item_query(index)

    def _get_item_query(self, index):
        """Need to batch"""
        raw_data = self.query_data[index]

        meta = dict(
            query_id=raw_data["query_id"],
            query=raw_data["query"],
            video_id=raw_data["clip_id"]
        )

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta['query_id'])
        return dict(meta=meta, model_inputs=model_inputs)

    def _get_item_context(self, index):
        """No need to batch, since it has already been batched here"""
        video_id = self.video_data[index]

        # initialize with basic data
        meta = dict(
            video_id=video_id,
        )

        model_inputs = dict()
        model_inputs["video_feat"] = self._get_video_appearance_feat_by_vid(meta['video_id'])
        return dict(meta=meta, model_inputs=model_inputs)
