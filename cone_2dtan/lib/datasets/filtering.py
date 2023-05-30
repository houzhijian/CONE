import torch.utils.data as data
from core.config import config
from core.utils import l2_normalize_np_array
import numpy as np
import lmdb
import io
import os
import json
import torch


class PreFilteringDataset(data.Dataset):
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

    def __init__(self, split, data_mode="context"):
        self.data_mode = data_mode
        self.data_dir = config.DATA_DIR
        self.appearance_visual_env = lmdb.open(config.DATASET.APPEARANCE_FEAT_DIR, readonly=True, create=False,
                                               max_readers=4096 * 8, readahead=False)
        self.appearance_visual_txn = self.appearance_visual_env.begin(buffers=True)
        self.textual_env = lmdb.open(config.DATASET.Q_FEAT_DIR, readonly=True, create=False, max_readers=4096 * 8,
                                     readahead=False)
        self.textual_txn = self.textual_env.begin(buffers=True)

        # data
        with open(os.path.join(self.data_dir, '{}.jsonl'.format(split)), "r") as f:
            self.query_data = [json.loads(l.strip("\n")) for l in f.readlines()]

        self.video_data = list(set([item["clip_id"] for item in self.query_data]))
        self.video2idx = {v: idx for idx, v in enumerate(self.video_data)}


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
        #print(vid)
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
            try:
                cls_q_feat = q_dump['cls_features']
            except:
                cls_q_feat = q_dump['eot_features']
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