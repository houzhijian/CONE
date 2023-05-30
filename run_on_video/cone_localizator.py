import torch
import torch.nn.functional as F

import math
from easydict import EasyDict as edict

from cone.span_utils import span_cxw_to_xx
from cone.model import build_transformer, build_position_encoding, CONE

from run_on_video.temporal_nms import temporal_nms

opt = {
    "num_workers": 4,
    "v_motion_feat_dim": 256,
    "v_appear_feat_dim": 256,
    "t_feat_dim": 768,
    "ctx_mode": "video",
    "adapter_module": "linear",
    "position_embedding": "sine",
    "enc_layers": 2,
    "dec_layers": 2,
    "dim_feedforward": 1024,
    "hidden_dim": 256,
    "input_dropout": 0.5,
    "dropout": 0.1,
    "use_txt_pos": False,
    "nheads": 8,
    "num_queries": 5,
    "pre_norm": False,
    "n_input_proj": 2,
    "max_q_l": 20,
    "max_v_l": 90,
    "aux_loss": True,
    "span_loss_type": "l1",
    "topk_window": 20,
    "clip_length": 0.5333,
}


def normalize_score(pre_list):
    amin, amax = min(pre_list), max(pre_list)

    if amin == amax:
        return pre_list
    else:
        try:
            return [(val - amin) / (amax - amin) for val in pre_list]
        except:
            print(pre_list)
            return pre_list


args = edict(opt)


class CONELocalizator:
    def __init__(self, load_checkpoint_path="ckpt/model_best.ckpt", device="cuda"):
        print("Loading CONE models")
        transformer = build_transformer(args)
        position_embedding, txt_position_embedding = build_position_encoding(args)
        model = CONE(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_motion_dim=args.v_motion_feat_dim,
            vid_appear_dim=args.v_appear_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            span_loss_type=args.span_loss_type,
            adapter_module=args.adapter_module,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
        )
        checkpoint = torch.load(load_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        self.device = device
        self.localizator = model.to(self.device)
        self.slide_window_size = int(args.max_v_l / 2)
        self.max_v_l = args.max_v_l

    @torch.no_grad()
    def compute_window_ranklist(self, video_feats, text_cls_feat):
        frame_matching_score = torch.einsum('db,b->d', video_feats, text_cls_feat).detach().cpu()
        ctx_l = len(video_feats)
        num_window = math.ceil(ctx_l / self.slide_window_size) + 1

        # compute the matching score for each window
        window_score_list = []
        for i in range(num_window):
            new_start = max((i - 1) * self.slide_window_size, 0)
            new_end = min((i - 1) * self.slide_window_size + self.max_v_l, ctx_l)
            # pick the maximum frame matching score inside the window as the window-level matching score
            window_score = torch.max(frame_matching_score[new_start:new_end])
            window_score_list.append(window_score)

        window_score_tensor = torch.Tensor(window_score_list)
        scores, indices = torch.sort(window_score_tensor, descending=True)
        return indices.tolist()

    def pad_feature(self, feature, max_ctx_len):
        """
            Args:
                feature: original feature without padding
                max_ctx_len: the maximum length of video clips (or query token)

            Returns:
                 feat_pad : padded feature
                 feat_mask : feature mask
        """
        N_clip, feat_dim = feature.shape

        feat_pad = torch.zeros((max_ctx_len, feat_dim))
        feat_mask = torch.zeros(max_ctx_len, dtype=torch.long)
        feat_pad[:N_clip, :] = feature
        feat_mask[:N_clip] = 1

        return feat_pad, feat_mask

    @torch.no_grad()
    def predict_moment(self, video_feats, text_feats):
        """
        Args:
            video_feats: video_feats
            text_feats:  text_feats
        """

        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)

        text_token_feats, text_cls_feat = text_feats

        text_token_feats = F.normalize(text_token_feats, dim=-1, eps=1e-5)

        if args.adapter_module == "linear":
            adapter_video_feats = self.localizator.adapter_layer(video_feats) + video_feats
        else:
            adapter_video_feats = video_feats

        window_idx_ranklist = self.compute_window_ranklist(adapter_video_feats, text_cls_feat)

        windowidx = window_idx_ranklist[:args.topk_window]

        video_start_torch = torch.zeros(args.topk_window, dtype=torch.int)

        video_appear_feat_torch = torch.zeros((args.topk_window, args.max_v_l, args.v_appear_feat_dim),
                                              dtype=torch.float, device=self.device)
        video_feat_mask_torch = torch.zeros((args.topk_window, args.max_v_l), dtype=torch.long, device=self.device)

        query_feat_torch = torch.zeros((args.topk_window, args.max_q_l, args.t_feat_dim),
                                       dtype=torch.float, device=self.device)
        query_feat_mask_torch = torch.zeros((args.topk_window, args.max_q_l), dtype=torch.long, device=self.device)

        query_cls_feat_torch = torch.zeros((args.topk_window, args.v_appear_feat_dim),
                                           dtype=torch.float, device=self.device)

        for idx, window_idx in enumerate(windowidx):
            new_start = max((window_idx - 1) * self.slide_window_size, 0)
            new_end = min((window_idx - 1) * self.slide_window_size + self.max_v_l, len(video_feats))
            tmp_video_appearance_feat = video_feats[new_start:new_end, :]
            video_start_torch[idx] = new_start
            query_cls_feat_torch[idx] = text_cls_feat

            video_feat_pad, video_feat_mask = \
                self.pad_feature(tmp_video_appearance_feat, args.max_v_l)

            video_appear_feat_torch[idx] = video_feat_pad
            video_feat_mask_torch[idx] = video_feat_mask

            query_feat_pad, query_feat_mask = \
                self.pad_feature(text_token_feats, args.max_q_l)

            query_feat_torch[idx] = query_feat_pad
            query_feat_mask_torch[idx] = query_feat_mask

        outputs = self.localizator(query_feat_torch, query_feat_mask_torch, video_appear_feat_torch,
                                   video_feat_mask_torch)

        prob = F.softmax(outputs["pred_logits"], -1)

        matching_scores = self.localizator.forward_clip_matching(query_cls_feat_torch, video_appear_feat_torch,
                                                                 video_feat_mask_torch, proposal=outputs["pred_spans"])

        scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)

        total_predictions = []

        for idx, (spans, score, matching_score) in enumerate(
                zip(pred_spans.cpu(), scores.cpu(), matching_scores.cpu())):
            spans = (span_cxw_to_xx(spans) * args.max_v_l + video_start_torch[idx]) * args.clip_length

            # (#queries, 4), [st(float), ed(float), score(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None], matching_score[:, None]], dim=1).tolist()

            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

            total_predictions.extend(cur_ranked_preds)

        return_dict = {}
        proposal_score_list = [item[2] for item in total_predictions]
        matching_score_list = [item[3] for item in total_predictions]

        after_proposal_score_list = normalize_score(proposal_score_list)
        after_matching_score_list = normalize_score(matching_score_list)
        fusion_score_list = [sum(x) for x in zip(after_proposal_score_list, after_matching_score_list)]

        for item, score in zip(total_predictions, fusion_score_list):
            return_dict[(item[0], item[1])] = [item[2], item[3], score]

        predicted_moments = [[k[0], k[1], v[2]] for k, v in return_dict.items()]

        predicted_moments = sorted(predicted_moments, key=lambda x: x[2], reverse=True)  # descending order

        after_nms_predicted_moments = temporal_nms(
            predicted_moments[:100],
            nms_thd=0.5,
            max_after_nms=5,
        )

        return after_nms_predicted_moments
