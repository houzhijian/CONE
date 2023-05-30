from torch import nn
import torch
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules
import models.adapter_modules as adapter_modules


class CONE_TAN(nn.Module):
    def __init__(self):
        super(CONE_TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
        self.adapter_module = config.MODEL.ADAPTER
        print("self.adapter_module: ", self.adapter_module)
        if self.adapter_module == "linear":
            self.adapter_layer = getattr(adapter_modules, config.TAN.ADAPTER_MODULE.NAME)(
                config.TAN.ADAPTER_MODULE.PARAMS)

    def forward(self, textual_input, textual_mask, visual_input):

        vis_h = self.frame_layer(visual_input.transpose(1, 2))

        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, torch.unsqueeze(textual_mask, -1), map_h, map_mask)

        fused_h = self.map_layer(fused_h, map_mask)

        prediction = self.pred_layer(fused_h) * map_mask

        return prediction, map_mask

    def forward_clip_matching(self, src_cls_txt, src_vid_appear, src_vid_appear_mask, proposal=None,
                              is_groundtruth=False):
        """
        The forward expects following tensors:
            - src_cls_txt: [batch_size, D_txt]
            - src_vid_appear: [batch_size, L_vid, D_vid]
            - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
            - proposal:
            - is_groundtruth: whether the proposal comes from the ground-truth (during training)
            or proposal generation prediction (during inference).
        It returns a proposal-query similarity matrix.
        """
        text_cls_features = src_cls_txt / src_cls_txt.norm(dim=1, keepdim=True)

        if is_groundtruth:
            tgt_proposals = torch.vstack([t["proposal"][0] for t in proposal])  # (#spans, 2)
            proposal_feat = self._get_groundtruth_proposal_feat(src_vid_appear, tgt_proposals)
            proposal_features = proposal_feat / proposal_feat.norm(dim=1, keepdim=True)
            return torch.einsum('bd,ad->ba', proposal_features, text_cls_features)
        else:
            proposal_feat = self._get_predicted_proposal_feat(src_vid_appear, src_vid_appear_mask, proposal)
            proposal_features = proposal_feat / proposal_feat.norm(dim=2, keepdim=True)
            return torch.einsum('bld,bd->bl', proposal_features, text_cls_features)

    def _get_groundtruth_proposal_feat(self, src_vid_appear, groundtruth_proposal):
        """
        The forward expects following tensors:
           - src_vid_appear: [batch_size, L_vid, D_vid]
           - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
           - proposal: [batch_size, 2], ground-truth start and end timestamps
       It returns proposal features for ground-truth moments.
        """
        proposal_feat_list = []
        for idx, (feat, start_end_list) in enumerate(zip(src_vid_appear, groundtruth_proposal)):
            clip_feat = feat[start_end_list[0]:start_end_list[1]]
            # mean pooling inside each proposal
            proposal_feat_list.append(clip_feat.mean(axis=0))

        proposal_feat = torch.vstack(proposal_feat_list)

        # adapter module
        if self.adapter_module == "linear":
            proposal_feat = self.adapter_layer(proposal_feat) + proposal_feat
        else:
            proposal_feat = proposal_feat

        return proposal_feat

    def _get_predicted_proposal_feat(self, src_vid_appear, src_vid_appear_mask, proposal):
        """
        The forward expects following tensors:
          - src_vid_appear: [batch_size, L_vid, D_vid]
          - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
          - proposal: [batch_size, N_query, 2], predicted start and end timestamps for each moment queries
        It returns proposal features for predicted proposals.
        """
        vid_appear_dim = src_vid_appear.shape[2]

        torch.set_printoptions(profile="full")
        bsz = len(proposal)
        n_query = proposal[0].shape[0]
        torch.set_printoptions(profile="default")

        proposal_feat_list = []
        for idx, (feat, sorted_start_end_index) in enumerate(zip(src_vid_appear, proposal)):
            for start, end in zip(sorted_start_end_index[:, 0], sorted_start_end_index[:, 1]):
                clip_feat = feat[start:end]
                # mean pooling inside each proposal
                proposal_feat_list.append(clip_feat.mean(axis=0))
        proposal_feat = torch.stack(proposal_feat_list)

        # adapter module
        if self.adapter_module == "linear":
            proposal_feat = self.adapter_layer(proposal_feat) + proposal_feat
        else:
            proposal_feat = proposal_feat

        proposal_feat = proposal_feat.reshape(bsz, n_query, vid_appear_dim)

        return proposal_feat

    # def _get_predicted_proposal_feat(self, src_vid_appear, map_masks):
    #     """
    #     """
    #     # vid_appear_dim = src_vid_appear.shape[2]
    #     T = map_masks.shape[-1]
    #     bsz = map_masks.shape[0]
    #     map_masks = map_masks.squeeze()
    #     unsorted_indexs = torch.nonzero(map_masks).reshape(bsz, -1, 3)
    #     proposal_feat_list = []
    #     for vid_appear, unsorted_start_end_index in zip(src_vid_appear, unsorted_indexs):
    #         temp_proposal_feat_list = []
    #         for start, end in zip(unsorted_start_end_index[:, 1], unsorted_start_end_index[:, 2] + 1):
    #             clip_feat = vid_appear[start:end]
    #             temp_proposal_feat_list.append(clip_feat.mean(axis=0))
    #
    #         proposal_feat_list.append(torch.stack(temp_proposal_feat_list))
    #
    #     proposal_feat = torch.stack(proposal_feat_list)
    #
    #     # adapter module
    #     if self.adapter_module == "linear":
    #         proposal_feat = self.adapter_layer(proposal_feat) + proposal_feat
    #     else:
    #         proposal_feat = proposal_feat
    #
    #     return proposal_feat
