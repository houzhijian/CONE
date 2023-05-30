import torch
import torch.nn as nn
from core.config import config
from core.utils import pad_sequences_1d





def collate_fn(batch):
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]

    # for idx, item in enumerate(batch_map_gt):
    #     print("batch_word_vectors :", idx, item, item.shape)

    max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
    padded_batch_map_gt = torch.zeros(len(batch_map_gt), 1, max_num_clips, max_num_clips)
    for i, map_gt in enumerate(batch_map_gt):
        num_clips = map_gt.shape[-1]
        padded_batch_map_gt[i][0, :num_clips, :num_clips] = map_gt

    # for idx, item in enumerate(batch_word_vectors):
    #     print("batch_word_vectors :", idx, item, item.shape)

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        'batch_map_gt': padded_batch_map_gt,
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_duration': batch_duration,
    }

    #print("batch_map_gt :", padded_batch_map_gt, padded_batch_map_gt.shape)
    #print("batch_word_vectors :", batch_data["batch_word_vectors"], batch_data["batch_word_vectors"].shape)
    #exit(1)

    return batch_data

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
        if k in ["video_start", "video_length"]:
            batched_data[k] = torch.IntTensor([item[k] for e in batch for item in e['model_inputs']])
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
            batched_clip_data[k] = torch.vstack([item[k] for e in batch for item in e['model_clip_inputs']])
            continue
        seq = [item[k] for e in batch for item in e['model_clip_inputs']]
        batched_clip_data[k] = pad_sequences_1d(
            seq, dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data, batched_clip_data

def prepare_batch_inputs(batched_model_inputs, batched_clip_model_inputs, device, non_blocking=False):
    pos_model_inputs = dict(
        textual_input=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        textual_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        visual_input=batched_model_inputs["video_motion_feat"][0].to(device, non_blocking=non_blocking),
        #src_vid_motion_mask=batched_model_inputs["video_motion_feat"][1].to(device, non_blocking=non_blocking),
    )
    pos_clip_model_inputs = dict(
        src_cls_txt=batched_clip_model_inputs["query_cls_feat"].to(device, non_blocking=non_blocking),
        src_vid_appear=batched_clip_model_inputs["video_appear_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_appear_mask=batched_clip_model_inputs["video_appear_feat"][1].to(device, non_blocking=non_blocking),
    )
    if "neg_window_motion_feat" in batched_model_inputs:
        neg_model_inputs = dict(
            textual_input=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
            textual_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
            visual_input=batched_model_inputs["neg_window_motion_feat"][0].to(device, non_blocking=non_blocking),
            #src_vid_motion_mask=batched_model_inputs["neg_window_motion_feat"][1].to(device, non_blocking=non_blocking),
        )
        neg_clip_model_inputs = dict(
            src_cls_txt=batched_clip_model_inputs["query_cls_feat"].to(device, non_blocking=non_blocking),
            src_vid_appear=batched_clip_model_inputs["neg_window_appear_feat"][0].to(device,
                                                                                     non_blocking=non_blocking),
            src_vid_appear_mask=batched_clip_model_inputs["neg_window_appear_feat"][1].to(device, non_blocking=non_blocking),
        )
    else:
        neg_model_inputs = None
        neg_clip_model_inputs = None

    targets = None
    if "pos_overlaps" in batched_model_inputs:
        targets = dict(
            pos_overlaps_gt=batched_model_inputs["pos_overlaps"][0].to(device, non_blocking=non_blocking),
            neg_overlaps_gt=batched_model_inputs["neg_overlaps"][0].to(device, non_blocking=non_blocking),
        )
    # if "span_labels" in batched_model_inputs:
    #     targets["span_labels"] = [
    #         dict(spans=e["spans"].to(device, non_blocking=non_blocking))
    #         for e in batched_model_inputs["span_labels"]
    #     ]
    # if "saliency_pos_labels" in batched_model_inputs:
    #     for name in ["saliency_pos_labels", "saliency_neg_labels"]:
    #         targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)
    if "span_proposal" in batched_clip_model_inputs:
        targets["span_proposal"] = [
            dict(proposal=e["proposal"].to(device, non_blocking=non_blocking))
            for e in batched_clip_model_inputs["span_proposal"]
        ]

    # targets = None if len(targets) == 0 else targets
    return pos_model_inputs, pos_clip_model_inputs, (neg_model_inputs, neg_clip_model_inputs), targets


def average_to_fixed_length(visual_input):
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips
    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input


# from datasets.activitynet import ActivityNet
# from datasets.charades import Charades
# from datasets.tacos import TACoS
from datasets.ego4d import Ego4d
from datasets.filtering import PreFilteringDataset
from datasets.mad import MAD