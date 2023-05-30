import json
import argparse
import numpy as np
from terminaltables import AsciiTable
import os
from core.config import config, update_config


def iou(pred, gt):  # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def rank(pred, gt):
    return pred.index(gt) + 1


def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0, len(dets), 1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


def eval(segments, data):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL, str) else [
        config.TEST.RECALL]

    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    for seg, dat in zip(segments, data):
        seg = nms(seg, thresh=config.TEST.NMS_THRESH, top_k=max_recall).tolist()
        overlap = iou(seg, [dat['timestamps']])
        average_iou.append(np.mean(np.sort(overlap[0])[-3:]))

        for i, t in enumerate(tious):
            for j, r in enumerate(recalls):
                eval_result[i][j].append((overlap > t)[:r].any())
    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)

    return eval_result, miou


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


def score_fusion(prediction):
    return_dict = {}
    proposal_score_list = [item[2] for item in prediction]
    matching_score_list = [item[3] for item in prediction]

    after_proposal_score_list = normalize_score(proposal_score_list)
    after_matching_score_list = normalize_score(matching_score_list)
    fusion_score_list = [sum(x) for x in zip(after_proposal_score_list, after_matching_score_list)]

    for item, score in zip(prediction, fusion_score_list):
        return_dict[(item[0], item[1])] = [item[2], item[3], score]

    return return_dict


def post_processing_mr_nms(value, key_idx=2, verbose=False):
    # output = value.copy()
    if verbose:
        print("value[predicted_times]: ", type(value["predicted_times"]),
              len(value["predicted_times"]), value["predicted_times"][:10])
    output_segment_before_nms = sorted(value["predicted_times"], key=lambda x: x[key_idx], reverse=True)

    output_segment_after_nms = nms(output_segment_before_nms, thresh=config.TEST.NMS_THRESH,
                                   top_k=config.TEST.NMS_TOP_K)
    if verbose:
        print("fusion_segment_before_nms: ", len(output_segment_before_nms), output_segment_before_nms[:10])
        print("fusion_segment_after_nms: ", output_segment_after_nms)
    return output_segment_after_nms


def eval_predictions(segments, data, epoch, dataset, train_t=0, verbose=True, split='val'):
    qid2result = {}
    if dataset == "Ego4d":
        for item in segments:
            qid = item['query_id']
            if qid not in qid2result:
                temp_list = qid.split("_")
                assert len(temp_list) == 2
                qid2result[qid] = {
                    'query_idx': int(temp_list[1]),
                    'annotation_uid': temp_list[0],
                    'predicted_times': [],
                    'clip_uid': item['clip_id'],
                }
            pred_windows = item['pred_relevant_windows']
            qid2result[qid]['predicted_times'].extend(pred_windows)
    else:
        for item in segments:
            qid = item['query_id']
            if qid not in qid2result:
                qid2result[qid] = {
                    'query_id': qid,
                    'predicted_times': [],
                    'video_id': item['video_id'],
                }
            pred_windows = item['pred_relevant_windows']
            qid2result[qid]['predicted_times'].extend(pred_windows)

    fusion_segments = []
    proposal_segments = []
    matching_segments = []
    fusion_results = []
    proposal_results = []
    matching_results = []
    for idx, (qid, value) in enumerate(qid2result.items()):
        if config.MODEL.USE_MATCHING_SCORE:
            value['predicted_times'] = [list(_key) + _value
                                        for _key, _value in score_fusion(value['predicted_times']).items()]

            fusion_output = value.copy()
            # fusion_segment_before_nms = sorted(fusion_output["predicted_times"], key=lambda x: x[4], reverse=True)
            # fusion_segment_after_nms = nms(fusion_segment_before_nms, thresh=config.TEST.NMS_THRESH,
            #                                top_k=config.TEST.TOP_K)
            if idx == 1:
                fusion_segment_after_nms = post_processing_mr_nms(fusion_output, key_idx=4, verbose=True)
            else:
                fusion_segment_after_nms = post_processing_mr_nms(fusion_output, key_idx=4)
            # fusion_segment_after_nms = post_processing_mr_nms(fusion_output, key_idx=4)
            fusion_segments.append(fusion_segment_after_nms)
            fusion_output["predicted_times"] = fusion_segment_after_nms.tolist()
            fusion_results.append(fusion_output)

            proposal_output = value.copy()
            proposal_segment_after_nms = post_processing_mr_nms(proposal_output, key_idx=2)
            # proposal_segment_before_nms = sorted(proposal_output["predicted_times"], key=lambda x: x[2], reverse=True)
            # proposal_segment_after_nms = nms(proposal_segment_before_nms, thresh=config.TEST.NMS_THRESH,
            #                                  top_k=config.TEST.TOP_K)
            proposal_segments.append(proposal_segment_after_nms)
            proposal_output["predicted_times"] = proposal_segment_after_nms.tolist()
            proposal_results.append(proposal_output)

            matching_output = value.copy()
            matching_segment_after_nms = post_processing_mr_nms(matching_output, key_idx=3)
            # matching_segment_before_nms = sorted(matching_output["predicted_times"], key=lambda x: x[3], reverse=True)
            # matching_segment_after_nms = nms(matching_segment_before_nms, thresh=config.TEST.NMS_THRESH,
            #                                  top_k=config.TEST.TOP_K)
            matching_segments.append(matching_segment_after_nms)
            matching_output["predicted_times"] = matching_segment_after_nms.tolist()
            matching_results.append(matching_output)
        else:
            fusion_output = value.copy()
            if idx == 1:
                fusion_segment_after_nms = post_processing_mr_nms(fusion_output, key_idx=2, verbose=True)
            else:
                fusion_segment_after_nms = post_processing_mr_nms(fusion_output, key_idx=2)

            fusion_segments.append(fusion_segment_after_nms)
            fusion_output["predicted_times"] = fusion_segment_after_nms.tolist()
            fusion_results.append(fusion_output)

    if dataset == "Ego4d":
        with open(os.path.join(config.SAVE_PREDFILE_DIR,
                               "{}_epoch_{}_iter_{}_fusion_submission.json".format(split,epoch, train_t)),
                  "w") as file_id:
            json.dump(
                {
                    "version": "1.0",
                    "challenge": "ego4d_nlq_challenge",
                    "results": fusion_results,
                }, file_id
            )

        if config.MODEL.USE_MATCHING_SCORE and config.SVAE_ALL:
            with open(os.path.join(config.SAVE_PREDFILE_DIR,
                                   "{}_epoch_{}_iter_{}_proposal_submission.json".format(split,epoch, train_t)),
                      "w") as file_id:
                json.dump(
                    {
                        "version": "1.0",
                        "challenge": "ego4d_nlq_challenge",
                        "results": proposal_results,
                    }, file_id
                )
            with open(os.path.join(config.SAVE_PREDFILE_DIR,
                                   "{}_epoch_{}_iter_{}_matching_submission.json".format(split,epoch, train_t)),
                      "w") as file_id:
                json.dump(
                    {
                        "version": "1.0",
                        "challenge": "ego4d_nlq_challenge",
                        "results": matching_results,
                    }, file_id
                )
    else:
        # save prediction file to disk
        with open(os.path.join(config.SAVE_PREDFILE_DIR,
                               "{}_epoch_{}_iter_{}_fusion_submission.jsonl".format(split,epoch, train_t)), "w") as f:
            f.write("\n".join([json.dumps(e) for e in fusion_results]))
        if config.MODEL.USE_MATCHING_SCORE and config.SVAE_ALL:
            with open(os.path.join(config.SAVE_PREDFILE_DIR,
                                   "{}_epoch_{}_iter_{}_proposal_submission.jsonl".format(split,epoch, train_t)),
                      "w") as f:
                f.write("\n".join([json.dumps(e) for e in proposal_results]))
            with open(os.path.join(config.SAVE_PREDFILE_DIR,
                                   "{}_epoch_{}_iter_{}_matching_submission.jsonl".format(split,epoch, train_t)),
                      "w") as f:
                f.write("\n".join([json.dumps(e) for e in matching_results]))

    fusion_eval_result = None
    fusion_miou = None
    if dataset != "Ego4d" or split != "test":
        fusion_eval_result, fusion_miou = eval(fusion_segments, data)
        if verbose:
            print(display_results(fusion_eval_result, fusion_miou, 'Fusion'))

        if config.MODEL.USE_MATCHING_SCORE:
            proposal_eval_result, proposal_miou = eval(proposal_segments, data)
            matching_eval_result, matching_miou = eval(matching_segments, data)
            print(display_results(proposal_eval_result, proposal_miou, 'Proposal'))
            print(display_results(matching_eval_result, matching_miou, 'Matching'))

    return fusion_eval_result, fusion_miou


# def eval_predictions(segments, data, verbose=True):
#     eval_result, miou = eval(segments, data)
#     if verbose:
#         print(display_results(eval_result, miou, ''))
#
#     return eval_result, miou

def display_results(eval_result, miou, title=None):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL, str) else [
        config.TEST.RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i, j) for i in recalls for j in tious] + ['mIoU']]
    eval_result = eval_result * 100
    miou = miou * 100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        + ['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious) * len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.verbose:
        config.VERBOSE = args.verbose


if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)
    train_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/train_data.json', 'r'))
    val_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/val_data.json', 'r'))

    moment_frequency_dict = {}
    for d in train_data:
        times = [t for t in d['timestamps']]
        for time in times:
            time = tuple(time)
            if time not in moment_frequency_dict.keys():
                moment_frequency_dict[time] = 0
            moment_frequency_dict[time] += 1

    prior = sorted(moment_frequency_dict, key=moment_frequency_dict.get, reverse=True)
    prior = [list(item) for item in prior]
    prediction = [prior for d in val_data]

    eval_predictions(prediction, val_data)
