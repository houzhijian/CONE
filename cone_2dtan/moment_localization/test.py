import os
import math
import argparse
import pickle as pkl

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import _init_paths
from core.engine import Engine
import datasets
import models
from core.utils import AverageMeter
from core.config import config, update_config
from core.eval import eval_predictions, display_results
import models.loss as loss

# from moment_loclization.train import network, on_test_start, on_test_forward, on_test_end
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Test localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # testing
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--split', default='val', required=True, choices=['train', 'val', 'test', 'val_2'], type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--filtered_number', default=0, type=int, help='filtered window number')
    parser.add_argument('--batch_size', default=0, type=int, help='batch size')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.OUTPUT_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.filtered_number:
        config.TEST.WINDOW_TOP_K = args.filtered_number
    if args.batch_size:
        config.TEST.BATCH_SIZE = args.batch_size


if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME
    config.SAVE_PREDFILE_DIR = os.path.join(config.RESULT_DIR, config.DATASET.NAME,
                                            "_".join([args.split, config.MODEL.NAME, config.DATASET.OUTPUT_SUFFIX,
                                                      time.strftime("%Y_%m_%d_%H_%M_%S")]))

    # config.SAVE_MODELFILE_DIR = os.path.join(config.MODEL_DIR, dataset_name,
    #                                          "_".join([model_name, config.DATASET.OUTPUT_SUFFIX,
    #                                                    time.strftime("%Y_%m_%d_%H_%M_%S")]))
    if not os.path.exists(config.SAVE_PREDFILE_DIR):
        os.makedirs(config.SAVE_PREDFILE_DIR)
    #
    # if not os.path.exists(config.SAVE_MODELFILE_DIR):
    #     os.makedirs(config.SAVE_MODELFILE_DIR)
    # if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
    #     print("Resume the checkpoints....")
    #     print("from the path", config.MODEL.CHECKPOINT)
    #     model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    #     model.load_state_dict(model_checkpoint)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, config.MODEL.NAME)()
    print(model)
    print("Resume the checkpoints from the path", config.MODEL.CHECKPOINT)
    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    test_dataset = getattr(datasets, config.DATASET.NAME)(args.split)
    print(config.TEST.BATCH_SIZE)
    dataloader = DataLoader(test_dataset,
                            batch_size=config.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=False,
                            collate_fn=datasets.start_end_collate)


    def network(sample, epoch=0):

        pos_model_inputs, pos_clip_model_inputs, neg_model_inputs_tuple, targets \
            = datasets.prepare_batch_inputs(sample[1], sample[2], "cuda", non_blocking=True)

        prediction, map_mask = model(**pos_model_inputs)

        loss_value = None

        joint_prob = torch.sigmoid(prediction) * map_mask

        # start_time = time.time()

        sorted_proposal_span_score_list, sorted_proposal_list = get_proposal_results(joint_prob, sample[0])
        # print("proposal generation time: ",time.time() - start_time)

        if config.MODEL.USE_MATCHING_SCORE:
            if torch.cuda.device_count() > 1:
                matching_scores_list = model.module.forward_clip_matching(**pos_clip_model_inputs,
                                                                          proposal=sorted_proposal_list)
            else:
                matching_scores_list = model.forward_clip_matching(**pos_clip_model_inputs,
                                                                   proposal=sorted_proposal_list)

            total_prediction = []
            for (meta, pred_span_scores, matching_scores) in \
                    zip(sample[0], sorted_proposal_span_score_list, matching_scores_list):
                temp_matching_scores = torch.unsqueeze(matching_scores.cpu().detach(), 1)
                sorted_proposal_span_score = torch.cat([pred_span_scores, temp_matching_scores], dim=1)
                if not isinstance(sorted_proposal_span_score, list):
                    sorted_proposal_span_score = sorted_proposal_span_score.tolist()
                cur_query_pred = dict(
                    query_id=meta["query_id"],
                    query=meta["query"],
                    video_id=meta["video_id"],
                    clip_id=meta["clip_id"],
                    pred_relevant_windows=sorted_proposal_span_score,  # .tolist()
                )
                total_prediction.append(cur_query_pred)
        else:
            total_prediction = []
            for (meta, sorted_proposal_span_score) in \
                    zip(sample[0], sorted_proposal_span_score_list):
                if not isinstance(sorted_proposal_span_score, list):
                    sorted_proposal_span_score = sorted_proposal_span_score.tolist()
                cur_query_pred = dict(
                    query_id=meta["query_id"],
                    query=meta["query"],
                    video_id=meta["video_id"],
                    clip_id=meta["clip_id"],
                    pred_relevant_windows=sorted_proposal_span_score,
                )
                total_prediction.append(cur_query_pred)

        return loss_value, total_prediction


    def pre_filtering(eval_inter_window_dataset):
        assert eval_inter_window_dataset is not None
        #####
        # Inter-window Pre-filtering
        #####
        eval_inter_window_dataset.set_data_mode("context")
        eval_inter_window_context_loader = DataLoader(
            eval_inter_window_dataset,
            batch_size=1,
            num_workers=config.WORKERS,
            shuffle=False,
            pin_memory=False,
        )

        video_context_feat = []
        for batch in tqdm(eval_inter_window_context_loader, desc="compute video feat"):
            visual_feat = batch["model_inputs"]["video_feat"].to("cuda", non_blocking=True)
            # In practice, we also use the adapted feature for inter window pre-filtering
            if config.MODEL.ADAPTER == "linear":
                if torch.cuda.device_count() > 1:
                    adapted_appear_feat = model.module.adapter_layer(visual_feat) + visual_feat
                else:
                    adapted_appear_feat = model.adapter_layer(visual_feat) + visual_feat
                # adapted_appear_feat = model.adapter_layer(visual_feat) + visual_feat
                video_context_feat.append(adapted_appear_feat[0])
            else:
                video_context_feat.append(visual_feat[0])

        eval_inter_window_dataset.set_data_mode("query")

        eval_inter_window_query_loader = DataLoader(
            eval_inter_window_dataset,
            batch_size=1,
            num_workers=config.WORKERS,
            shuffle=False,
            pin_memory=False
        )
        max_v_l = config.DATASET.NUM_SAMPLE_CLIPS
        slide_window_size = int(config.DATASET.NUM_SAMPLE_CLIPS / 2)
        query_id2windowidx = dict()

        # compute the window matching rank-list for each query
        for batch in tqdm(eval_inter_window_query_loader, desc="compute window-level matching scores"):
            text_cls_feat = batch["model_inputs"]["query_feat"].to("cuda", non_blocking=True)[0]
            meta = batch['meta']

            query_id = meta['query_id'][0]
            video_id = meta['video_id'][0]
            idx = eval_inter_window_dataset.video2idx[video_id]
            vid_appear_feat = video_context_feat[idx]
            frame_matching_score = torch.einsum('db,b->d', vid_appear_feat, text_cls_feat).detach().cpu()
            ctx_l = len(vid_appear_feat)
            num_window = math.ceil(ctx_l / slide_window_size) + 1

            # compute the matching score for each window
            window_score_list = []
            for i in range(num_window):
                new_start = max((i - 1) * slide_window_size, 0)
                new_end = min((i - 1) * slide_window_size + max_v_l, ctx_l)
                # pick the maximum frame matching score inside the window as the window-level matching score
                window_score = torch.max(frame_matching_score[new_start:new_end])
                window_score_list.append(window_score)

            window_score_tensor = torch.Tensor(window_score_list)
            scores, indices = torch.sort(window_score_tensor, descending=True)
            query_id2windowidx[query_id] = indices.tolist()
        return query_id2windowidx


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


    def get_proposal_results(scores, meta_item):
        # assume all valid scores are larger than one

        out_sorted_times = []
        sorted_proposal_list = []
        for score, meta in zip(scores, meta_item):
            video_start = meta["video_start"]
            T = score.shape[-1]
            score_cpu = score.cpu().detach().numpy()

            sorted_indexs = np.dstack(
                np.unravel_index(np.argsort(score_cpu.ravel())[::-1], (T, T))).tolist()

            sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]])

            sorted_indexs[:, 1] = sorted_indexs[:, 1] + 1

            # sorted_indexs_after_nms = nms(sorted_indexs, thresh=config.TEST.NMS_THRESH_PROPOSAL,
            #                               top_k=config.TEST.PROPOSAL_TOP_K)
            if config.TEST.USE_NMS_WITHIN_WINDOW:
                sorted_indexs_after_nms = nms(sorted_indexs, thresh=config.TEST.NMS_THRESH_WITHIN_WINDOW,
                                              top_k=config.TEST.PROPOSAL_TOP_K)
            else:
                sorted_indexs_after_nms = sorted_indexs[:config.TEST.PROPOSAL_TOP_K]

            sorted_scores = torch.FloatTensor([score_cpu[0, x[0], x[1] - 1] for x in sorted_indexs_after_nms])

            sorted_indexs_after_nms = torch.from_numpy(sorted_indexs_after_nms * config.DATASET.TARGET_STRIDE)

            sorted_proposal_list.append(sorted_indexs_after_nms)

            sorted_time = (sorted_indexs_after_nms + video_start) * config.DATASET.CLIP_LEN

            sorted_proposal_span_score = torch.cat([sorted_time, torch.unsqueeze(sorted_scores, 1)], dim=1)

            # sorted_proposal_span_score = np.concatenate([sorted_time, sorted_scores[:, np.newaxis]], axis=1)[
            #                              :config.TEST.PROPOSAL_TOP_K]

            # sorted_proposal_span_score = [[t[0], t[1], s] for t, s in zip(sorted_time, sorted_scores)][
            #                              :config.TEST.PROPOSAL_TOP_K]

            out_sorted_times.append(sorted_proposal_span_score)

        return out_sorted_times, sorted_proposal_list


    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['sorted_segments_list'].extend(state['output'])


    # def on_test_end(state):
    #     annotations = state['iterator'].dataset.annotations
    #     state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations,
    #                                                                   state['epoch'], ,
    #                                                                   state['train_t'], verbose=True)
    #     if config.VERBOSE:
    #         state['progress_bar'].close()

    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        state['output'] = []
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset) / config.TEST.BATCH_SIZE))

        state['score_writer'] = open(
            os.path.join(config.SAVE_PREDFILE_DIR, "eval_results.txt"), mode="w", encoding="utf-8"
        )


    def on_test_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()
            print()

        annotations = test_dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval_predictions(state['sorted_segments_list'], annotations,
                                                                 state['epoch'], config.DATASET.NAME,
                                                                 0, verbose=True, split=args.split)
        # state['Rank@N,mIoU@M'], state['miou'] = eval_predictions(state['sorted_segments_list'], annotations, verbose=True)

        if args.split != "test" or config.DATASET.NAME != "Ego4d":
            loss_message = '\ntest loss {:.4f}'.format(state['loss_meter'].avg)
            print(loss_message)
            state['loss_meter'].reset()
            test_table = display_results(state['Rank@N,mIoU@M'], state['miou'],
                                         'performance on testing set')
            table_message = '\n' + test_table
            print(table_message)

            state["score_writer"].write(table_message)
            state["score_writer"].flush()

        # save_scores(state['sorted_segments_list'], annotations, config.DATASET.NAME, args.split)


    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    test_inter_window_dataset = getattr(datasets, "PreFilteringDataset")(args.split)
    query_id2windowidx = pre_filtering(test_inter_window_dataset)
    test_dataset.query_id2windowidx = query_id2windowidx
    engine.test(network, dataloader, args.split)
