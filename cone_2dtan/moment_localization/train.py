from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter
from core import eval
from core.utils import create_logger
import models.loss as loss
import math
import time

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
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
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag


def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    config.SAVE_PREDFILE_DIR = os.path.join(config.RESULT_DIR, config.DATASET.NAME,
                                            "_".join([config.MODEL.NAME, config.DATASET.OUTPUT_SUFFIX,
                                                      time.strftime("%Y_%m_%d_%H_%M_%S")]))

    config.SAVE_MODELFILE_DIR = os.path.join(config.MODEL_DIR, dataset_name,
                                             "_".join([model_name, config.DATASET.OUTPUT_SUFFIX,
                                                       time.strftime("%Y_%m_%d_%H_%M_%S")]))

    if not os.path.exists(config.SAVE_PREDFILE_DIR):
        os.makedirs(config.SAVE_PREDFILE_DIR)

    if not os.path.exists(config.SAVE_MODELFILE_DIR):
        os.makedirs(config.SAVE_MODELFILE_DIR)

    train_dataset = getattr(datasets, dataset_name)('train', False)

    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val')
    test_dataset = getattr(datasets, dataset_name)('val')
    test_inter_window_dataset = getattr(datasets, "PreFilteringDataset")('val')

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        print("Resume the checkpoints....")
        print("from the path",config.MODEL.CHECKPOINT)
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    count_parameters(model)
    print(model)
    # print("config.TEST.USE_NEW_PROPOSAL_METHOD: ", config.TEST.USE_NEW_PROPOSAL_METHOD)
    # exit(1)

    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999),
                           weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR,
                                                     patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)


    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.start_end_collate)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.start_end_collate)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.start_end_collate)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.start_end_collate)
        else:
            raise NotImplementedError

        return dataloader


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


    def network(sample,epoch=0):

        pos_model_inputs, pos_clip_model_inputs, neg_model_inputs_tuple, targets \
            = datasets.prepare_batch_inputs(sample[1], sample[2], "cuda", non_blocking=True)

        prediction, map_mask = model(**pos_model_inputs)

        if model.training:
            loss_value, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask,
                                                                     torch.unsqueeze(targets["pos_overlaps_gt"], 1),
                                                                     config.LOSS.PARAMS)

            if config.TRAIN.CONTRASTIVE_LOSS:
                neg_prediction, neg_map_mask = model(**neg_model_inputs_tuple[0])
                neg_loss_value, neg_joint_prob = getattr(loss, config.LOSS.NAME)(neg_prediction, neg_map_mask,
                                                                                 torch.unsqueeze(
                                                                                     targets["neg_overlaps_gt"], 1),
                                                                                 config.LOSS.PARAMS)
                loss_value += neg_loss_value

            if config.TRAIN.ADAPTER_LOSS and epoch > config.TRAIN.ADAPTER_START_EPOCH:
                if torch.cuda.device_count() > 1:
                    logits_per_video = model.module.forward_clip_matching(**pos_clip_model_inputs,
                                                                          proposal=targets["span_proposal"],
                                                                          is_groundtruth=True)
                else:
                    logits_per_video = model.forward_clip_matching(**pos_clip_model_inputs,
                                                                   proposal=targets["span_proposal"],
                                                                   is_groundtruth=True)
                adapter_loss_value = getattr(loss, config.LOSS.ADAPTER_NAME)(logits_per_video, config)*config.TRAIN.ADAPTER_LOSS_WEIGHT
                loss_value += adapter_loss_value

            total_prediction = None
        else:
            loss_value = None

            joint_prob = torch.sigmoid(prediction) * map_mask

            # start_time = time.time()
            if config.TEST.USE_NEW_PROPOSAL_METHOD:
                sorted_proposal_span_score_list, sorted_proposal_list = get_proposal_results(joint_prob, sample[0],
                                                                                             map_mask)
            else:
                sorted_proposal_span_score_list, sorted_proposal_list = _get_proposal_results_ori(joint_prob, sample[0])
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


    def _get_proposal_results_ori(scores, meta_item):
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

            sorted_indexs_after_nms = torch.from_numpy(sorted_indexs_after_nms* config.DATASET.TARGET_STRIDE)

            sorted_proposal_list.append(sorted_indexs_after_nms)

            sorted_time = (sorted_indexs_after_nms + video_start)*config.DATASET.CLIP_LEN

            sorted_proposal_span_score = torch.cat([sorted_time, torch.unsqueeze(sorted_scores, 1)], dim=1)

            # sorted_proposal_span_score = np.concatenate([sorted_time, sorted_scores[:, np.newaxis]], axis=1)[
            #                              :config.TEST.PROPOSAL_TOP_K]

            # sorted_proposal_span_score = [[t[0], t[1], s] for t, s in zip(sorted_time, sorted_scores)][
            #                              :config.TEST.PROPOSAL_TOP_K]

            out_sorted_times.append(sorted_proposal_span_score)

        return out_sorted_times, sorted_proposal_list


    def get_proposal_results(scores, meta_item, map_masks):
        # assume all valid scores are larger than one

        proposal_span_score_list = []
        sorted_proposal_list = []
        for score, meta, map_mask in zip(scores, meta_item, map_masks):
            video_start = meta["video_start"]
            score_cpu = score.cpu().detach()
            map_mask_cpu = map_mask.squeeze().cpu().detach()
            unsorted_indexs = torch.nonzero(map_mask_cpu)

            unsorted_score = torch.Tensor([score_cpu[0, x[0], x[1]] for x in unsorted_indexs])

            _, sorted_indexs = torch.sort(unsorted_score, descending=True)

            unsorted_indexs[:, 1] = (unsorted_indexs[:, 1] + 1) * config.DATASET.TARGET_STRIDE
            unsorted_time = ((unsorted_indexs  + video_start) * config.DATASET.CLIP_LEN)

            unsorted_proposal_span_score = torch.cat([unsorted_time, torch.unsqueeze(unsorted_score, 1)], dim=1)

            sorted_proposal = unsorted_indexs[sorted_indexs]
            sorted_proposal_span_score = unsorted_proposal_span_score[sorted_indexs]

            sorted_proposal_list.append(sorted_proposal[:config.TEST.PROPOSAL_TOP_K])
            proposal_span_score_list.append(sorted_proposal_span_score[:config.TEST.PROPOSAL_TOP_K])

        return proposal_span_score_list, sorted_proposal_list


    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset) / config.TRAIN.BATCH_SIZE / config.TEST.INTERVAL)
        state['test_epoch_interval'] = config.TEST.EPOCH_INTERVAL
        state['dataset'] = config.DATASET.NAME
        # model.eval()
        # engine.test(network, iterator("test"), 'test', state['epoch'])
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])

        state['score_writer'] = open(
            os.path.join(config.SAVE_PREDFILE_DIR, "eval_results.txt"), mode="w", encoding="utf-8"
        )


    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)


    def on_update(state):  # Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0 and state['epoch'] % state['test_epoch_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\nepoch: {} iter: {} train loss {:.4f}'.format(state['epoch'], state['t'],
                                                                           state['loss_meter'].avg)
            table_message = ''

            query_id2windowidx = pre_filtering(test_inter_window_dataset)
            test_dataset.query_id2windowidx = query_id2windowidx
            test_state = engine.test(network, iterator("test"), 'test', state['epoch'],state['t'])
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                              'performance on testing set')
            table_message += '\n' + test_table

            message = loss_message + table_message + '\n'
            logger.info(message)

            state["score_writer"].write(message)
            state["score_writer"].flush()

            saved_model_filename = os.path.join(config.SAVE_MODELFILE_DIR,
                                                'iter{:06d}-{:.4f}-{:.4f}.pkl'.format(
                                                    state['t'], test_state['Rank@N,mIoU@M'][0, 0],
                                                    test_state['Rank@N,mIoU@M'][0, 1]))

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)

            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()


    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['dataset'] = config.DATASET.NAME
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset) / config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError


    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['sorted_segments_list'].extend(state['output'])


    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations,
                                                                      state['epoch'], state['dataset'],
                                                                      state['train_t'], verbose=True)
        if config.VERBOSE:
            state['progress_bar'].close()


    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end

    engine.train(network,
                 iterator("train"),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)
