from tqdm import tqdm
from collections import defaultdict
import time
import os
import math
import torch
import logging
import json
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from cone.config import TestOptions
from cone.model import build_model
from cone.span_utils import span_cxw_to_xx
from cone.ego4d_mad_dataloader import prepare_batch_inputs, StartEndDataset, start_end_collate, PreFilteringDataset
from utils.basic_utils import save_jsonl, normalize_score, load_jsonl, AverageMeter
from utils.temporal_nms import temporal_nms
import standalone_eval.evaluate_pre_filtered_window as window_eval
import standalone_eval.evaluate_mad as mad_eval
import standalone_eval.evaluate_ego4d_nlq as ego4d_eval

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, model_clip_inputs, _, targets = prepare_batch_inputs(batch[1], batch[2], opt.device,
                                                                           non_blocking=opt.pin_memory)

        outputs = model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)

        matching_scores = model.forward_clip_matching(**model_clip_inputs,proposal=outputs["pred_spans"])

        if opt.span_loss_type == "l1":
            scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)
            # _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
            # saliency_scores = []
            # valid_vid_lengths = model_inputs["src_vid_motion_mask"].sum(1).cpu().tolist()
            # for j in range(len(valid_vid_lengths)):
            #     valid_saliency_scores = _saliency_scores[j, :int(valid_vid_lengths[j])]
            #     saliency_scores.append(valid_saliency_scores.tolist())
        else:
            bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
            # TODO use more advanced decoding method with st_ed product
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions
        for idx, (meta, spans, score, matching_score) in enumerate(
                zip(query_meta, pred_spans.cpu(), scores.cpu(), matching_scores.cpu())):

            if opt.span_loss_type == "l1":
                if "video_start" in meta:
                    spans = (span_cxw_to_xx(spans) * meta["duration"] + meta["video_start"]) * opt.clip_length
                else:
                    spans = (span_cxw_to_xx(spans) * meta["duration"]) * opt.clip_length

            # (#queries, 4), [st(float), ed(float), score(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None], matching_score[:, None]], dim=1).tolist()
            if not opt.no_sort_results:
                cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query_id=meta["query_id"],
                query=meta["query"],
                video_id=meta["video_id"],
                clip_id=meta["clip_id"],
                pred_relevant_windows=cur_ranked_preds,
            )
            mr_res.append(cur_query_pred)

        if opt.debug:
            break

    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    return mr_res, loss_meters


def post_processing_mr_nms(opt, return_dict, idx):
    predicted_moments = [[k[0], k[1], v[idx]] for k, v in return_dict.items()]

    predicted_moments = sorted(predicted_moments, key=lambda x: x[2], reverse=True)  # descending order

    before_nms_output = [[_item[0], _item[1]] + return_dict[(_item[0], _item[1])] for _item in predicted_moments]

    if opt.nms_thd != -1:
        # logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        after_nms_predicted_moments = temporal_nms(
            predicted_moments[:opt.max_before_nms],
            nms_thd=opt.nms_thd,
            max_after_nms=opt.max_after_nms
        )
        if opt.eval_split_name == "val":
            after_nms_output = [[_item[0], _item[1]] + return_dict[(_item[0], _item[1])]
                                for _item in after_nms_predicted_moments]
        elif opt.eval_split_name == "test":
            after_nms_output = [[_item[0], _item[1]] + return_dict[(_item[0], _item[1])]
                                for _item in after_nms_predicted_moments]
            # Add those " + return_dict[(_item[0], _item[1])] " for ensemble

        return after_nms_output
    else:
        return before_nms_output[:opt.max_after_nms]


def postprocessing_format_ego4d(submission, opt):
    qid2result = {}
    for item in submission:
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

    results = list(qid2result.values())
    fusion_results = []
    proposal_results = []
    matching_results = []

    for item in tqdm(results, desc="compute nms"):
        return_dict = score_fusion(item['predicted_times'])

        fusion_output = item.copy()
        fusion_output["predicted_times"] = post_processing_mr_nms(opt, return_dict, idx=2)
        fusion_results.append(fusion_output)

        proposal_output = item.copy()
        proposal_output["predicted_times"] = post_processing_mr_nms(opt, return_dict, idx=0)
        proposal_results.append(proposal_output)

        matching_output = item.copy()
        matching_output["predicted_times"] = post_processing_mr_nms(opt, return_dict, idx=1)
        matching_results.append(matching_output)

    return fusion_results, proposal_results, matching_results


def postprocessing_format_mad(submission, opt):
    qid2result = {}
    for item in submission:
        qid = item['query_id']
        if qid not in qid2result:
            qid2result[qid] = {
                'query_id': qid,
                'predicted_times': [],
                'video_id': item['video_id'],
            }
        pred_windows = item['pred_relevant_windows']
        qid2result[qid]['predicted_times'].extend(pred_windows)

    results = list(qid2result.values())
    fusion_results = []
    proposal_results = []
    matching_results = []

    for item in tqdm(results, desc="compute nms"):
        return_dict = score_fusion(item['predicted_times'])

        fusion_output = item.copy()
        fusion_output["predicted_times"] = post_processing_mr_nms(opt, return_dict, idx=2)
        fusion_results.append(fusion_output)

        proposal_output = item.copy()
        proposal_output["predicted_times"] = post_processing_mr_nms(opt, return_dict, idx=0)
        proposal_results.append(proposal_output)

        matching_output = item.copy()
        matching_output["predicted_times"] = post_processing_mr_nms(opt, return_dict, idx=1)
        matching_results.append(matching_output)

    return fusion_results, proposal_results, matching_results


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


def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion,
                                                    tb_writer)  # list(dict)
    return eval_res, eval_loss_meters


def eval_epoch(model, eval_inter_window_dataset, eval_intra_window_dataset, opt, save_submission_filename, epoch_i=None,
               criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_intra_window_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    start_time = time.time()

    #####
    # Inter-window Pre-filtering
    #####
    eval_inter_window_dataset.set_data_mode("context")
    eval_inter_window_context_loader = DataLoader(
        eval_inter_window_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    video_context_feat = []
    for batch in tqdm(eval_inter_window_context_loader, desc="compute video feat"):
        visual_feat = batch["model_inputs"]["video_feat"].to(opt.device, non_blocking=True)
        # In practice, we also use the adapted feature for inter window pre-filtering
        if opt.adapter_module == "linear":
            adapted_appear_feat = model.adapter_layer(visual_feat) + visual_feat
            ##Todo normalize adapted video features
            adapted_appear_feat = adapted_appear_feat / adapted_appear_feat.norm(dim=2, keepdim=True)
            video_context_feat.append(adapted_appear_feat[0])
        else:
            video_context_feat.append(visual_feat[0])

    eval_inter_window_dataset.set_data_mode("query")

    eval_inter_window_query_loader = DataLoader(
        eval_inter_window_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )
    max_v_l = opt.max_v_l
    slide_window_size = int(opt.max_v_l / 2)
    query_id2windowidx = dict()

    # compute the window matching rank-list for each query
    for batch in tqdm(eval_inter_window_query_loader, desc="compute window-level matching scores"):
        text_cls_feat = batch["model_inputs"]["query_feat"].to(opt.device, non_blocking=True)[0]
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

    eval_intra_window_dataset.query_id2windowidx = query_id2windowidx

    #####
    # Intra-window Fine-grained Ranking
    #####
    eval_loader = DataLoader(
        eval_intra_window_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)

    print("total model running time: ", time.time() - start_time)

    latest_file_paths = []
    if opt.dset_name == "mad":
        # Post-processing for MAD dataset
        submission, submission_proposal, submission_matching = postprocessing_format_mad(submission, opt)
        submission_path = os.path.join(opt.results_dir, save_submission_filename)
        # save prediction file to disk
        save_jsonl(submission, submission_path)
        if opt.save_all or opt.eval_modality != "both":
            submission_proposal_path = submission_path.replace("preds", "proposal_preds")
            save_jsonl(submission_proposal, submission_proposal_path)

            submission_matching_path = submission_path.replace("preds", "matching_preds")
            save_jsonl(submission_matching, submission_matching_path)
        # performance evaluation
        if opt.eval_split_name in ["val", "test"]:
            ground_truth = load_jsonl(opt.eval_path)
            thresholds = torch.tensor([0.1, 0.3, 0.5])
            topK = torch.tensor([1, 5, 10, 50, 100])

            # window pre-filtering recall
            window_ranklist_results = window_eval.windows_selection(
                query_id2windowidx, ground_truth, torch.tensor([1, 5, 10, 30, 50, 100, 200]), opt
            )
            title = f"Window Pre-filtering Epoch {epoch_i}"
            window_score_str = window_eval.display_window_results(
                window_ranklist_results, torch.tensor([1, 5, 10, 30, 50, 100, 200]), title=title
            )
            print(window_score_str, flush=True)

            # Score fusion between two proposal and matching scores
            results = mad_eval.evaluate_nlq_performance(
                submission, ground_truth, thresholds, topK
            )
            title = f"Fusion Epoch {epoch_i}"
            score_str = mad_eval.display_results(
                results, thresholds, topK, title=title
            )
            print(score_str, flush=True)

            # Proposal score
            results_proposal = mad_eval.evaluate_nlq_performance(
                submission_proposal, ground_truth, thresholds, topK
            )
            title = f"Proposal Epoch {epoch_i}"
            score_str_proposal = mad_eval.display_results(
                results_proposal, thresholds, topK, title=title
            )
            print(score_str_proposal, flush=True)

            # Matching score
            results_matching = mad_eval.evaluate_nlq_performance(
                submission_matching, ground_truth, thresholds, topK
            )
            title = f"Matching Epoch {epoch_i}"
            score_str_matching = mad_eval.display_results(
                results_matching, thresholds, topK, title=title
            )
            print(score_str_matching, flush=True)
            save_metrics_path = submission_path.replace(".jsonl", ".txt")
            with open(save_metrics_path, mode="w", encoding="utf-8") as score_writer:
                score_writer.write(score_str)
                score_writer.write(score_str_proposal)
                score_writer.write(score_str_matching)
                score_writer.flush()

            latest_file_paths.append(save_metrics_path)

    if opt.dset_name == "ego4d":
        # Post-processing for Ego4d-NLQ dataset
        submission, submission_proposal, submission_matching = postprocessing_format_ego4d(submission, opt)
        submission_path = os.path.join(opt.results_dir, save_submission_filename)
        # save prediction file to disk
        with open(submission_path, "w") as file_id:
            json.dump(
                {
                    "version": "1.0",
                    "challenge": "ego4d_nlq_challenge",
                    "results": submission,
                }, file_id
            )
        if opt.save_all or opt.eval_modality != "both":
            submission_proposal_path = submission_path.replace("preds", "proposal_preds")
            with open(submission_proposal_path, "w") as file_id:
                json.dump(
                    {
                        "version": "1.0",
                        "challenge": "ego4d_nlq_challenge",
                        "results": submission_proposal,
                    }, file_id
                )
            submission_matching_path = submission_path.replace("preds", "matching_preds")
            with open(submission_matching_path, "w") as file_id:
                json.dump(
                    {
                        "version": "1.0",
                        "challenge": "ego4d_nlq_challenge",
                        "results": submission_matching,
                    }, file_id
                )
        # performance evaluation
        if opt.eval_split_name in ["val"]:
            with open("data/ego4d_ori_data/nlq_val.json") as file_id:
                ground_truth = json.load(file_id)
            thresholds = [0.3, 0.5]
            topK = [1, 5, 10, 50, 100]
            eval_ground_truth = load_jsonl(opt.eval_path)

            # window pre-filtering recall
            window_ranklist_results = window_eval.windows_selection(
                query_id2windowidx, eval_ground_truth, torch.tensor([1, 5, 10, 30, 50]), opt
            )
            title = f"Window Pre-filtering Epoch {epoch_i}"
            window_score_str = window_eval.display_window_results(
                window_ranklist_results, torch.tensor([1, 5, 10, 30, 50]), title=title
            )
            print(window_score_str, flush=True)

            # Score fusion between two proposal and matching scores
            results, mIoU = ego4d_eval.evaluate_nlq_performance(
                submission, ground_truth, thresholds, topK
            )
            title = f"Fusion Epoch {epoch_i}"
            score_str = ego4d_eval.display_results(
                results, mIoU, thresholds, topK, title=title
            )
            print(score_str, flush=True)
            # exit(1)

            # Proposal score
            results_proposal, mIoU_proposal = ego4d_eval.evaluate_nlq_performance(
                submission_proposal, ground_truth, thresholds, topK
            )
            title = f"Proposal Epoch {epoch_i}"
            score_str_proposal = ego4d_eval.display_results(
                results_proposal, mIoU_proposal, thresholds, topK, title=title
            )
            print(score_str_proposal, flush=True)

            # Matching score
            results_matching, mIoU_matching = ego4d_eval.evaluate_nlq_performance(
                submission_matching, ground_truth, thresholds, topK
            )
            title = f"Matching Epoch {epoch_i}"
            score_str_matching = ego4d_eval.display_results(
                results_matching, mIoU_matching, thresholds, topK, title=title
            )
            print(score_str_matching, flush=True)

            save_metrics_path = submission_path.replace(".json", ".txt")
            with open(save_metrics_path, mode="w", encoding="utf-8") as score_writer:
                score_writer.write(window_score_str)
                score_writer.write(score_str)
                score_writer.write(score_str_proposal)
                score_writer.write(score_str_matching)
                score_writer.flush()
            latest_file_paths.append(save_metrics_path)
        else:
            print("end of inference on test split")
            exit(0)

    if opt.eval_modality == "both":
        output_results = results
        if opt.dset_name == "ego4d":
            output_mIoU = mIoU
        latest_file_paths.append(submission_path)
    elif opt.eval_modality == "proposal":
        output_results = results_proposal
        if opt.dset_name == "ego4d":
            output_mIoU = mIoU_proposal
        latest_file_paths.append(submission_proposal_path)
    elif opt.eval_modality == "clip":
        output_results = results_matching
        if opt.dset_name == "ego4d":
            output_mIoU = mIoU_matching
        latest_file_paths.append(submission_matching_path)

    if opt.dset_name == "mad":
        output_mIoU = None

    return output_results, output_mIoU, [window_score_str, score_str, score_str_proposal,
                                         score_str_matching], latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ###
    # set lower learning rate for adapter module
    ###
    adapter_param_tp = [(n, p) for n, p in param_dicts if n.startswith("adapter_layer.")]
    detr_param_tp = [(n, p) for n, p in param_dicts if not n.startswith("adapter_layer.")]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in adapter_param_tp], 'lr': opt.lr * opt.coef_lr},
        {'params': [p for n, p in detr_param_tp], 'lr': opt.lr},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, criterion, optimizer, lr_scheduler


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    opt.results_dir = opt.model_dir
    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None

    pre_filtering_dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        appearance_feat_dir=opt.appearance_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
    )

    eval_inter_window_dataset = PreFilteringDataset(**pre_filtering_dataset_config)

    eval_inter_window_loader = DataLoader(
        eval_inter_window_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )
    for batch in tqdm(eval_inter_window_loader, desc="TEST"):
        break

    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        motion_feat_dir=opt.motion_feat_dir,
        appearance_feat_dir=opt.appearance_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio,
        topk_window=opt.topk_window,
    )
    dataset_config["is_eval"] = True
    eval_intra_window_dataset = StartEndDataset(**dataset_config)

    model, criterion, _, _ = setup_model(opt)

    if opt.dset_name == "mad":
        save_submission_filename = "inference_{}_{}_{}_preds.jsonl".format(
            opt.dset_name, opt.eval_split_name, opt.eval_id)
    else:
        save_submission_filename = "inference_{}_{}_{}_preds.json".format(
            opt.dset_name, opt.eval_split_name, opt.eval_id)
    logger.info("Starting inference...")

    with torch.no_grad():
        results, mIoU, score_str, latest_file_paths = \
            eval_epoch(model, eval_inter_window_dataset, eval_intra_window_dataset, opt, save_submission_filename,
                       epoch_i=None,
                       criterion=criterion,
                       tb_writer=None)


if __name__ == '__main__':
    start_inference()
