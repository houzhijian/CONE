import os
import time
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from cone.config import BaseOptions
from cone.ego4d_mad_dataloader import StartEndDataset, start_end_collate, prepare_batch_inputs, PreFilteringDataset
from cone.inference import eval_epoch, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i + 1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        # global_step = epoch_i * num_training_examples + batch_idx
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        pos_model_inputs, pos_clip_model_inputs, neg_model_inputs_tuple, targets \
            = prepare_batch_inputs(batch[1], batch[2], opt.device, non_blocking=True)

        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        timer_start = time.time()
        pos_outputs = model.forward(**pos_model_inputs)
        if opt.neg_loss:
            neg_outputs = model.forward(**neg_model_inputs_tuple[0])
            loss_dict = criterion(pos_outputs, targets, neg_outputs)
        else:
            loss_dict = criterion(pos_outputs, targets, None)

        weight_dict = criterion.weight_dict

        losses = 0
        for k in loss_dict.keys():
            if k in weight_dict:
                losses += loss_dict[k] * weight_dict[k]

        if opt.adapter_loss and epoch_i >= opt.start_epoch_for_adapter:
            pos_outputs["logits_per_video"] = model.forward_clip_matching(**pos_clip_model_inputs,
                                                                 proposal=targets["span_proposal"],
                                                                 is_groundtruth=True)
            adapter_loss = criterion.loss_adapter(pos_outputs)["loss_adapter"]
            losses += adapter_loss * weight_dict["loss_adapter"]

        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()

        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        if opt.adapter_loss and epoch_i >= opt.start_epoch_for_adapter:
            k = "loss_adapter"
            loss_meters["loss_adapter"].update(
                float(adapter_loss) * weight_dict[k] if k in weight_dict else float(adapter_loss))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i + 1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i + 1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i + 1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_inter_window_dataset, eval_intra_window_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    # start training
    score_writer = open(
        os.path.join(opt.results_dir, "eval_results.txt"), mode="w", encoding="utf-8"
    )

    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = 0
    else:
        start_epoch = opt.start_epoch
    if opt.debug:
        start_epoch = 0
    if opt.dset_name == "mad":
        save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    else:
        save_submission_filename = "latest_{}_{}_preds.json".format(opt.dset_name, opt.eval_split_name)

    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        if opt.eval_path is not None and (epoch_i + 1) % opt.eval_epoch_interval == 0:
            with torch.no_grad():
                results, mIoU, score_str, latest_file_paths = \
                    eval_epoch(model, eval_inter_window_dataset, eval_intra_window_dataset, opt,
                               save_submission_filename, epoch_i=epoch_i, criterion=None,  tb_writer=tb_writer)

            for score_item in score_str:
                score_writer.write(score_item)
            score_writer.flush()

            if opt.dset_name == "mad":
                stop_score = torch.mean(results[0])
                print("stop_score ", stop_score)
            else:
                stop_score = (results[0][0] + results[1][0]) / 2
                print("stop_score ", stop_score)

            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score * 100}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()


def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
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

    dataset_config["data_path"] = opt.train_path
    train_dataset = StartEndDataset(**dataset_config)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    for batch in tqdm(train_loader, desc="TEST"):
        break

    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining
        dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        dataset_config["is_eval"] = True
        eval_intra_window_dataset = StartEndDataset(**dataset_config)
        pre_filtering_dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.eval_path,
            appearance_feat_dir=opt.appearance_feat_dir,
            q_feat_dir=opt.t_feat_dir,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
        )
        eval_inter_window_dataset = PreFilteringDataset(**pre_filtering_dataset_config)
    else:
        eval_intra_window_dataset = None
        eval_inter_window_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_inter_window_dataset, eval_intra_window_dataset, opt)
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug


if __name__ == '__main__':
    best_ckpt_path, eval_split_name, eval_path, debug = start_training()
