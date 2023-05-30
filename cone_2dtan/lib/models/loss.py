import torch
import torch.nn.functional as F


def bce_rescale_loss(scores, masks, targets, cfg):
    # print("score: ",scores.shape)
    # print("masks: ", masks.shape)
    # print("targets: ", targets.shape)
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    joint_prob = torch.sigmoid(scores) * masks
    target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    # print("target_prob non zero len",torch.count_nonzero(target_prob))
    # print("torch.isnan(joint_prob): ",torch.isnan(joint_prob),torch.isnan(joint_prob).shape)
    # print("any(torch.isnan(joint_prob)): ",torch.any(torch.isnan(joint_prob)))
    # if torch.any(torch.isnan(joint_prob)):
    #     print("joint_prob contains nan value")
    #     exit(1)
    # if torch.any(torch.isnan(target_prob)):
    #     print("target_prob contains nan value")
    #     torch.set_printoptions(profile="full")
    #     #print("targets: ", targets)
    #     for idx,item in enumerate(target_prob):
    #         if torch.any(torch.isnan(item)):
    #             print("idx  : ", idx)
    #             print("target_prob item : ", item)
    #     #print("target_prob: ", target_prob)
    #     torch.set_printoptions(profile="default")
    #     exit(1)
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    loss_value = torch.sum(loss) / torch.sum(masks)

    # torch.set_printoptions(profile="full")
    # print("joint_prob: ", joint_prob[0])
    # print("target_prob: ", target_prob[0])
    # print("loss: ", loss)
    # print("loss_value: ", loss_value)
    # torch.set_printoptions(profile="default")
    # exit(1)

    return loss_value, joint_prob


def adapter_loss(input, cfg):
    ######
    # additional adapter NCE loss, followed by CLIP implementation
    #####
    # assert 'logits_per_video' in pos_outputs

    logits_per_video = input / cfg.ADAPER_TEMPERATURE
    bsz = len(logits_per_video)
    diagonal_indices = torch.arange(bsz).to(logits_per_video.device)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss_per_video = criterion(logits_per_video, diagonal_indices)
    loss_per_text = criterion(logits_per_video.T, diagonal_indices)
    loss = (loss_per_video + loss_per_text) / 2
    return loss
