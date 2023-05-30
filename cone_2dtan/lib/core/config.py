from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

config = edict()

config.WORKERS = 16
config.LOG_DIR = ''
config.MODEL_DIR = ''
config.RESULT_DIR = ''
config.DATA_DIR = ''
config.VERBOSE = False
config.TAG = ''
config.ADAPER_TEMPERATURE = 0.07
config.SVAE_ALL = False

# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# TAN related params
config.TAN = edict()
config.TAN.ADAPTER_MODULE = edict()
config.TAN.ADAPTER_MODULE.NAME = ''
config.TAN.ADAPTER_MODULE.PARAMS = None
config.TAN.FRAME_MODULE = edict()
config.TAN.FRAME_MODULE.NAME = ''
config.TAN.FRAME_MODULE.PARAMS = None
config.TAN.PROP_MODULE = edict()
config.TAN.PROP_MODULE.NAME = ''
config.TAN.PROP_MODULE.PARAMS = None
config.TAN.FUSION_MODULE = edict()
config.TAN.FUSION_MODULE.NAME = ''
config.TAN.FUSION_MODULE.PARAMS = None
config.TAN.MAP_MODULE = edict()
config.TAN.MAP_MODULE.NAME = ''
config.TAN.MAP_MODULE.PARAMS = None
config.TAN.PRED_INPUT_SIZE = 512

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.USE_MATCHING_SCORE = False
config.MODEL.ADAPTER = ''
config.MODEL.CHECKPOINT = ''  # The checkpoint for the best performance

# DATASET related params
config.DATASET = edict()
config.DATASET.APPEARANCE_FEAT_DIR = ''
config.DATASET.Q_FEAT_DIR = ''
config.DATASET.CLIP_LEN = 0.5333
config.DATASET.ROOT = ''
config.DATASET.NAME = ''
config.DATASET.MODALITY = ''
config.DATASET.VIS_INPUT_TYPE = ''
config.DATASET.OUTPUT_SUFFIX = ''
config.DATASET.NO_VAL = False
config.DATASET.BIAS = 0
config.DATASET.NUM_SAMPLE_CLIPS = 256
config.DATASET.TARGET_STRIDE = 16
config.DATASET.DOWNSAMPLING_STRIDE = 16
config.DATASET.SPLIT = ''
config.DATASET.NORMALIZE = False
config.DATASET.RANDOM_SAMPLING = False

# train
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 20
config.TRAIN.MAX_EPOCH = 20
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False
config.TRAIN.CONTRASTIVE_LOSS = False
config.TRAIN.ADAPTER_LOSS = False
config.TRAIN.ADAPTER_LOSS_WEIGHT = 0.1
config.TRAIN.ADAPTER_START_EPOCH = 27

config.LOSS = edict()
config.LOSS.ADAPTER_NAME = 'adapter_loss'
config.LOSS.NAME = 'bce_loss'
config.LOSS.PARAMS = None

# test
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 0.5
config.TEST.INTERVAL = 1
config.TEST.EPOCH_INTERVAL = 2
config.TEST.EVAL_TRAIN = False
config.TEST.BATCH_SIZE = 1
config.TEST.NMS_TOP_K = 5
config.TEST.WINDOW_TOP_K = 15
config.TEST.PROPOSAL_TOP_K = 30
config.TEST.USE_NEW_PROPOSAL_METHOD = False
config.TEST.USE_NMS_WITHIN_WINDOW = False
config.TEST.NMS_THRESH_WITHIN_WINDOW = 0.3

def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k], v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
