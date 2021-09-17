from easydict import EasyDict as edict
import yaml

"""
Add default config for DTFVOS.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.POSITION_EMBEDDING = 'sine'  # sine or learned
cfg.MODEL.PREDICT_MASK = True
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "resnet50"  # resnet18 or resnet50
cfg.MODEL.BACKBONE.DILATION = False
# MODEL.TRANSFORMER
cfg.MODEL.TRANSFORMER = edict()
cfg.MODEL.TRANSFORMER.NHEADS = 8
cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 2048
cfg.MODEL.TRANSFORMER.ENC_LAYERS = 6
cfg.MODEL.TRANSFORMER.DEC_LAYERS = 6
cfg.MODEL.TRANSFORMER.PRE_NORM = False
cfg.MODEL.TRANSFORMER.DIVIDE_NORM = False
cfg.MODEL.TRANSFORMER.ONLY_ENCODER = False
# MODEL.SEGMENTATION
cfg.MODEL.SEGMENTATION = edict()
cfg.MODEL.SEGMENTATION.M_DIM = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 1e-4
cfg.TRAIN.WEIGHT_DECAY = 1e-4
cfg.TRAIN.EPOCH = 160
cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.FREEZE_BACKBONE_BN = True
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.CLIP_MAX_NORM = 0.1
cfg.TRAIN.MODE = 'normal'  # 'normal' or 'recurrent'
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.STEP_SIZE = 40

# DATA
cfg.DATA = edict()
cfg.DATA.PRETRAIN_ROOT = '/public/datasets/VOS/pretrain'
cfg.DATA.DAVIS_ROOT = '/public/datasets/DAVIS'
cfg.DATA.YTBVOS_ROOT = '/public/datasets/YTBVOS'
cfg.DATA.SIZE = (480, 854)
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.MAX_OBJECTS = 3
cfg.DATA.TRAIN.FRAMES_PER_CLIP = 2
cfg.DATA.TRAIN.DATASETS_NAME = ["DAVIS17", "YTBVOS"]
cfg.DATA.TRAIN.DATASETS_RATIO = [5, 1]
cfg.DATA.TRAIN.DAVIS_SKIP_INCREMENT = [5, 5]
cfg.DATA.TRAIN.YTBVOS_SKIP_INCREMENT = [3, 1]
cfg.DATA.TRAIN.EPOCH_PER_INCREMENT = 5
cfg.DATA.TRAIN.SAMPLES_PER_VIDEO = 2
cfg.DATA.TRAIN.SAMPLE_CHOICE = 'random'  # "order" or "random"
cfg.DATA.TRAIN.CROP = False  # if False, resize cfg.DATA.SIZE; else, random_resize_crop 400
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASET_NAME = "DAVIS17" # "DAVIS16", "DAVIS17" or "YTBVOS"
# DATA.TEST
cfg.DATA.TEST = edict()
cfg.DATA.TEST.DATASET_NAME = "DAVIS17" # "DAVIS17" or "YTBVOS"


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)

