import os

from detectron2.engine import DefaultPredictor

from constants import SYNTHETIC_DATASET_NAME, REAL_DATASET_NAME, MODES


def get_detectron2_predictor(cfg):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.DATASETS.TEST = (SYNTHETIC_DATASET_NAME[MODES[1]], REAL_DATASET_NAME[MODES[1]])
    return DefaultPredictor(cfg)
