import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

from constants import SYNTHETIC_DATASET_NAME, REAL_DATASET_NAME, MODES, OUTPUT_PATH
from dataset_creation.synthetic import generate_synthetic_datasets
from dataset_creation.real import generate_real_datasets


def train_detectron2():
    dataset_storage = {
        'synthetic': generate_synthetic_datasets(),
        'real': generate_real_datasets()
    }

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (SYNTHETIC_DATASET_NAME[MODES[0]], REAL_DATASET_NAME[MODES[0]])
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataset_storage["synthetic"][MODES[0]]['unit_classes'])
    cfg.OUTPUT_DIR = str(OUTPUT_PATH)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return cfg
