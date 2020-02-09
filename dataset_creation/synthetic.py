import pickle
import random
from shutil import rmtree
from time import sleep

import cv2 as cv
import numpy as np
from detectron2.structures import BoxMode
from tqdm import tqdm

from constants import ARENAS_PATH, UNITS_PATH, DATASET_PATH, GAUSS_SIGMA, PROBABILITY_DECAY, MODES,\
    SYNTHETIC_DATASET_NAME, N_SAMPLES_PER_ARENA

from dataset_creation.utils import create_counter, get_unit_class, get_unit_classes_list
from dataset_creation.cv_utils import get_convex_hull, paste_unit, scale_unit

arenas_glob = [str(s) for s in ARENAS_PATH.glob('*.png')]
units_glob = [str(s) for s in UNITS_PATH.glob('*.png')]

arenas = [cv.imread(filename, cv.IMREAD_UNCHANGED) for filename in arenas_glob]

unit_classes = get_unit_classes_list()
get_next_counter = create_counter()


def generate_sample(arena, units):
    annotations = []

    arena_img = arena.copy()
    arena_h, arena_w, _ = arena_img.shape

    probability = 1

    for unit in units:
        if random.random() > probability:
            break
        probability *= PROBABILITY_DECAY

        unit_img, unit_img_bw, unit_class = (unit['img'], unit['img_bw'], unit['unit_class'])
        category_id = unit_classes.index(unit_class)

        if unit_class in ['axe', 'miner']:
            unit_img, unit_img_bw = scale_unit(unit_img, unit_img_bw, 1.5)

        h, w, _ = unit_img.shape
        y_range, x_range = arena_h - h, arena_w - w

        pos_y = round(min(y_range, max(0, random.gauss(y_range // 2, GAUSS_SIGMA))))
        pos_x = round(min(x_range, max(0, random.gauss(x_range // 2, GAUSS_SIGMA))))
        arena_img = paste_unit(arena_img, unit_img, pos_y, pos_x)

        hull = get_convex_hull(unit_img_bw, pos_y, pos_x)
        polygon = hull.flatten().astype(float)

        x_min, y_min = np.min(hull, axis=0)[0]
        x_max, y_max = np.max(hull, axis=0)[0]

        annotations.append({
            'bbox': [x_min, y_min, x_max, y_max],
            'bbox_mode': BoxMode.XYXY_ABS,
            'category_id': category_id,
            'segmentation': [polygon],
            'iscrowd': 0
        })

    return arena_img, annotations


def generate_synthetic_datasets():

    units = [{
        'img': cv.imread(filename, cv.IMREAD_UNCHANGED),
        'img_bw': cv.imread(filename, cv.IMREAD_GRAYSCALE),
        'unit_class': get_unit_class(filename, unit_classes)
    } for filename in units_glob]

    dataset_path = DATASET_PATH
    datasets = {}

    try:
        if dataset_path.exists():
            prompt = input('Are you sure you want to wipe out the existing dataset? (yes/[no]): ')
            if prompt != 'yes':
                raise UserWarning('You decided to save the existing dataset.')
            rmtree(dataset_path)
        dataset_path.mkdir()

        for mode in MODES:

            datasets[mode] = {
                "unit_classes": unit_classes,
                "dicts": []
            }

            mode_dir = dataset_path / SYNTHETIC_DATASET_NAME[mode]
            mode_dir.mkdir()

            for arena in tqdm(arenas):
                for i in range(N_SAMPLES_PER_ARENA[mode]):
                    image_id = get_next_counter()
                    filename = f'{mode}_img_{image_id}.png'
                    height, width = arena.shape[:2]

                    permuted_units = np.random.permutation(units)
                    img, annotations = generate_sample(arena, permuted_units)

                    file_path = str(mode_dir / filename)
                    cv.imwrite(file_path, img)

                    datasets[mode]['dicts'].append({
                        'file_name': file_path,
                        'image_id': image_id,
                        'annotations': annotations,
                        'height': height,
                        'width': width
                    })

                    sleep(1)

            with open(mode_dir / f'{SYNTHETIC_DATASET_NAME[mode]}.pkl', 'wb') as f:
                pickle.dump(datasets[mode], f)

    except UserWarning:
        for mode in MODES:
            mode_dir = dataset_path / SYNTHETIC_DATASET_NAME[mode]
            with open(mode_dir / f'{SYNTHETIC_DATASET_NAME[mode]}.pkl', 'rb') as f:
                datasets[mode] = pickle.load(f)

    finally:
        if datasets == {}:
            raise ValueError('Dataset should not be empty.')
        for mode in MODES:
            if len(list((dataset_path / SYNTHETIC_DATASET_NAME[mode]).iterdir())) != len(datasets[mode]['dicts']) + 1:
                raise ValueError(f'Dataset description does not match with the number of files for the {mode} mode.')

    return datasets
