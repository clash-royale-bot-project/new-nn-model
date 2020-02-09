import json
import re

import cv2 as cv
import imageio
import xmltodict
from detectron2.structures import BoxMode
from tqdm import tqdm

from constants import DATASET_PATH, MODES, REAL_DATASET_NAME, ANNOTATED_PATH, TRIM_POSTFIX
from dataset_creation.utils import create_counter, get_unit_classes_list
from dataset_creation.cv_utils import get_polygon_from_bbox

unit_classes = get_unit_classes_list()
get_next_counter = create_counter()


def generate_real_datasets():
    dataset_path = DATASET_PATH
    datasets = {}

    for mode in MODES:
        datasets[mode] = {
            "unit_classes": unit_classes,
            "dicts": []
        }

        mode_dir = dataset_path / REAL_DATASET_NAME[mode]
        mode_dir.mkdir(exist_ok=True)

    img_paths = sorted(
        list((ANNOTATED_PATH / 'images').glob('*.jpg')) + list((ANNOTATED_PATH / 'images').glob('*.png')))
    anno_paths = sorted((ANNOTATED_PATH / 'annotations').glob('*.xml'))
    index = range(len(img_paths))

    assert len(img_paths) == len(anno_paths)

    for i, img, anno in tqdm(zip(index, img_paths, anno_paths)):
        assert img.stem == anno.stem, 'Files have different ID in their names.'
        # print(img.stem, int(img.stem))
        # assert img.stem == str(int(img.stem))

        if i % 11 == 0:
            mode = MODES[1]
        else:
            mode = MODES[0]

        ds = datasets[mode]
        mode_dir = dataset_path / REAL_DATASET_NAME[mode]

        filename = img.name
        real_image = cv.imread(str(img), cv.IMREAD_UNCHANGED)

        image_id = get_next_counter()
        annotations = []
        height, width = real_image.shape[:2]

        new_filename = f'{mode}_img_{image_id}.png'
        file_path = str(mode_dir / new_filename)
        imageio.imwrite(file_path, real_image)

        anno_xml = xmltodict.parse(open(anno, 'r').read())['annotation']
        if 'object' in anno_xml:
            anno_objects = json.loads(json.dumps(anno_xml['object']))
        else:
            continue

        if type(anno_objects) == dict:
            anno_objects = [anno_objects]

        for obj in anno_objects:
            unit_class = re.sub(TRIM_POSTFIX, '', obj['name'])
            category_id = unit_classes.index(unit_class)
            bbox = obj['bndbox']
            bbox = [float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])]
            annotations.append({
                'bbox': bbox,
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': category_id,
                'segmentation': [get_polygon_from_bbox(bbox)],
                'iscrowd': 0
            })

        ds['dicts'].append({
            'file_name': file_path,
            'image_id': image_id,
            'annotations': annotations,
            'height': height,
            'width': width
        })

    return datasets
