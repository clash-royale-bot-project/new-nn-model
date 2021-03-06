import re
from pathlib import Path

BASE_PATH = Path('./clash_royal_source_data')
OUTPUT_PATH = Path('./clash_royal_output')
CLASSES_FILE = Path('./predefined_classes.txt')

ARENAS_PATH = BASE_PATH / 'source' / 'arenas'
UNITS_PATH = BASE_PATH / 'source' / 'units'
ANNOTATED_PATH = BASE_PATH / 'source' / 'annotated'
DATASET_PATH = BASE_PATH / 'dataset'
SCREENSHOTS_PATH = BASE_PATH / 'screenshots'

MODES = ['train', 'val']

DATASET_NAME_BASE = 'clash_royale_v0'

SYNTHETIC_DATASET_NAME = {
    MODES[0]: f'{DATASET_NAME_BASE}_synthetic_{MODES[0]}',
    MODES[1]: f'{DATASET_NAME_BASE}_synthetic_{MODES[1]}'
}

REAL_DATASET_NAME = {
    MODES[0]: f'{DATASET_NAME_BASE}_real_{MODES[0]}',
    MODES[1]: f'{DATASET_NAME_BASE}_real_{MODES[1]}'
}

N_SAMPLES_PER_ARENA = {
    MODES[0]: 25,
    MODES[1]: 5
}

TRIM_POSTFIX = re.compile(r"_(red|blue)$")

GAUSS_SIGMA = 160
PROBABILITY_DECAY = 0.6
