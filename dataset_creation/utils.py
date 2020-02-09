import re

from constants import CLASSES_FILE, TRIM_POSTFIX


def _create_counter(start_with: int = 0):
    """Return an iterator with the integer counter."""
    c = start_with
    while True:
        yield c
        c += 1


def create_counter():
    counter = iter(_create_counter())

    def next_counter():
        return next(counter)

    return next_counter


def get_unit_classes_list():
    unit_classes = set()
    for line in open(CLASSES_FILE, 'r'):
        # TODO : don't trim the team color, and make sure there are units of both colors in the synthetic dataset
        unit_class = re.sub(TRIM_POSTFIX, '', line.strip())
        unit_classes.add(unit_class)
    return sorted(unit_classes)


def get_unit_class(filename, unit_classes):
    name_without_type = '_'.join(filename.split('_')[1:])
    for unit_class in unit_classes:
        if name_without_type.find(unit_class) == 0:
            return unit_class
    raise ValueError(f'No unit class found for {filename}.')
