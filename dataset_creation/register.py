from constants import MODES

from detectron2.data import DatasetCatalog, MetadataCatalog


def register_datasets(dataset_storage: dict):
    DatasetCatalog.clear()

    for name in dataset_storage.keys():
        for mode in MODES:
            ds = dataset_storage[name][mode]
            ds_name = f'clash_{name}_{mode}'
            DatasetCatalog.register(ds_name, lambda x=None: ds['dicts'])
            MetadataCatalog.get(ds_name).set(thing_classes=ds['unit_classes'])
