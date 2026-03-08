from pathlib import Path
from polyis.utilities import get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
DATASETS_DIR = CONFIG['DATA']['DATASETS_DIR']
SOURCE_DIR = CONFIG['DATA']['SOURCE_DIR']
OTIF_DATASET = CONFIG['DATA']['OTIF_DATASET']
TRAINING_DIR = str(Path(DATASETS_DIR).parent / 'training')


def dataset(dataset_name, *args):
    # Path to a processed dataset
    path = Path(DATASETS_DIR) / dataset_name
    for arg in args:
        path /= arg
    return path


def source(dataset_name, *args):
    # Path to raw source data
    path = Path(SOURCE_DIR) / dataset_name
    for arg in args:
        path /= arg
    return path


def otif(otif_dataset, *args):
    # Path to OTIF dataset
    path = Path(OTIF_DATASET) / otif_dataset
    for arg in args:
        path /= arg
    return path


def training(framework, dataset_name, *args):
    # Path to training data for a specific framework
    path = Path(TRAINING_DIR) / framework / dataset_name
    for arg in args:
        path /= arg
    return path
