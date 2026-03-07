from pathlib import Path
from polyis.utilities import get_config


CONFIG = get_config()
DATASETS = CONFIG['EXEC']['DATASETS']
DATASETS_DIR = CONFIG['DATA']['DATASETS_DIR']
SOURCE_DIR = CONFIG['DATA']['SOURCE_DIR']
OTIF_DATASET = CONFIG['DATA']['OTIF_DATASET']