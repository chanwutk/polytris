from pathlib import Path
from polyis.utilities import get_config

config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']

EXEC_STAGE_MAP = {
    'naive-groundtruth': '000_groundtruth',
    'naive': '002_naive',
    'groundtruth': '003_groundtruth',
    'relevancy': '020_relevancy',
    'pruned-polyominoes': '022_pruned_polyominoes',
    'comp-frames': '033_compressed_frames',
    'comp-dets': '040_compressed_detections',
    'ucomp-dets': '050_uncompressed_detections',
    'ucomp-tracks': '060_uncompressed_tracks',
}
INDEX_STAGE_MAP = {
'det': 'segment/detection',
'training': 'training',
'never-relevant': 'always_relevant' ,
'track_rates': 'track_rate',
}
EVAL_STAGE_MAP = {
    'acc': '070_accuracy',
    'acc_vis': '072_accuracy_visualize',
    'tp': '080_throughput',
    'tp-vis': '082_throughput_visualize',
    'compress': '083_compress',
    'tradeoff': '090_tradeoff',
    'tradeoff-vis': '091_tradeoff',
}


def root(dataset: str, *args):
    # Base cache path for a dataset
    path = Path(CACHE_DIR) / dataset
    for arg in args:
        path /= arg
    return path


def execution(dataset: str, *args):
    # Base execution directory for a dataset
    path = Path(CACHE_DIR) / dataset / 'execution'
    for arg in args:
        path /= arg
    return path


def exec(dataset: str, stage: str, video: str, *args):
    path = Path(CACHE_DIR) / dataset / 'execution'/ video / EXEC_STAGE_MAP[stage]
    for arg in args:
        path /= arg
    return path


def index(dataset: str, stage: str, *args):
    path = Path(CACHE_DIR) / dataset / 'indexing' / INDEX_STAGE_MAP[stage]
    for arg in args:
        path /= arg
    return path


def eval(dataset: str, stage: str, *args):
    path = Path(CACHE_DIR) / dataset / 'evaluation' / EVAL_STAGE_MAP[stage]
    for arg in args:
        path /= arg
    return path


def summary(stage, *args):
    # Cross-dataset summary path (uppercase SUMMARY)
    path = Path(CACHE_DIR) / 'SUMMARY' / stage
    for arg in args:
        path /= arg
    return path


def summary_dataset(dataset, category, *args):
    # Per-dataset summary path (lowercase summary)
    path = Path(CACHE_DIR) / 'summary' / dataset / category
    for arg in args:
        path /= arg
    return path


def sota(system, dataset, *args):
    # SOTA system comparison path
    path = Path(CACHE_DIR) / 'SOTA' / system / dataset
    for arg in args:
        path /= arg
    return path