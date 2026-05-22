#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import tempfile
import multiprocessing as mp
from functools import partial
import typing

import torch
import torch.utils.data
import torch.optim

from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import ImageFolder
from torch.optim import AdamW

from polyis.models.classifier.classify_image_with_position import ClassifyImageWithPosition
from polyis.models.classifier.yolo import YoloN, YoloS, YoloM, YoloL, YoloX
from polyis.train.training_loop import train
from polyis.utilities import ProgressBar, get_config
from polyis.io import cache


config = get_config()
TILE_SIZES = config['EXEC']['TILE_SIZES']
DATASETS = config['EXEC']['DATASETS']
CLASSIFIERS = [c for c in config['EXEC']['CLASSIFIERS'] if c != 'Perfect']


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--clear', action='store_true',
                        help='Clear existing results directories before training')
    parser.add_argument('--no_visualize', action='store_true', default=False,
                        help='Disable training progress visualizations during training')
    return parser.parse_args()


def ShuffleNet05_factory(_tile_size: int):
    from polyis.models.classifier.shufflenet import ShuffleNet05
    return ShuffleNet05()

def ShuffleNet20_factory(_tile_size: int):
    from polyis.models.classifier.shufflenet import ShuffleNet20
    return ShuffleNet20()

def MobileNetL_factory(_tile_size: int):
    from polyis.models.classifier.mobilenet import MobileNetL
    return MobileNetL()

def MobileNetS_factory(_tile_size: int):
    from polyis.models.classifier.mobilenet import MobileNetS
    return MobileNetS()

def WideResNet50_factory(_tile_size: int):
    from polyis.models.classifier.wide_resnet import WideResNet50
    return WideResNet50()

def WideResNet101_factory(_tile_size: int):
    from polyis.models.classifier.wide_resnet import WideResNet101
    return WideResNet101()

def ResNet152_factory(_tile_size: int):
    from polyis.models.classifier.resnet import ResNet152
    return ResNet152()

def ResNet101_factory(_tile_size: int):
    from polyis.models.classifier.resnet import ResNet101
    return ResNet101()

def ResNet18_factory(_tile_size: int):
    from polyis.models.classifier.resnet import ResNet18
    return ResNet18()

def EfficientNetS_factory(_tile_size: int):
    from polyis.models.classifier.efficientnet import EfficientNetS
    return EfficientNetS()

def EfficientNetL_factory(_tile_size: int):
    from polyis.models.classifier.efficientnet import EfficientNetL
    return EfficientNetL()


MODEL_ZOO = {
    'YoloN': YoloN,
    'YoloS': YoloS,
    'YoloM': YoloM,
    'YoloL': YoloL,
    'YoloX': YoloX,
    'ShuffleNet05': ShuffleNet05_factory,
    'ShuffleNet20': ShuffleNet20_factory,
    'MobileNetL': MobileNetL_factory,
    'MobileNetS': MobileNetS_factory,
    'WideResNet50': WideResNet50_factory,
    'WideResNet101': WideResNet101_factory,
    'ResNet152': ResNet152_factory,
    'ResNet101': ResNet101_factory,
    'ResNet18': ResNet18_factory,
    'EfficientNetS': EfficientNetS_factory,
    'EfficientNetL': EfficientNetL_factory,
}


class ImageFolderWithPosition(ImageFolder):
    def __init__(self, root: str, transform: typing.Callable | None = None):
        super().__init__(root, transform)
        self.mem: list[tuple[torch.Tensor, torch.Tensor, int, tuple[int, int]] | None] = [None] * len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        mem = self.mem[index]
        if mem is not None:
            sample, diff_sample, target, pos = mem
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            diff_path = path.replace('/data/', '/diff/')
            assert os.path.exists(diff_path), f"Diff sample not found: {diff_path}"
            diff_sample = self.loader(diff_path)

            parts = path.split('.')
            pos = parts[-2]
            y, x = pos.split('_')[-2:]
            pos = (int(y), int(x))

            self.mem[index] = (sample, diff_sample, target, pos)

        if self.transform is not None:
            sample = self.transform(sample)
            diff_sample = self.transform(diff_sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = torch.cat([sample, diff_sample], dim=0)
        pos = torch.tensor(pos, dtype=torch.float32)
        return sample, target, pos


def train_classifier(dataset: str, tile_size: int, model_type: str,
                     visualize: bool, gpu_id: int, command_queue: mp.Queue):
    device = f'cuda:{gpu_id}'

    original_training_path = cache.index(dataset, 'training')

    results_dir = os.path.join(original_training_path, 'results', f'{model_type}_{tile_size}')
    os.makedirs(results_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(suffix=f'_polyis_training_data_{model_type}_{tile_size}') as tmpdir:
        training_path = os.path.join(tmpdir, 'training')
        training_path = original_training_path

        if model_type not in MODEL_ZOO:
            raise ValueError(f"Unsupported model type: {model_type}")
        model = MODEL_ZOO[model_type](tile_size).to(device)
        model = ClassifyImageWithPosition(model, pos_encode_size=16).to(device)

        training_data_path = os.path.join(training_path, 'data', f'tilesize_{tile_size}')

        assert os.path.exists(training_data_path), \
            f"Training data directory {training_data_path} does not exist. " \
            "Please run p012_tune_create_training_data.py first."

        loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_data = ImageFolderWithPosition(training_data_path, transform=transform)

        generator = torch.Generator().manual_seed(0)
        split = int(0.8 * len(train_data))
        train_data, test_data = torch.utils.data.random_split(
            dataset=train_data,
            lengths=[split, len(train_data) - split],
            generator=generator
        )

        batch_size = 256
        min_batch_size = 8

        max_model_name_length = max(len(name) for name in MODEL_ZOO)
        run_train = partial(
            train,
            results_dir=results_dir,
            model_type=model_type,
            device=device,
            tile_size=tile_size,
            visualize=visualize,
            command_queue=command_queue,
            dataset=dataset,
            pos_in_batch=True,
            max_model_name_length=max_model_name_length,
        )

        while batch_size > min_batch_size:
            try:
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

                base_batch_size = 256
                lr_scale = batch_size / base_batch_size
                lr_frozen = 1e-3 * lr_scale
                lr_unfrozen = 1e-4 * lr_scale

                lr_frozen = lr_frozen * 0.5
                lr_unfrozen = lr_unfrozen * 0.5

                model.freeze_base_model()
                optimizer_frozen = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen, weight_decay=0.01)
                _, best_raw_wts_stage1, best_loss_stage1 = run_train(
                    model, loss_fn, optimizer_frozen, train_loader, test_loader, n_epochs=7, frozen=True
                )

                model.load_state_dict(best_raw_wts_stage1)

                model.unfreeze_base_model()
                optimizer_unfrozen = AdamW(model.parameters(), lr=lr_unfrozen, weight_decay=0.01)
                run_train(
                    model, loss_fn, optimizer_unfrozen, train_loader, test_loader, n_epochs=7,
                    initial_best_loss=best_loss_stage1
                )

                break

            except Exception:
                batch_size = batch_size // 2
                if batch_size < min_batch_size:
                    print(f"Batch size reduced to minimum (1) for {model_type} (tile_size={tile_size}), but still failing")
                    raise
                if torch.cuda.is_available() and device.startswith('cuda:'):
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()


def main(args):
    mp.set_start_method('spawn', force=True)

    funcs: list[partial] = []

    for dataset_name in DATASETS:
        training_dir = cache.index(dataset_name, 'training')

        if not os.path.exists(training_dir):
            print(f"Training directory {training_dir} does not exist, skipping...")
            continue

        results_dir = os.path.join(training_dir, 'results')
        if args.clear and os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        for classifier in CLASSIFIERS:
            for tile_size in TILE_SIZES:
                func = partial(train_classifier, dataset_name, tile_size, classifier, not args.no_visualize)
                funcs.append(func)

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"

    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
