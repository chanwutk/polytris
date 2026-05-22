#!/usr/local/bin/python

import argparse
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

from polyis.models.classifier.baseline import ShuffleNet05Baseline
from polyis.train.training_loop import train
from polyis.utilities import ProgressBar, get_config
from polyis.io import cache


config = get_config()
TILE_SIZES = config['EXEC']['TILE_SIZES']
DATASETS = config['EXEC']['DATASETS']
CLASSIFIERS = [c for c in config['EXEC']['CLASSIFIERS'] if c != 'Perfect']


def parse_args():
    parser = argparse.ArgumentParser(description='Train image-only baseline classifiers')
    parser.add_argument('--clear', action='store_true',
                        help='Clear existing baseline results directories before training')
    parser.add_argument('--no_visualize', action='store_true', default=False,
                        help='Disable training progress visualizations during training')
    return parser.parse_args()


def ShuffleNet05Baseline_factory(_tile_size: int) -> ShuffleNet05Baseline:
    return ShuffleNet05Baseline()


BASELINE_MODEL_ZOO: dict[str, typing.Callable[[int], torch.nn.Module]] = {
    'ShuffleNet05': ShuffleNet05Baseline_factory,
}


class ImageFolderWithDummyPosition(ImageFolder):
    """RGB tiles only; emits a zero position tensor so the shared dataloader shape matches p014."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        sample, target = super().__getitem__(index)
        pos = torch.zeros(2, dtype=torch.float32)
        return sample, target, pos


def train_classifier_baseline(dataset: str, tile_size: int, model_type: str,
                              visualize: bool, gpu_id: int, command_queue: mp.Queue):
    if model_type not in BASELINE_MODEL_ZOO:
        return

    device = f'cuda:{gpu_id}'

    original_training_path = cache.index(dataset, 'training')
    baseline_name = f'{model_type}Baseline'
    results_dir = os.path.join(original_training_path, 'results', f'{baseline_name}_{tile_size}')
    os.makedirs(results_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(suffix=f'_polyis_baseline_training_{model_type}_{tile_size}') as _tmpdir:
        training_path = original_training_path

        model = BASELINE_MODEL_ZOO[model_type](tile_size).to(device)

        training_data_path = os.path.join(training_path, 'data', f'tilesize_{tile_size}')

        assert os.path.exists(training_data_path), (
            f"Training data directory {training_data_path} does not exist. "
            "Please run p012_tune_create_training_data.py first."
        )

        loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_data = ImageFolderWithDummyPosition(training_data_path, transform=transform)

        generator = torch.Generator().manual_seed(0)
        split = int(0.8 * len(train_data))
        train_data, test_data = torch.utils.data.random_split(
            dataset=train_data,
            lengths=[split, len(train_data) - split],
            generator=generator
        )

        batch_size = 256
        min_batch_size = 8

        max_model_name_length = max(len(name) for name in BASELINE_MODEL_ZOO)
        run_train = partial(
            train,
            results_dir=results_dir,
            model_type=baseline_name,
            device=device,
            tile_size=tile_size,
            visualize=visualize,
            command_queue=command_queue,
            dataset=dataset,
            pos_in_batch=False,
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
                optimizer_frozen = AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr_frozen,
                    weight_decay=0.01,
                )
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
                    print(
                        f"Batch size reduced to minimum for {baseline_name} (tile_size={tile_size}), "
                        f"but still failing"
                    )
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

        results_root = os.path.join(training_dir, 'results')
        if args.clear:
            for classifier in CLASSIFIERS:
                if classifier not in BASELINE_MODEL_ZOO:
                    continue
                baseline_name = f'{classifier}Baseline'
                for tile_size in TILE_SIZES:
                    rd = os.path.join(results_root, f'{baseline_name}_{tile_size}')
                    if os.path.exists(rd):
                        shutil.rmtree(rd)
        os.makedirs(results_root, exist_ok=True)

        for classifier in CLASSIFIERS:
            if classifier not in BASELINE_MODEL_ZOO:
                continue
            for tile_size in TILE_SIZES:
                func = partial(
                    train_classifier_baseline,
                    dataset_name,
                    tile_size,
                    classifier,
                    not args.no_visualize,
                )
                funcs.append(func)

    if not funcs:
        print("No baseline trainers scheduled (check CLASSIFIERS vs BASELINE_MODEL_ZOO).")
        return

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"

    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
