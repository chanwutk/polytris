#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
import multiprocessing as mp
from functools import partial
import traceback
import typing
import copy

import numpy as np
import torch
import torch.utils.data
import torch.optim
import altair as alt
import pandas as pd

from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import ImageFolder
from torch.optim import AdamW

from polyis.models.classifier.classify_image_with_position import ClassifyImageWithPosition
from polyis.models.classifier.yolo import YoloN, YoloS, YoloM, YoloL, YoloX
from polyis.utilities import format_time, ProgressBar, get_config


config = get_config()
CACHE_DIR = config['DATA']['CACHE_DIR']
DATASETS_DIR = config['DATA']['DATASETS_DIR']
TILE_SIZES = config['EXEC']['TILE_SIZES']
DATASETS = config['EXEC']['DATASETS']
CLASSIFIERS = [c for c in config['EXEC']['CLASSIFIERS'] if c != 'Perfect']


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--clear', action='store_true',
                        help='Clear existing results directories before training')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate training progress visualizations during training')
    return parser.parse_args()


# Factory functions for models that don't accept tile_size parameter
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


class History(typing.NamedTuple):
    loss: list[float]
    tp: list[int]
    tn: list[int]
    fp: list[int]
    fn: list[int]
    time: list[float]

    def add(self, result: dict):
        self.loss.append(result['cumulative_loss'])
        self.tp.append(result['tp'])
        self.tn.append(result['tn'])
        self.fp.append(result['fp'])
        self.fn.append(result['fn'])
        self.time.append(result['runtime'])


def plot_training_progress(train_history: History, val_history: History,
                           results_dir: str, train_images_processed: int,
                           val_images_processed: int,
                           throughput_per_epoch: list[list[dict[str, float | int | str]]],
                           frozen: bool):
    """Plot training progress with time on x-axis and loss/accuracy on y-axis"""
    try:
        assert len(val_history.time) == len(train_history.time)
        epoch = len(train_history.loss)

        # Create DataFrame for loss and accuracy data
        plot_data = []
        for i in range(len(train_history.loss)):
            train_precision, train_recall, train_f1 = calculate_metrics(train_history.tp[i], train_history.fp[i], train_history.fn[i])
            val_precision, val_recall, val_f1 = calculate_metrics(val_history.tp[i], val_history.fp[i], val_history.fn[i])
            plot_data.extend([
                {'Epoch': i, 'Value': train_history.loss[i], 'Metric': 'Train Loss', 'Type': 'Loss'},
                {'Epoch': i, 'Value': val_history.loss[i], 'Metric': 'Validation Loss', 'Type': 'Loss'},
                {'Epoch': i, 'Value': train_precision, 'Metric': 'Train Precision', 'Type': 'Precision'},
                {'Epoch': i, 'Value': val_precision, 'Metric': 'Validation Precision', 'Type': 'Precision'},
                {'Epoch': i, 'Value': train_recall, 'Metric': 'Train Recall', 'Type': 'Recall'},
                {'Epoch': i, 'Value': val_recall, 'Metric': 'Validation Recall', 'Type': 'Recall'},
                {'Epoch': i, 'Value': train_f1, 'Metric': 'Train F1', 'Type': 'F1'},
                {'Epoch': i, 'Value': val_f1, 'Metric': 'Validation F1', 'Type': 'F1'}
            ])
        df = pd.DataFrame(plot_data)
        
        # Extract Train/Validation and specific metric from Metric field
        df['Split'] = df['Metric'].str.split(' ').str[0]  # Train or Validation
        df['MetricType'] = df['Metric'].str.split(' ', n=1).str[1]  # Loss or Accuracy

        # Left chart: Loss and accuracy progress
        chart1 = alt.Chart(df).mark_line(point=True).encode(
            x='Epoch:Q',
            y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 1])),
            color='MetricType:N',
            strokeDash='Split:N'
        ).properties(
            title=f'Training Progress - Epoch {epoch + 1}',
            width=400,
            height=300
        ).resolve_scale(color='independent', strokeDash='independent')

        # Right chart: ms/frame stacked bar chart
        # Flatten throughput data into a DataFrame
        throughput_flat = []
        for epoch_throughput in throughput_per_epoch[1:]:
            for op_data in epoch_throughput:
                throughput_flat.append(op_data)
        if len(throughput_flat) == 0:
            return

        # Create DataFrame from flattened throughput data
        throughput_df = pd.DataFrame(throughput_flat)
        throughput_df['op'] = throughput_df['op'].astype(str)
        throughput_df['time'] = throughput_df['time'].astype(float)
        
        # Group by operation and sum times
        op_times_df = throughput_df.groupby('op')['time'].sum().reset_index()
        
        # Determine phase (Training/Validation) based on operation name prefix
        op_times_df['Phase'] = np.where(op_times_df['op'].str.startswith('train'), 'Training', 'Validation')
        
        # Filter out load_data operations
        op_times_df = op_times_df[~op_times_df['op'].str.endswith('load_data')]
        
        # Calculate ms per frame based on phase
        op_times_df['ms_per_frame'] = op_times_df.apply(
            lambda row: row['time'] / train_images_processed if row['Phase'] == 'Training' else row['time'] / val_images_processed,
            axis=1
        )
        
        # Create clean operation labels
        assert isinstance(op_times_df, pd.DataFrame)
        op_times_df['Operation'] = op_times_df['op'].str.replace(r'^(train_|test_)', '', regex=True)
        
        # Select final columns
        bar_df = op_times_df[['Phase', 'Operation', 'ms_per_frame']]
        assert isinstance(bar_df, pd.DataFrame)

        # Only create the plot if we have data
        assert not bar_df.empty
        # Create stacked bar chart
        chart2 = alt.Chart(bar_df).mark_bar().encode(
            x='Phase:N',
            y='ms_per_frame:Q',
            color='Operation:N',
            tooltip=['Phase', 'Operation', alt.Tooltip('ms_per_frame:Q', format='.2f')]
        ).properties(
            title='Milliseconds per Frame by Operation',
            width=300,
            height=300
        ).resolve_scale(color='independent')

        # Combine charts horizontally
        combined_chart = alt.hconcat(chart1, chart2, spacing=20)

        # Save the plot
        plot_path = os.path.join(results_dir, f'training_progress_{"frozen" if frozen else "finetuned"}.png')
        combined_chart.save(plot_path, scale_factor=2)
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error plotting training progress: {e}")
        print("\n" * 20)
        return


def normalize_output_shape(outputs: "torch.Tensor", model_type: str) -> "torch.Tensor":
    # Handle different output formats
    if model_type.startswith('Yolo'):
        # YOLO outputs probabilities directly, ensure they're in the right shape
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
    return outputs


def calculate_confusion_matrix(predictions: "torch.Tensor", labels: "torch.Tensor") -> tuple[int, int, int, int]:
    # Calculate confusion matrix metrics: TP, TN, FP, FN
    tp = torch.sum((predictions == 1) & (labels == 1)).item()
    tn = torch.sum((predictions == 0) & (labels == 0)).item()
    fp = torch.sum((predictions == 1) & (labels == 0)).item()
    fn = torch.sum((predictions == 0) & (labels == 1)).item()
    return int(tp), int(tn), int(fp), int(fn)


def calculate_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    # Calculate precision, recall, and F1 score from confusion matrix
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def train_step(model: "torch.nn.Module", loss_fn: "torch.nn.modules.loss._Loss",
               optimizer: "torch.optim.Optimizer", inputs: "torch.Tensor",
               pos_inputs: "torch.Tensor", labels: "torch.Tensor", model_type: str,
               max_grad_norm: float = 10.0):
    optimizer.zero_grad()

    outputs: "torch.Tensor" = model(inputs, pos_inputs)
    outputs = normalize_output_shape(outputs, model_type)

    loss: "torch.Tensor" = loss_fn(outputs, labels)

    loss.backward()
    
    # Apply gradient clipping to prevent gradient explosions
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    
    optimizer.step()

    return loss.item(), outputs


def run_training_epoch(model: "torch.nn.Module", loss_fn: "torch.nn.modules.loss._Loss",
                       optimizer: "torch.optim.Optimizer", train_loader: "torch.utils.data.DataLoader",
                       device: str, model_type: str, command_queue: mp.Queue, description: str,
                       ema_model: "torch.nn.Module | None" = None, ema_decay: float = 0.999):
    # Initialize timing accumulators
    data_loading_time: float = 0.
    gpu_transfer_time: float = 0.
    step_time: float = 0.
    
    # Initialize metrics accumulators
    cumulative_loss = 0.
    num_samples: int = 0
    train_tp: int = 0
    train_tn: int = 0
    train_fp: int = 0
    train_fn: int = 0
    
    model.train()
    batch_start_time = time.time_ns() / 1e6
    
    command_queue.put((device, { 'description': description.format('T  0%') }))
    
    for idx, (x_batch, y_batch, pos_batch) in enumerate(train_loader):
        # Data loading time (time since last batch completed or epoch started)
        data_loading_time += (time.time_ns() / 1e6) - batch_start_time
        
        # Measure GPU transfer time
        gpu_transfer_start_time = time.time_ns() / 1e6
        x_batch = x_batch.to(device)
        pos_batch = pos_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1).float()
        gpu_transfer_time += (time.time_ns() / 1e6) - gpu_transfer_start_time
        
        # Measure train step time
        step_start_time = time.time_ns() / 1e6
        loss, y_hat = train_step(model, loss_fn, optimizer, x_batch, pos_batch, y_batch, model_type)
        step_time += (time.time_ns() / 1e6) - step_start_time
        
        # Update EMA model if provided
        if ema_model is not None:
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
        
        cumulative_loss += loss
        
        # Calculate training accuracy and confusion matrix metrics
        with torch.no_grad():
            y_hat = normalize_output_shape(y_hat, model_type)
            # Apply sigmoid to convert logits to probabilities for accuracy calculation
            y_hat_probs = torch.sigmoid(y_hat)
            predictions = y_hat_probs > 0.5
            num_samples += len(y_batch)
            
            # Calculate confusion matrix metrics
            batch_tp, batch_tn, batch_fp, batch_fn = calculate_confusion_matrix(predictions, y_batch)
            
            # Accumulate confusion matrix metrics
            train_tp += batch_tp
            train_tn += batch_tn
            train_fp += batch_fp
            train_fn += batch_fn
        
        # Start timing for next batch data loading
        desc = f'T {int(idx * 100 / len(train_loader)):>2}%'
        command_queue.put((device, { 'description': description.format(desc) }))
        batch_start_time = time.time_ns() / 1e6
    
    return {
        'cumulative_loss': float(cumulative_loss) / len(train_loader),
        'tp': train_tp,
        'tn': train_tn,
        'fp': train_fp,
        'fn': train_fn,
        'num_samples': num_samples,
        'runtime': data_loading_time + gpu_transfer_time + step_time,

        'data_loading_time': data_loading_time,
        'gpu_transfer_time': gpu_transfer_time,
        'step_time': step_time,
    }


def run_validation_epoch(model: "torch.nn.Module", loss_fn: "torch.nn.modules.loss._Loss",
                         test_loader: "torch.utils.data.DataLoader", device: str, model_type: str,
                         command_queue: mp.Queue, description: str):
    # Initialize timing accumulators
    data_loading_time: float = 0.
    gpu_transfer_time: float = 0.
    inference_time: float = 0.
    loss_time: float = 0.
    
    # Initialize metrics accumulators
    cumulative_loss = 0.
    num_samples = 0
    val_tp: int = 0
    val_tn: int = 0
    val_fp: int = 0
    val_fn: int = 0
    
    model.eval()
    val_batch_start_time = time.time_ns() / 1e6
    
    command_queue.put((device, { 'description': description.format('V 00%') }))
    
    for idx, (x_batch, y_batch, pos_batch) in enumerate(test_loader):
        # Validation data loading time
        data_loading_time += (time.time_ns() / 1e6) - val_batch_start_time
        
        # Measure validation GPU transfer time
        val_gpu_transfer_start_time = time.time_ns() / 1e6
        x_batch = x_batch.to(device)
        pos_batch = pos_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1).float()
        gpu_transfer_time += (time.time_ns() / 1e6) - val_gpu_transfer_start_time
        
        # Measure inference time
        val_inference_start_time = time.time_ns() / 1e6
        yhat = model(x_batch, pos_batch)
        inference_time += (time.time_ns() / 1e6) - val_inference_start_time
        
        # Handle different output formats
        yhat = normalize_output_shape(yhat, model_type)
        
        val_loss_start_time = time.time_ns() / 1e6
        val_loss = loss_fn(yhat, y_batch)
        cumulative_loss += val_loss
        loss_time += (time.time_ns() / 1e6) - val_loss_start_time
        
        # Apply sigmoid to convert logits to probabilities for accuracy calculation
        yhat_probs = torch.sigmoid(yhat)
        ans = yhat_probs > 0.5
        num_samples += len(y_batch)
        
        # Calculate confusion matrix metrics for validation
        batch_tp, batch_tn, batch_fp, batch_fn = calculate_confusion_matrix(ans, y_batch)
        
        val_tp += batch_tp
        val_tn += batch_tn
        val_fp += batch_fp
        val_fn += batch_fn
        
        # Start timing for next validation batch data loading
        desc = f'V {int(idx * 100 / len(test_loader)):>2}%'
        command_queue.put((device, { 'description': description.format(desc) }))
        val_batch_start_time = time.time_ns() / 1e6
    
    return {
        'cumulative_loss': float(cumulative_loss) / len(test_loader),
        'tp': val_tp,
        'tn': val_tn,
        'fp': val_fp,
        'fn': val_fn,
        'num_samples': num_samples,
        'runtime': data_loading_time + gpu_transfer_time + inference_time + loss_time,

        'data_loading_time': data_loading_time,
        'gpu_transfer_time': gpu_transfer_time,
        'inference_time': inference_time,
        'loss_time': loss_time,
    }


def check_early_stopping(val_loss: float, best_loss: float, early_stopping_counter: int,
                         early_stopping_tolerance: int, early_stopping_threshold: float) -> tuple[bool, int]:
    # Check if validation loss improved
    if val_loss < best_loss:
        return False, 0
    
    early_stopping_counter += 1
    
    # Check early stopping conditions
    if (early_stopping_counter >= early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
        return True, early_stopping_counter
    
    return False, early_stopping_counter


def save_training_results(results_dir: str, epoch_train_losses: list[dict], 
                          epoch_test_losses: list[dict], 
                          throughput_per_epoch: list[list[dict[str, float | int | str]]]):
    # Save training and validation losses to JSON files
    with open(os.path.join(results_dir, 'test_losses.json'), 'w') as f:
        f.write(json.dumps(epoch_test_losses))
    
    with open(os.path.join(results_dir, 'train_losses.json'), 'w') as f:
        f.write(json.dumps(epoch_train_losses))
    
    with open(os.path.join(results_dir, 'throughput_per_epoch.jsonl'), 'w') as f:
        for t in throughput_per_epoch:
            f.write(json.dumps(t) + '\n')


def format_throughput(throughput: dict[str, float | int | str], prefix: str):
    return format_time(**{
        f'{prefix}_{k[:-len("_time")]}': float(v)
        for k, v in throughput.items()
        if k.endswith('_time')
    })


def save_model(model: "torch.nn.Module", ema_model: "torch.nn.Module | None", results_dir: str, name: str | None = None):
    # Save EMA weights if available, otherwise raw weights
    if ema_model is not None:
        best_model_wts = ema_model.state_dict()
        best_raw_model_wts = model.state_dict()
        # Clear any accumulated gradients to save memory in saved model
        ema_model.zero_grad(set_to_none=True)
        with open(os.path.join(results_dir, name or 'model_best.pth'), 'wb') as f:
            torch.save(ema_model, f)
    else:
        best_model_wts = model.state_dict()
        best_raw_model_wts = model.state_dict()
        # Clear any accumulated gradients to save memory in saved model
        model.zero_grad(set_to_none=True)
        with open(os.path.join(results_dir, name or 'model_best.pth'), 'wb') as f:
            torch.save(model, f)
    
    return best_model_wts, best_raw_model_wts


def train(
    model: "torch.nn.Module",
    loss_fn: "torch.nn.modules.loss._Loss",
    optimizer: "torch.optim.Optimizer",
    train_loader: "torch.utils.data.DataLoader",
    test_loader: "torch.utils.data.DataLoader",
    n_epochs: int,
    results_dir: str,
    model_type: str,
    device: str,
    command_queue: mp.Queue,
    dataset: str,
    tile_size: int,
    visualize: bool = False,
    frozen: bool = False,
    initial_best_loss: float = float('inf')
):
    early_stopping_tolerance = 10
    early_stopping_threshold = 0.01
    
    epoch_train_losses: list[dict] = []
    epoch_test_losses: list[dict] = []
    
    # Track accuracies and times for plotting
    train_history: History = History([], [], [], [], [], [])
    val_history: History = History([], [], [], [], [], [])
    
    # Track cumulative images processed for throughput calculation
    cumulative_train_images: int = 0
    cumulative_val_images: int = 0
    
    best_model_wts: "dict[str, torch.Tensor]" = model.state_dict()
    best_raw_model_wts: "dict[str, torch.Tensor]" = model.state_dict()
    best_epoch: int = 0
    best_loss: float = initial_best_loss
    early_stopping_counter: int = 0
    
    throughput_per_epoch: list[list[dict[str, float | int | str]]] = []
    
    # Initialize EMA model for stable evaluation
    # Start with None, will be initialized after warmup epochs
    ema_model = None
    ema_decay = 0.999
    ema_warmup_epochs = 3
    
    # Extract dataset name from training path for description
    max_model_name_length = max(len(name) for name in MODEL_ZOO)
    description = f"{dataset} {tile_size:>3} {model_type:>{max_model_name_length}} {test_loader.batch_size:>4} {'{}'}"
    command_queue.put((device, { 'description': description.format('T'),
                                 'total': n_epochs, 'completed': 0 }))
    
    for epoch in range(n_epochs):
        # Initialize EMA model after warmup epochs (once model has learned something)
        if epoch == ema_warmup_epochs and ema_model is None:
            ema_model = copy.deepcopy(model)
            ema_model.eval()
        
        # Run training epoch with EMA updates (only if EMA initialized)
        train_result = run_training_epoch(model, loss_fn, optimizer, train_loader, 
                                          device, model_type, command_queue, description,
                                          ema_model=ema_model, ema_decay=ema_decay)
        
        # Store training epoch results
        epoch_train_losses.append({
            'op': 'train',
            'loss': train_result['cumulative_loss'],
            'tp': train_result['tp'],
            'tn': train_result['tn'],
            'fp': train_result['fp'],
            'fn': train_result['fn'],
        })
        
        # Format training throughput
        throughput: list[dict[str, float | int | str]] = []
        throughput.extend(format_throughput(train_result, 'train'))
        
        # Update cumulative training images
        cumulative_train_images += train_result['num_samples']
        
        # Run validation epoch using EMA model if available, otherwise use raw model
        validation_model = ema_model if ema_model is not None else model
        with torch.no_grad():
            val_result = run_validation_epoch(validation_model, loss_fn, test_loader, device, model_type, 
                                              command_queue, description)
            
            # Store validation epoch results
            epoch_test_losses.append({
                'op': 'test',
                'loss': val_result['cumulative_loss'],
                'tp': val_result['tp'],
                'tn': val_result['tn'],
                'fp': val_result['fp'],
                'fn': val_result['fn'],
            })
            
            # Format validation throughput
            throughput.extend(format_throughput(val_result, 'val'))
            
            throughput_per_epoch.append(throughput)
            
            # Update cumulative validation images
            cumulative_val_images += val_result['num_samples']
            
            # Update training history for plotting
            train_history.add(train_result)
            val_history.add(val_result)

            # Generate plot at the end of each epoch
            if results_dir and visualize:
                plot_training_progress(train_history, val_history, results_dir,
                                       cumulative_train_images, cumulative_val_images,
                                       throughput_per_epoch, frozen)
            
            # Save best model and check early stopping
            save_model(model, ema_model, results_dir, name=f'model_{epoch:02d}.pth')
            if val_result['cumulative_loss'] < best_loss:
                best_model_wts, best_raw_model_wts = save_model(model, ema_model, results_dir)
                best_epoch = epoch
                best_loss = val_result['cumulative_loss']
                early_stopping_counter = 0
            else:
                should_stop, early_stopping_counter = check_early_stopping(
                    val_result['cumulative_loss'], best_loss, early_stopping_counter,
                    early_stopping_tolerance, early_stopping_threshold)
                
                if should_stop:
                    break
        
        command_queue.put((device, { 'completed': epoch + 1 }))
    
    # Save all training results
    save_training_results(results_dir, epoch_train_losses, epoch_test_losses, throughput_per_epoch)
    
    assert best_model_wts is not None, "Best model weights are None"
    return best_model_wts, best_raw_model_wts, best_loss


MODEL_ZOO = {
    'YoloN': YoloN,
    'YoloS': YoloS,
    'YoloM': YoloM,
    'YoloL': YoloL,
    'YoloX': YoloX,
    # 'ShuffleNet05Q': ShuffleNet05Q_factory,
    'ShuffleNet05': ShuffleNet05_factory,
    'ShuffleNet20': ShuffleNet20_factory,
    'MobileNetL': MobileNetL_factory,
    # 'MobileNetLQ': MobileNetLQ_factory,
    'MobileNetS': MobileNetS_factory,
    'WideResNet50': WideResNet50_factory,
    'WideResNet101': WideResNet101_factory,
    'ResNet152': ResNet152_factory,
    'ResNet101': ResNet101_factory,
    'ResNet18': ResNet18_factory,
    # 'ResNet18Q': ResNet18Q_factory,
    'EfficientNetS': EfficientNetS_factory,
    'EfficientNetL': EfficientNetL_factory,
}


class ImageFolderWithPosition(ImageFolder):
    def __init__(self, root: str, transform: typing.Callable | None = None):
        super().__init__(root, transform)
        self.mem = {}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        if index in self.mem:
            sample, diff_sample, target, pos = self.mem[index]
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
    """
    Train a classifier model for a specific video, tile size, and model type.

    Args:
        dataset: Dataset name
        tile_size: Tile size for the model
        model_type: Model type to use
        visualize: Whether to visualize the training progress
        gpu_id: GPU ID to use for training
        command_queue: Queue for progress updates
    """
    device = f'cuda:{gpu_id}'
    # print(f'Training {model_type} (tile_size={tile_size}) on {device}\n')

    original_training_path = os.path.join(CACHE_DIR, dataset, 'indexing', 'training')

    # Create results directory early so we can save plots during training
    results_dir = os.path.join(original_training_path, 'results', f'{model_type}_{tile_size}')
    os.makedirs(results_dir, exist_ok=True)

    # Use a temporary directory for training_path
    with tempfile.TemporaryDirectory(suffix=f'_polyis_training_data_{model_type}_{tile_size}') as tmpdir:
        # Move training_path to temporary directory
        training_path = os.path.join(tmpdir, 'training')
        # shutil.copytree(original_training_path, training_path)
        training_path = original_training_path

        # Instantiate the correct model based on model_type
        if model_type not in MODEL_ZOO:
            raise ValueError(f"Unsupported model type: {model_type}")
        model = MODEL_ZOO[model_type](tile_size).to(device)
        model = ClassifyImageWithPosition(model, pos_encode_size=16).to(device)

        training_data_path = os.path.join(training_path, 'data', f'tilesize_{tile_size}')

        assert os.path.exists(training_data_path), \
            f"Training data directory {training_data_path} does not exist. " \
            "Please run p012_tune_create_training_data.py first."

        # # Count files directly from directories using simple shell commands
        # neg_dir = os.path.join(training_data_path, 'neg')
        # pos_dir = os.path.join(training_data_path, 'pos')
        
        # # Use ls -1 | wc -l to count files (simple and fast)
        # neg_count = int(subprocess.check_output(['sh', '-c', f'ls -1 {neg_dir} | wc -l'], text=True).strip())
        # pos_count = int(subprocess.check_output(['sh', '-c', f'ls -1 {pos_dir} | wc -l'], text=True).strip())
        
        # # Calculate pos_weight for binary classification (inverse frequency weighting)
        # pos_weight = neg_count / pos_count
        
        # Use BCEWithLogitsLoss with pos_weight to handle class imbalance
        # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
        
        # Apply ImageNet normalization to all 6 channels (3 from image + 3 from diff)
        # This matches the pretrained models' expected input distribution
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

        # Try-Catch to lower the batch size if fails.
        batch_size = 256
        min_batch_size = 8
        best_model_wts = None

        run_train = partial(train, results_dir=results_dir, model_type=model_type,
                            device=device, tile_size=tile_size, visualize=visualize,
                            command_queue=command_queue, dataset=dataset)

        while batch_size > min_batch_size:
            try:
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

                # Calculate learning rates based on batch size (linear scaling rule)
                # Base LR for batch_size=256: lr_frozen=1e-3, lr_unfrozen=1e-4
                # Scale linearly with batch size
                base_batch_size = 256
                lr_scale = batch_size / base_batch_size
                lr_frozen = 1e-3 * lr_scale
                lr_unfrozen = 1e-4 * lr_scale

                # # Scale down the learning rate
                lr_frozen = lr_frozen * 0.5
                lr_unfrozen = lr_unfrozen * 0.5
                
                # Stage 1: Train with base model frozen (higher learning rate)
                # Train adapter, pos_encoder, and classifier head
                model.freeze_base_model()
                optimizer_frozen = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen, weight_decay=0.01)
                _, best_raw_wts_stage1, best_loss_stage1 = run_train(model, loss_fn, optimizer_frozen, train_loader, test_loader, n_epochs=25, frozen=True)
                
                # Load best raw model weights from stage 1 (not EMA) to continue training
                model.load_state_dict(best_raw_wts_stage1)
                
                # Stage 2: Fine-tune with base model unfrozen (lower learning rate)
                # Train entire network with much lower learning rate
                # Pass best_loss from stage 1 to ensure stage 2 only saves if it improves
                model.unfreeze_base_model()
                optimizer_unfrozen = AdamW(model.parameters(), lr=lr_unfrozen, weight_decay=0.01)
                run_train(model, loss_fn, optimizer_unfrozen, train_loader, test_loader, n_epochs=50, initial_best_loss=best_loss_stage1)
                
                break  # Success, exit the retry loop

            except Exception as e:
                # Reduce batch size by half for next attempt
                batch_size = batch_size // 2
                if batch_size < min_batch_size:
                    print(f"Batch size reduced to minimum (1) for {model_type} (tile_size={tile_size}), but still failing")
                    raise e
                # Clear GPU memory before retry (only for the current device)
                if torch.cuda.is_available() and device.startswith('cuda:'):
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()


def main(args):
    mp.set_start_method('spawn', force=True)

    funcs: list[partial] = []

    for dataset_name in DATASETS:
        dataset_dir = os.path.join(CACHE_DIR, dataset_name)

        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist, skipping...")
            continue

        # Use dataset-level training data instead of video-level
        results_dir = os.path.join(dataset_dir, 'indexing', 'training', 'results')
        if args.clear and os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        for classifier in CLASSIFIERS:
            for tile_size in TILE_SIZES:
                func = partial(train_classifier, dataset_name, tile_size, classifier, args.visualize)
                funcs.append(func)

    # Set up multiprocessing with ProgressBar
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"

    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs), refresh_per_second=5).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
