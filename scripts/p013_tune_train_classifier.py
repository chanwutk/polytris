#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import subprocess
import time
import multiprocessing as mp
from functools import partial
from typing import Callable

import torch
import torch.utils.data
import torch.optim
import altair as alt
import pandas as pd

from torchvision import datasets, transforms
from torch.optim import Adam

from polyis.models.classifier.simple_cnn import SimpleCNN
from polyis.models.classifier.yolo import YoloN, YoloS, YoloM, YoloL, YoloX
from polyis.utilities import CACHE_DIR, CLASSIFIERS_CHOICES, CLASSIFIERS_TO_TEST, format_time, ProgressBar, DATASETS_TO_TEST, TILE_SIZES

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


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--datasets', required=False,
                        default=DATASETS_TO_TEST,
                        nargs='+',
                        help='Dataset names (space-separated)')
    parser.add_argument('--classifiers', required=False,
                        # default=['WideResNet50'],
                        default=CLASSIFIERS_TO_TEST,
                        choices=CLASSIFIERS_CHOICES,
                        nargs='+',
                        help='Model types to train (can specify multiple). For '
                             'example: --classifiers YoloN ShuffleNet05 ResNet18')
    parser.add_argument('--clear', action='store_true',
                        help='Clear existing results directories before training')
    return parser.parse_args()


def plot_training_progress(train_losses: list[float], train_accuracies: list[float],
                           val_losses: list[float], val_accuracies: list[float],
                           train_times: list[float], val_times: list[float],
                           results_dir: str, epoch: int, train_images_processed: int,
                           val_images_processed: int,
                           throughput_per_epoch: list[list[dict[str, float | int | str]]]):
    """Plot training progress with time on x-axis and loss/accuracy on y-axis"""
    # Calculate cumulative times
    cumulative_train_times: list[float] = []
    cumulative_val_times: list[float] = []
    total_time: float = 0

    for i in range(len(train_times)):
        total_time += train_times[i]
        cumulative_train_times.append(total_time / 1000)  # Convert to seconds
        total_time += val_times[i]
        cumulative_val_times.append(total_time / 1000)  # Convert to seconds

    # Convert accuracies from percentage to [0, 1] scale
    train_accuracies_scaled: list[float] = [acc / 100.0 for acc in train_accuracies]
    val_accuracies_scaled: list[float] = [acc / 100.0 for acc in val_accuracies]

    # Prepare data for loss and accuracy plot
    epoch_range = list(range(len(train_losses)))

    # Create DataFrame for loss and accuracy data
    plot_data = []
    for i, epoch_num in enumerate(epoch_range):
        plot_data.extend([
            {'Epoch': epoch_num, 'Value': train_losses[i], 'Metric': 'Train Loss', 'Type': 'Loss'},
            {'Epoch': epoch_num, 'Value': val_losses[i], 'Metric': 'Validation Loss', 'Type': 'Loss'},
            {'Epoch': epoch_num, 'Value': train_accuracies_scaled[i], 'Metric': 'Train Accuracy', 'Type': 'Accuracy'},
            {'Epoch': epoch_num, 'Value': val_accuracies_scaled[i], 'Metric': 'Validation Accuracy', 'Type': 'Accuracy'}
        ])

    df = pd.DataFrame(plot_data)

    # Left chart: Loss and accuracy progress
    chart1 = alt.Chart(df).mark_line(point=True).encode(
        x='Epoch:Q',
        y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 1])),
        color='Metric:N',
        strokeDash='Type:N'
    ).properties(
        title=f'Training Progress - Epoch {epoch + 1}',
        width=400,
        height=300
    )

    # Right chart: ms/frame stacked bar chart
    train_op_times: dict[str, float] = {}
    val_op_times: dict[str, float] = {}
    all_ops: set[str] = set()

    for epoch_throughput in throughput_per_epoch:
        for op_data in epoch_throughput:
            op_name = str(op_data['op'])
            op_time = float(op_data['time'])  # in ms
            all_ops.add(op_name)
            if op_name.startswith('train'):
                train_op_times[op_name] = train_op_times.get(op_name, 0) + op_time
            else:
                val_op_times[op_name] = val_op_times.get(op_name, 0) + op_time

    # Prepare data for stacked bar chart
    op_names = sorted(list(all_ops))
    legend_labels = { op: op.replace('train_', '').replace('test_', '')
                      for op in op_names }

    # Create data for stacked bar chart
    bar_data = []

    # Training data
    for op in op_names:
        if op in train_op_times and not op.endswith('load_data'):
            op_time_ms = train_op_times[op]
            op_ms_per_frame = (op_time_ms / train_images_processed)
            bar_data.append({
                'Phase': 'Training',
                'Operation': legend_labels[op],
                'ms_per_frame': op_ms_per_frame
            })

    # Validation data
    for op in op_names:
        if op in val_op_times and not op.endswith('load_data'):
            op_time_ms = val_op_times[op]
            op_ms_per_frame = (op_time_ms / val_images_processed)
            bar_data.append({
                'Phase': 'Validation',
                'Operation': legend_labels[op],
                'ms_per_frame': op_ms_per_frame
            })

    if bar_data:  # Only create the plot if we have data
        bar_df = pd.DataFrame(bar_data)

        # Create stacked bar chart
        chart2 = alt.Chart(bar_df).mark_bar().encode(
            x='Phase:N',
            y='ms_per_frame:Q',
            color='Operation:N',
            tooltip=['Phase', 'Operation', alt.Tooltip('ms_per_frame:Q', format='.2f')]
        ).properties(
            title='Milliseconds per Frame by Operation',
            width=400,
            height=300
        )

        # Combine charts horizontally
        combined_chart = alt.hconcat(chart1, chart2, spacing=20)
    else:
        combined_chart = chart1

    # Save the plot
    plot_path = os.path.join(results_dir, 'training_progress.png')
    combined_chart.save(plot_path, scale_factor=2)


def train_step(model: "torch.nn.Module", loss_fn: "torch.nn.modules.loss._Loss",
               optimizer: "torch.optim.Optimizer", inputs: "torch.Tensor",
               labels: "torch.Tensor", model_type: str = 'SimpleCNN'):
    optimizer.zero_grad()

    outputs: "torch.Tensor" = model(inputs)

    # Handle different output formats
    if model_type.startswith('Yolo'):
        # YOLO outputs probabilities directly, ensure they're in the right shape
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)

    loss: "torch.Tensor" = loss_fn(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item(), outputs


def train(model: "torch.nn.Module", loss_fn: "torch.nn.modules.loss._Loss",
          optimizer: "torch.optim.Optimizer", train_loader: "torch.utils.data.DataLoader",
          test_loader: "torch.utils.data.DataLoader", n_epochs: int, results_dir: str,
          model_type: str, device: str, command_queue: mp.Queue, training_path: str, tile_size: int):
    early_stopping_tolerance = 10
    early_stopping_threshold = 0.001

    epoch_train_losses: list[dict] = []
    epoch_test_losses: list[dict] = []

    # Track accuracies and times for plotting
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    val_accuracy_history: list[float] = []
    train_time_history: list[float] = []
    val_time_history: list[float] = []

    # Track cumulative images processed for throughput calculation
    cumulative_train_images: int = 0
    cumulative_val_images: int = 0

    best_model_wts: "dict[str, torch.Tensor] | None" = None
    best_loss: float = float('inf')
    early_stopping_counter: int = 0

    throughput_per_epoch: list[list[dict[str, float | int | str]]] = []

    # Extract dataset name from training path for description
    dataset_name = training_path.split('/')[-3]  # indexing/training -> dataset_name
    max_model_name_length = max(len(name) for name in MODEL_ZOO)
    description = f"{dataset_name} {tile_size:>3} {model_type:>{max_model_name_length}} {'{}'}"
    command_queue.put((device, { 'description': description.format('T'),
                                 'total': n_epochs, 'completed': 0 }))
    for epoch in range(n_epochs):
        epoch_loss = 0

        # Record training start time
        train_start_time = time.time_ns() / 1e6

        throughput: list[dict[str, float | int | str]] = []

        # Initialize timing accumulators for this epoch
        total_data_loading_time: float = 0.
        total_gpu_transfer_time: float = 0.
        total_train_step_time: float = 0.

        # Track training accuracy and confusion matrix metrics
        train_correct: int = 0
        train_total: int = 0
        train_tp: int = 0
        train_tn: int = 0
        train_fp: int = 0
        train_fn: int = 0

        model.train()

        # Measure data loading time by tracking iterator timing
        batch_start_time = time.time_ns() / 1e6

        command_queue.put((device, { 'description': description.format('T  0%') }))
        for idx, (x_batch, y_batch) in enumerate(train_loader):
            # Data loading time (time since last batch completed or epoch started)
            total_data_loading_time += (time.time_ns() / 1e6) - batch_start_time

            # Measure GPU transfer time
            gpu_transfer_start_time = time.time_ns() / 1e6
            x_batch = x_batch.to(device) # move to gpu
            y_batch = y_batch.to(device).unsqueeze(1).float() # convert target to same nn output shape
            total_gpu_transfer_time += (time.time_ns() / 1e6) - gpu_transfer_start_time

            # Measure train step time
            train_step_start_time = time.time_ns() / 1e6
            loss, y_hat = train_step(model, loss_fn, optimizer, x_batch, y_batch, model_type)
            total_train_step_time += (time.time_ns() / 1e6) - train_step_start_time

            epoch_loss += loss / len(train_loader)

            # Calculate training accuracy and confusion matrix metrics
            with torch.no_grad():
                if model_type.startswith('Yolo'):
                    if y_hat.dim() == 1:
                        y_hat = y_hat.unsqueeze(1)
                # Apply sigmoid to convert logits to probabilities for accuracy calculation
                y_hat_probs = torch.sigmoid(y_hat)
                predictions = y_hat_probs > 0.5
                train_correct += int(torch.sum(predictions == y_batch).item())
                train_total += len(y_batch)

                # Calculate confusion matrix metrics
                batch_tp = torch.sum((predictions == 1) & (y_batch == 1)).item()
                batch_tn = torch.sum((predictions == 0) & (y_batch == 0)).item()
                batch_fp = torch.sum((predictions == 1) & (y_batch == 0)).item()
                batch_fn = torch.sum((predictions == 0) & (y_batch == 1)).item()

                # Accumulate confusion matrix metrics
                train_tp += int(batch_tp)
                train_tn += int(batch_tn)
                train_fp += int(batch_fp)
                train_fn += int(batch_fn)

            # Start timing for next batch data loading
            batch_start_time = time.time_ns() / 1e6
            desc = f'T {int(idx * 100 / len(train_loader)):>2}%'
            command_queue.put((device, { 'description': description.format(desc) }))

        # Record training end time
        train_time = (time.time_ns() / 1e6) - train_start_time
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0

        epoch_train_losses.append({
            'op': 'train',
            'loss': float(epoch_loss),
            'accuracy': float(train_accuracy),
            'time': train_time,
            'tp': train_tp,
            'tn': train_tn,
            'fp': train_fp,
            'fn': train_fn,
        })
        throughput.extend(format_time(
            train_load_data=total_data_loading_time,
            train_gpu_transfer=total_gpu_transfer_time,
            train_step=total_train_step_time))

        # Store for plotting
        train_loss_history.append(float(epoch_loss))
        train_accuracy_history.append(float(train_accuracy))
        train_time_history.append(train_time)

        # Update cumulative training images
        cumulative_train_images += train_total

        # validation doesnt requires gradient
        with torch.no_grad():
            cumulative_loss = 0

            # Record validation start time
            val_start_time = time.time_ns() / 1e6

            # Initialize timing accumulators for validation
            val_total_data_loading_time = 0
            val_total_gpu_transfer_time = 0
            val_total_inference_time = 0
            val_total_loss_time = 0

            misc_sum = 0
            num_samples = 0
            val_tp = 0
            val_tn = 0
            val_fp = 0
            val_fn = 0
            model.eval()

            # Measure validation data loading time
            val_batch_start_time = time.time_ns() / 1e6

            command_queue.put((device, { 'description': description.format('V 00%') }))
            for idx, (x_batch, y_batch) in enumerate(test_loader):
                # Validation data loading time
                val_total_data_loading_time += (time.time_ns() / 1e6) - val_batch_start_time

                # Measure validation GPU transfer time
                val_gpu_transfer_start_time = time.time_ns() / 1e6
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1).float() # convert target to same nn output shape
                val_total_gpu_transfer_time += (time.time_ns() / 1e6) - val_gpu_transfer_start_time

                # Measure inference time
                val_inference_start_time = time.time_ns() / 1e6
                yhat = model(x_batch)
                val_total_inference_time += (time.time_ns() / 1e6) - val_inference_start_time

                # Handle different output formats
                if model_type.startswith('Yolo'):
                    # YOLO outputs probabilities directly, ensure they're in the right shape
                    if yhat.dim() == 1:
                        yhat = yhat.unsqueeze(1)

                val_loss_start_time = time.time_ns() / 1e6
                val_loss = loss_fn(yhat,y_batch)
                cumulative_loss += val_loss / len(test_loader)
                val_total_loss_time += (time.time_ns() / 1e6) - val_loss_start_time

                # Apply sigmoid to convert logits to probabilities for accuracy calculation
                yhat_probs = torch.sigmoid(yhat)
                ans = yhat_probs > 0.5
                misc = torch.sum(ans == y_batch)
                misc_sum += misc.item()
                num_samples += len(y_batch)

                # Calculate confusion matrix metrics for validation
                batch_tp = torch.sum((ans == 1) & (y_batch == 1)).item()
                batch_tn = torch.sum((ans == 0) & (y_batch == 0)).item()
                batch_fp = torch.sum((ans == 1) & (y_batch == 0)).item()
                batch_fn = torch.sum((ans == 0) & (y_batch == 1)).item()

                val_tp += int(batch_tp)
                val_tn += int(batch_tn)
                val_fp += int(batch_fp)
                val_fn += int(batch_fn)
                # print(f"Accuracy: {misc.item() * 100 / len(y_batch)} %\n")

                # Start timing for next validation batch data loading
                val_batch_start_time = time.time_ns() / 1e6
                desc = f'V {int(idx * 100 / len(test_loader)):>2}%'
                command_queue.put((device, { 'description': description.format(desc) }))

            # Record validation end time
            val_time = (time.time_ns() / 1e6) - val_start_time # no need to add to total time
            val_accuracy = misc_sum * 100 / num_samples if num_samples > 0 else 0

            epoch_test_losses.append({
                'op': 'test',
                'loss': float(cumulative_loss),
                'accuracy': float(val_accuracy),
                'time': val_time,
                'tp': val_tp,
                'tn': val_tn,
                'fp': val_fp,
                'fn': val_fn,
            })
            throughput.extend(format_time(
                test_load_data=val_total_data_loading_time,
                test_gpu_transfer=val_total_gpu_transfer_time,
                test_inference=val_total_inference_time,
                test_loss=val_total_loss_time))

            throughput_per_epoch.append(throughput)

            # Store for plotting
            val_loss_history.append(float(cumulative_loss))
            val_accuracy_history.append(float(val_accuracy))
            val_time_history.append(val_time)

            # Update cumulative validation images
            cumulative_val_images += num_samples

            # Generate plot at the end of each epoch
            if results_dir:
                plot_training_progress(train_loss_history, train_accuracy_history,
                                     val_loss_history, val_accuracy_history,
                                     train_time_history, val_time_history,
                                     results_dir, epoch, cumulative_train_images, cumulative_val_images,
                                     throughput_per_epoch)

            # save best model
            if cumulative_loss < best_loss:
                best_model_wts = model.state_dict()
                best_loss = cumulative_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if (early_stopping_counter >= early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
                # print("Terminating: early stopping")
                break # terminate training

        command_queue.put((device, { 'completed': epoch + 1 }))

    with open(os.path.join(results_dir, 'test_losses.json'), 'w') as f:
        f.write(json.dumps(epoch_test_losses))

    with open(os.path.join(results_dir, 'train_losses.json'), 'w') as f:
        f.write(json.dumps(epoch_train_losses))

    with open(os.path.join(results_dir, 'throughput_per_epoch.jsonl'), 'w') as f:
        for t in throughput_per_epoch:
            f.write(json.dumps(t) + '\n')

    return best_model_wts


MODEL_ZOO = {
    'SimpleCNN': SimpleCNN,
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


def train_classifier(training_path: str, tile_size: int, model_type: str,
                     gpu_id: int, command_queue: mp.Queue):
    """
    Train a classifier model for a specific video, tile size, and model type.

    Args:
        training_path: Path to the training data directory
        tile_size: Tile size for the model
        model_type: Model type to use
        gpu_id: GPU ID to use for training
        command_queue: Queue for progress updates
    """
    device = f'cuda:{gpu_id}'
    # print(f'Training {model_type} (tile_size={tile_size}) on {device}\n')

    # Create results directory early so we can save plots during training
    results_dir = os.path.join(training_path, 'results', f'{model_type}_{tile_size}')
    os.makedirs(results_dir, exist_ok=True)

    # Instantiate the correct model based on model_type
    if model_type not in MODEL_ZOO:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = MODEL_ZOO[model_type](tile_size).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)

    training_data_path = os.path.join(training_path, 'data', f'tilesize_{tile_size}')

    assert os.path.exists(training_data_path), \
        f"Training data directory {training_data_path} does not exist. " \
        "Please run p012_tune_create_training_data.py first."

    # Count files directly from directories using simple shell commands
    neg_dir = os.path.join(training_data_path, 'neg')
    pos_dir = os.path.join(training_data_path, 'pos')
    
    # Use ls -1 | wc -l to count files (simple and fast)
    neg_count = int(subprocess.check_output(['sh', '-c', f'ls -1 {neg_dir} | wc -l'], text=True).strip())
    pos_count = int(subprocess.check_output(['sh', '-c', f'ls -1 {pos_dir} | wc -l'], text=True).strip())
    
    # Calculate pos_weight for binary classification (inverse frequency weighting)
    pos_weight = neg_count / pos_count
    
    # Use BCEWithLogitsLoss with pos_weight to handle class imbalance
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)).to(device)
    
    train_data = datasets.ImageFolder(training_data_path, transform=transforms.ToTensor())

    generator = torch.Generator().manual_seed(0)
    split = int(0.8 * len(train_data))
    train_data, test_data = torch.utils.data.random_split(
        dataset=train_data,
        lengths=[split, len(train_data) - split],
        generator=generator
    )

    # Try-Catch to lower the batch size if fails.
    batch_size = 512
    max_retries = 4
    retry_count = 0
    best_model_wts = None

    while retry_count <= max_retries:
        try:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

            # print(f"Training {model_type} (tile_size={tile_size}) with batch_size={batch_size}")
            best_model_wts = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=50,
                                   results_dir=results_dir, model_type=model_type, device=device,
                                   command_queue=command_queue, training_path=training_path, tile_size=tile_size)
            break  # Success, exit the retry loop

        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Training failed after {max_retries} retries for {model_type} (tile_size={tile_size}): {e}")
                raise e

            # Reduce batch size by half for next attempt
            batch_size = batch_size // 2
            if batch_size < 1:
                print(f"Batch size reduced to minimum (1) for {model_type} (tile_size={tile_size}), but still failing")
                raise e

            # Clear GPU memory before retry (only for the current device)
            if torch.cuda.is_available() and device.startswith('cuda:'):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    assert best_model_wts is not None, f"Training failed after {max_retries} retries for {model_type} (tile_size={tile_size})"

    # Load best model
    model.load_state_dict(best_model_wts)
    with open(os.path.join(results_dir, 'model.pth'), 'wb') as f:
        torch.save(model, f)


def main(args):
    mp.set_start_method('spawn', force=True)

    funcs: list[Callable[[int, mp.Queue], None]] = []

    for dataset_name in args.datasets:
        dataset_dir = os.path.join(CACHE_DIR, dataset_name)

        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist, skipping...")
            continue

        # Use dataset-level training data instead of video-level
        training_path = os.path.join(dataset_dir, 'indexing', 'training')
        results_dir = os.path.join(training_path, 'results')

        if args.clear and os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        for classifier in args.classifiers:
            for tile_size in TILE_SIZES:
                func = partial(train_classifier, training_path, tile_size, classifier)
                funcs.append(func)

    # Set up multiprocessing with ProgressBar
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"

    ProgressBar(num_workers=num_gpus, num_tasks=len(funcs)).run_all(funcs)


if __name__ == '__main__':
    main(parse_args())
