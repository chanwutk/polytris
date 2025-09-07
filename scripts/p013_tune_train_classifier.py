#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import time
import multiprocessing as mp

import torch
import torch.utils.data
import torch.optim
from rich.progress import track
from rich import progress
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.optim import Adam

from polyis.models.classifier.simple_cnn import SimpleCNN
from polyis.models.classifier.yolo import YoloN, YoloS, YoloM, YoloL, YoloX
from polyis.models.classifier.shufflenet import ShuffleNet05, ShuffleNet20
from polyis.models.classifier.mobilenet import MobileNetL, MobileNetS
from polyis.models.classifier.wide_resnet import WideResNet50, WideResNet101
from polyis.models.classifier.resnet import ResNet152, ResNet101, ResNet18
from polyis.models.classifier.efficientnet import EfficientNetS, EfficientNetL
from scripts.utilities import CACHE_DIR, format_time


# Factory functions for models that don't accept tile_size parameter
def ShuffleNet05_factory(_tile_size: int):
    return ShuffleNet05()

def ShuffleNet20_factory(_tile_size: int):
    return ShuffleNet20()

def MobileNetL_factory(_tile_size: int):
    return MobileNetL()

def MobileNetS_factory(_tile_size: int):
    return MobileNetS()

def WideResNet50_factory(_tile_size: int):
    return WideResNet50()

def WideResNet101_factory(_tile_size: int):
    return WideResNet101()

def ResNet152_factory(_tile_size: int):
    return ResNet152()

def ResNet101_factory(_tile_size: int):
    return ResNet101()

def ResNet18_factory(_tile_size: int):
    return ResNet18()

def EfficientNetS_factory(_tile_size: int):
    return EfficientNetS()

def EfficientNetL_factory(_tile_size: int):
    return EfficientNetL()


TILE_SIZES = [30, 60, 120]


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--classifier', required=False,
                        default=['ResNet18', 'ResNet152', 'ResNet101',
                                 'EfficientNetS', 'EfficientNetL',
                                 'ShuffleNet05', 'ShuffleNet20', 'MobileNetL',
                                 'MobileNetS', 'WideResNet50',], # 'WideResNet101',
                        choices=['SimpleCNN', 'YoloN', 'YoloS', 'YoloM', 'YoloL',
                                 'YoloX', 'ShuffleNet05', 'ShuffleNet20', 'MobileNetL',
                                 'MobileNetS', 'WideResNet50', 'WideResNet101',
                                 'ResNet152', 'ResNet101', 'ResNet18', 'EfficientNetS',
                                 'EfficientNetL'],
                        nargs='+',
                        help='Model types to train (can specify multiple): SimpleCNN, '
                             'YoloN, YoloS, YoloM, YoloL, YoloX, ShuffleNet05, '
                             'ShuffleNet20, MobileNetL, MobileNetS, WideResNet50, '
                             'WideResNet101, ResNet152, ResNet101, ResNet18, '
                             'EfficientNetS, EfficientNetL')
    return parser.parse_args()


def plot_training_progress(train_losses: list[float], train_accuracies: list[float],
                           val_losses: list[float], val_accuracies: list[float],
                           train_times: list[float], val_times: list[float],
                           results_dir: str, epoch: int, train_images_processed: int,
                           val_images_processed: int, throughput_per_epoch: list[list[dict[str, float | int | str]]]):
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

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: Loss and accuracy progress
    epoch_range = list(range(len(train_losses)))
    ax1.plot(epoch_range, train_losses,
             'b-', label='Train Loss', marker='o', linewidth=2)
    ax1.plot(epoch_range, val_losses,
             'r-', label='Validation Loss', marker='s', linewidth=2)
    ax1.plot(epoch_range, train_accuracies_scaled,
             'b--', label='Train Accuracy', marker='^', linewidth=2)
    ax1.plot(epoch_range, val_accuracies_scaled,
             'r--', label='Validation Accuracy', marker='d', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss / Accuracy')
    ax1.set_title(f'Training Progress - Epoch {epoch + 1}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Right subplot: ms/frame stacked bar chart
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

    # Prepare for plotting
    op_names = sorted(list(all_ops))
    legend_labels = { op: op.replace('train_', '').replace('test_', '')
                      for op in op_names }
    op_names = sorted(op_names,
                      key=lambda op:
                        op.replace('test_', 'train_') in train_op_times and
                        op.replace('train_', 'test_') in val_op_times,
                      reverse=True)

    # Using a colormap that is visually distinct
    colors = plt.cm.get_cmap('tab10', len(op_names))
    color_label_map = {op: colors(i) for i, op in enumerate(legend_labels.values())}
    color_map = {op: color_label_map[legend_labels[op]] for op in op_names}

    # Calculate ms per frame for each operation
    # ms/frame for op = (total op time in ms) / (images processed)
    bottom_train = 0.0
    for op in op_names:
        if op in train_op_times:
            op_time_ms = train_op_times[op]
            op_ms_per_frame = (op_time_ms / train_images_processed)
            ax2.bar('Training', op_ms_per_frame, bottom=bottom_train,
                    label=legend_labels[op], color=color_map[op])
            bottom_train += op_ms_per_frame

    bottom_val = 0.0
    for op in op_names:
        if op in val_op_times:
            op_time_ms = val_op_times[op]
            op_ms_per_frame = (op_time_ms / val_images_processed)
            ax2.bar('Validation', op_ms_per_frame, bottom=bottom_val,
                    label=legend_labels[op], color=color_map[op])
            bottom_val += op_ms_per_frame

    ax2.set_ylabel('ms per frame')
    ax2.set_title('Milliseconds per Frame by Operation')
    ax2.grid(True, alpha=0.3, axis='y')

    # To avoid duplicate labels in legend
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(results_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


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
    grey = "[grey50]"
    newline = " ┗━"

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

    video_filename = training_path.split('/')[-2].split('.')[0]
    max_model_name_length = max(len(name) for name in MODEL_ZOO)
    description = f"{video_filename} {tile_size:>3} {model_type:>{max_model_name_length}}"
    command_queue.put((device + ':epoch', { 'description': description, 'total': n_epochs, 'completed': 0 }))
    for epoch in range(n_epochs):
        epoch_loss = 0

        # Record training start time
        train_start_time = time.time_ns() / 1e6

        throughput: list[dict[str, float | int | str]] = []

        # Initialize timing accumulators for this epoch
        total_data_loading_time: float = 0.
        total_gpu_transfer_time: float = 0.
        total_train_step_time: float = 0.

        # Track training accuracy
        train_correct: int = 0
        train_total: int = 0

        model.train()

        # Measure data loading time by tracking iterator timing
        batch_start_time = time.time_ns() / 1e6

        train_description = ' Training'
        dash = '━' * (len(description) - len(newline) - len(train_description))
        command_queue.put((device, { 'description': grey + newline + dash + train_description,
                                     'total': len(train_loader), 'completed': 0 }))
        for x_batch, y_batch in train_loader:
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

            # Calculate training accuracy
            with torch.no_grad():
                if model_type.startswith('Yolo'):
                    if y_hat.dim() == 1:
                        y_hat = y_hat.unsqueeze(1)
                predictions = y_hat > 0.5
                train_correct += int(torch.sum(predictions == y_batch).item())
                train_total += len(y_batch)

            command_queue.put((device, { 'advance': 1 }))

            # Start timing for next batch data loading
            batch_start_time = time.time_ns() / 1e6
            # break

        # Record training end time
        train_time = (time.time_ns() / 1e6) - train_start_time
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0

        epoch_train_losses.append({
            'op': 'train',
            'loss': float(epoch_loss),
            'time': train_time,
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

        # # Print detailed timing information
        # print('Epoch : {}, train loss : {:.4f}, train accuracy: {:.1f}%'.format(epoch+1, epoch_loss, train_accuracy))
        # print('  Total time: {:.2f}s'.format(train_time))
        # print('  Data loading: {:.2f}s ({:.1f}%)'.format(total_data_loading_time, 100 * total_data_loading_time / train_time))
        # print('  GPU transfer: {:.2f}s ({:.1f}%)'.format(total_gpu_transfer_time, 100 * total_gpu_transfer_time / train_time))
        # print('  Train step: {:.2f}s ({:.1f}%)'.format(total_train_step_time, 100 * total_train_step_time / train_time))
        # print()

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
            model.eval()

            # Measure validation data loading time
            val_batch_start_time = time.time_ns() / 1e6

            val_description = " Validation"
            dash = '━' * (len(description) - len(newline) - len(val_description))
            command_queue.put((device, { 'description': grey + newline + dash + val_description,
                                         'total': len(test_loader), 'completed': 0 }))
            for x_batch, y_batch in test_loader:
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

                # Apply appropriate activation function based on model type
                ans = yhat
                ans = ans > 0.5
                misc = torch.sum(ans == y_batch)
                misc_sum += misc.item()
                num_samples += len(y_batch)
                # print(f"Accuracy: {misc.item() * 100 / len(y_batch)} %\n")

                command_queue.put((device, { 'advance': 1 }))

                # Start timing for next validation batch data loading
                val_batch_start_time = time.time_ns() / 1e6

            # Record validation end time
            val_time = (time.time_ns() / 1e6) - val_start_time # no need to add to total time
            val_accuracy = misc_sum * 100 / num_samples if num_samples > 0 else 0

            epoch_test_losses.append({
                'op': 'test',
                'loss': float(cumulative_loss),
                'time': val_time,
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

            # # Print detailed validation timing information
            # print('Validation - Loss: {:.4f}, Accuracy: {:.1f}%'.format(cumulative_loss, val_accuracy))
            # print('  Total time: {:.2f}s'.format(val_time))
            # print('  Data loading: {:.2f}s ({:.1f}%)'.format(val_total_data_loading_time, 100 * val_total_data_loading_time / val_time))
            # print('  GPU transfer: {:.2f}s ({:.1f}%)'.format(val_total_gpu_transfer_time, 100 * val_total_gpu_transfer_time / val_time))
            # print('  Inference: {:.2f}s ({:.1f}%)'.format(val_total_inference_time, 100 * val_total_inference_time / val_time))
            # print()

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
        
        command_queue.put((device + ':epoch', { 'advance': 1 }))

    # print(str(epoch_test_losses) + '\n')
    # print(str(epoch_train_losses) + '\n')

    # # Calculate total training and validation times
    # total_train_time = sum(epoch['time'] for epoch in epoch_train_losses)
    # total_val_time = sum(epoch['time'] for epoch in epoch_test_losses)

    # print(f'Total validation time: {total_val_time:.2f}s')
    # print(f'Total time: {total_train_time + total_val_time:.2f}s')

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
                     device: str, command_queue: mp.Queue):
    # print(f'Training {model_type} (tile_size={tile_size}) on {device}\n')

    # Create results directory early so we can save plots during training
    results_dir = os.path.join(training_path, 'results', f'{model_type}_{tile_size}')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    # Instantiate the correct model based on model_type
    if model_type not in MODEL_ZOO:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = MODEL_ZOO[model_type](tile_size).to(device)
    loss_fn = torch.nn.BCELoss().to(device)

    optimizer = Adam(model.parameters(), lr=0.001)

    training_data_path = os.path.join(training_path, 'data', f'tilesize_{tile_size}')
    train_data = datasets.ImageFolder(training_data_path, transform=transforms.ToTensor())

    generator = torch.Generator().manual_seed(0)
    split = int(0.8 * len(train_data))
    train_data, test_data = torch.utils.data.random_split(
        dataset=train_data,
        lengths=[split, len(train_data) - split],
        generator=generator
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=True)

    # print(f"Training {model_type} (tile_size={tile_size})")
    best_model_wts = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=50,
                           results_dir=results_dir, model_type=model_type, device=device,
                           command_queue=command_queue, training_path=training_path, tile_size=tile_size)
    assert best_model_wts is not None

    # Load best model
    model.load_state_dict(best_model_wts)
    with open(os.path.join(results_dir, 'model.pth'), 'wb') as f:
        torch.save(model, f)


def progress_bars(command_queue: mp.Queue, num_tasks: int):
    with progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        # progress.TimeRemainingColumn(),
        progress.MofNCompleteColumn(),
        progress.TimeElapsedColumn(),
        refresh_per_second=0.5,
    ) as p:
        bars: dict[str, progress.TaskID] = {}
        overall_progress = p.add_task(f"[green]Training {num_tasks} models", total=num_tasks)
        bars['overall'] = overall_progress
        for gpu_id in range(torch.cuda.device_count()):
            bars[f'cuda:{gpu_id}:epoch'] = p.add_task("jnc00  30 model")
            bars[f'cuda:{gpu_id}'] = p.add_task("  Training")

        while True:
            val = command_queue.get()
            if val is None: break
            progress_id, kwargs = val
            p.update(bars[progress_id], **kwargs)
        
        # remove all tasks
        for _, task_id in bars.items():
            p.remove_task(task_id)
        bars.clear()



def dispatch_task(training_path: str, tile_size: int, classifier: str,
                  gpu_id: int, queue: mp.Queue, command_queue: mp.Queue):
    try:
        train_classifier(training_path, tile_size, classifier,
                         device=f'cuda:{gpu_id}', command_queue=command_queue)
    finally: queue.put(gpu_id)


def main(args):
    mp.set_start_method('spawn', force=True)

    dataset_dir = os.path.join(CACHE_DIR, args.dataset)

    tasks: list[tuple[str, int, str]] = []
    for video in sorted(os.listdir(dataset_dir)):
        video_path = os.path.join(dataset_dir, video)
        if not os.path.isdir(video_path) and not video.endswith('.mp4'):
            continue

        # print(f"Processing video {video_path}")
        for tile_size in TILE_SIZES:
            training_path = os.path.join(video_path, 'training')
            for classifier in args.classifier:
                tasks.append((training_path, tile_size, classifier))

    num_gpus = torch.cuda.device_count()
    gpu_id_queue = mp.Queue(maxsize=num_gpus)
    for gpu_id in range(num_gpus):
        gpu_id_queue.put(gpu_id)

    command_queue = mp.Queue()
    progress_process = mp.Process(target=progress_bars,
                                  args=(command_queue, len(tasks)),
                                  daemon=True)
    progress_process.start()

    processes: list[mp.Process] = []
    for task in tasks:
        # print(task)
        gpu_id = gpu_id_queue.get()
        command_queue.put(( 'overall', { 'advance': 1 } ))
        process = mp.Process(target=dispatch_task,
                             args=(*task, gpu_id, gpu_id_queue, command_queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        process.terminate()

    command_queue.put(None)
    progress_process.join()
    progress_process.terminate()


if __name__ == '__main__':
    main(parse_args())
