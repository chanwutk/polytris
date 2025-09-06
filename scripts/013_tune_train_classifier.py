#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import time

import torch
import torch.utils.data
import torch.optim
from rich.progress import track
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.optim import Adam

from polyis.models.classifier.simple_cnn import SimpleCNN
from polyis.models.classifier.yolo import YoloN, YoloS, YoloM, YoloL, YoloX
from scripts.utilities import CACHE_DIR, format_time


TILE_SIZES = [30, 60, 120]


def plot_training_progress(train_losses: list[float], train_accuracies: list[float],
                           val_losses: list[float], val_accuracies: list[float], 
                           train_times: list[float], val_times: list[float],
                           results_dir: str, epoch: int, train_images_processed: int,
                           val_images_processed: int):
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
    
    # Calculate throughput (images per second)
    total_train_time = sum(train_times) / 1000  # Convert to seconds
    total_val_time = sum(val_times) / 1000  # Convert to seconds
    
    train_throughput: float = train_images_processed / total_train_time if total_train_time > 0 else 0
    val_throughput: float = val_images_processed / total_val_time if total_val_time > 0 else 0
    
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left subplot: Loss and accuracy progress
    ax1.plot(cumulative_train_times, train_losses, 'b-', label='Train Loss', marker='o', linewidth=2)
    ax1.plot(cumulative_val_times, val_losses, 'r-', label='Validation Loss', marker='s', linewidth=2)
    ax1.plot(cumulative_train_times, train_accuracies_scaled, 'b--', label='Train Accuracy', marker='^', linewidth=2)
    ax1.plot(cumulative_val_times, val_accuracies_scaled, 'r--', label='Validation Accuracy', marker='d', linewidth=2)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Loss / Accuracy')
    ax1.set_title(f'Training Progress - Epoch {epoch + 1}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)  # Set y-axis limits to [0, 1] since both metrics are in this range
    
    # Right subplot: Throughput bar chart
    throughput_labels = ['Training', 'Validation']
    throughput_values = [train_throughput, val_throughput]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax2.bar(throughput_labels, throughput_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Throughput (images/second)')
    ax2.set_title('Training & Validation Throughput')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, throughput_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(throughput_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(results_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--classifier', required=False,
                        default='SimpleCNN',
                        choices=['SimpleCNN', 'YoloN', 'YoloS', 'YoloM', 'YoloL', 'YoloX'],
                        help='Model type to train: SimpleCNN or YoloN, YoloS, YoloM, YoloL, YoloX')
    return parser.parse_args()


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
          model_type: str = 'SimpleCNN', device: str = 'cuda'):
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
        
        for x_batch, y_batch in track(train_loader, description=f"Training Epoch {epoch+1}", total=len(train_loader)): # iterate over batches
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
            
            # Start timing for next batch data loading
            batch_start_time = time.time_ns() / 1e6
        
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
        
        # Print detailed timing information
        print('Epoch : {}, train loss : {:.4f}, train accuracy: {:.1f}%'.format(epoch+1, epoch_loss, train_accuracy))
        print('  Total time: {:.2f}s'.format(train_time))
        print('  Data loading: {:.2f}s ({:.1f}%)'.format(total_data_loading_time, 100 * total_data_loading_time / train_time))
        print('  GPU transfer: {:.2f}s ({:.1f}%)'.format(total_gpu_transfer_time, 100 * total_gpu_transfer_time / train_time))
        print('  Train step: {:.2f}s ({:.1f}%)'.format(total_train_step_time, 100 * total_train_step_time / train_time))
        print()

        # validation doesnt requires gradient
        with torch.no_grad():
            cumulative_loss = 0
            
            # Record validation start time
            val_start_time = time.time_ns() / 1e6
            
            # Initialize timing accumulators for validation
            val_total_data_loading_time = 0
            val_total_gpu_transfer_time = 0
            val_total_inference_time = 0

            misc_sum = 0
            num_samples = 0
            model.eval()
            
            # Measure validation data loading time
            val_batch_start_time = time.time_ns() / 1e6
            
            for x_batch, y_batch in track(test_loader, description=f"Validation Epoch {epoch+1}", total=len(test_loader)):
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
                
                # Handle different output formats
                if model_type.startswith('Yolo'):
                    # YOLO outputs probabilities directly, ensure they're in the right shape
                    if yhat.dim() == 1:
                        yhat = yhat.unsqueeze(1)
                
                val_loss = loss_fn(yhat,y_batch)
                cumulative_loss += val_loss / len(test_loader)
                val_total_inference_time += (time.time_ns() / 1e6) - val_inference_start_time

                # Apply appropriate activation function based on model type
                ans = yhat
                ans = ans > 0.5
                misc = torch.sum(ans == y_batch)
                misc_sum += misc.item()
                num_samples += len(y_batch)
                # print(f"Accuracy: {misc.item() * 100 / len(y_batch)} %\n")
                
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
                test_inference=val_total_inference_time))

            throughput_per_epoch.append(throughput)

            # Store for plotting
            val_loss_history.append(float(cumulative_loss))
            val_accuracy_history.append(float(val_accuracy))
            val_time_history.append(val_time)
            
            # Update cumulative validation images
            cumulative_val_images += num_samples

            # Print detailed validation timing information
            print('Validation - Loss: {:.4f}, Accuracy: {:.1f}%'.format(cumulative_loss, val_accuracy))
            print('  Total time: {:.2f}s'.format(val_time))
            print('  Data loading: {:.2f}s ({:.1f}%)'.format(val_total_data_loading_time, 100 * val_total_data_loading_time / val_time))
            print('  GPU transfer: {:.2f}s ({:.1f}%)'.format(val_total_gpu_transfer_time, 100 * val_total_gpu_transfer_time / val_time))
            print('  Inference: {:.2f}s ({:.1f}%)'.format(val_total_inference_time, 100 * val_total_inference_time / val_time))
            print()
            
            # Generate plot at the end of each epoch
            if results_dir:
                plot_training_progress(train_loss_history, train_accuracy_history, 
                                     val_loss_history, val_accuracy_history,
                                     train_time_history, val_time_history, 
                                     results_dir, epoch, cumulative_train_images, cumulative_val_images)
            
            # save best model
            if cumulative_loss < best_loss:
                best_model_wts = model.state_dict()
                best_loss = cumulative_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if (early_stopping_counter >= early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
                print("Terminating: early stopping")
                break # terminate training


    print(str(epoch_test_losses) + '\n')
    print(str(epoch_train_losses) + '\n')

    # Calculate total training and validation times
    total_train_time = sum(epoch['time'] for epoch in epoch_train_losses)
    total_val_time = sum(epoch['time'] for epoch in epoch_test_losses)

    print(f'Total validation time: {total_val_time:.2f}s')
    print(f'Total time: {total_train_time + total_val_time:.2f}s')

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
}


def train_classifier(training_path: str, tile_size: int, model_type: str = 'SimpleCNN'):
    print(f'Training {model_type} (tile_size={tile_size})\n')
    
    # Create results directory early so we can save plots during training
    results_dir = os.path.join(training_path, 'results', f'{model_type}_{tile_size}')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    # Instantiate the correct model based on model_type
    if model_type not in MODEL_ZOO:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = MODEL_ZOO[model_type](tile_size).to('cuda')
    loss_fn = torch.nn.BCELoss().to('cuda')
    
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
    best_model_wts = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=100,
                           results_dir=results_dir, model_type=model_type, device='cuda')
    assert best_model_wts is not None

    # Load best model
    model.load_state_dict(best_model_wts)
    with open(os.path.join(results_dir, 'model.pth'), 'wb') as f:
        torch.save(model, f)


def main(args):
    dataset_dir = os.path.join(CACHE_DIR, args.dataset)

    for video in sorted(os.listdir(dataset_dir)):
        video_path = os.path.join(dataset_dir, video)
        if not os.path.isdir(video_path) and not video.endswith('.mp4'):
            continue

        print(f"Processing video {video_path}")
        for tile_size in TILE_SIZES:
            training_path = os.path.join(video_path, 'training') 
            train_classifier(training_path, tile_size, args.classifier)


if __name__ == '__main__':
    main(parse_args())