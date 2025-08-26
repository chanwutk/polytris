#!/usr/local/bin/python

import argparse
import json
import os
import shutil
import time

import torch
import torch.utils.data
import torch.optim
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.optim import Adam

from polyis.models.classifier.simple_cnn import SimpleCNN
from scripts.utilities import CACHE_DIR


TILE_SIZES = [32, 64, 128]


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    return parser.parse_args()


def train_step(model: "torch.nn.Module", loss_fn: "torch.nn.modules.loss._Loss",
               optimizer: "torch.optim.Optimizer", inputs: "torch.Tensor", labels: "torch.Tensor"):
    optimizer.zero_grad()

    outputs: "torch.Tensor" = model(inputs)
    loss: "torch.Tensor" = loss_fn(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def train(model: "torch.nn.Module", loss_fn: "torch.nn.modules.loss._Loss",
          optimizer: "torch.optim.Optimizer", train_loader: "torch.utils.data.DataLoader",
          test_loader: "torch.utils.data.DataLoader", n_epochs: int, device: str = 'cuda'):
    losses = []
    val_losses = []

    early_stopping_tolerance = 3
    early_stopping_threshold = 0.001

    epoch_train_losses: list[dict] = []
    epoch_test_losses: list[dict] = []

    best_model_wts: "dict[str, torch.Tensor] | None" = None
    best_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(n_epochs):
        epoch_loss = 0
        
        # Record training start time
        train_start_time = time.time_ns() / 1e6
        
        model.train()
        for x_batch, y_batch in tqdm(train_loader, total=len(train_loader)): # iterate ove batches
            x_batch = x_batch.to(device) # move to gpu
            y_batch = y_batch.to(device).unsqueeze(1).float() # convert target to same nn output shape

            loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)

            epoch_loss += loss / len(train_loader)
            losses.append(loss)
        
        # Record training end time
        train_end_time = time.time_ns() / 1e6
        train_time = train_end_time - train_start_time
        
        epoch_train_losses.append({
            'op': 'train',
            'loss': float(epoch_loss),
            'time': train_time
        })
        print('\nEpoch : {}, train loss : {}, train time : {:.2f}s\n'.format(epoch+1, epoch_loss, train_time))

        # validation doesnt requires gradient
        with torch.no_grad():
            cumulative_loss = 0
            
            # Record validation start time
            val_start_time = time.time_ns() / 1e6

            misc_sum = 0
            num_samples = 0
            model.eval()
            for x_batch, y_batch in tqdm(test_loader, total=len(test_loader)):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1).float() # convert target to same nn output shape
                # y_batch = y_batch.to(device)

                yhat = model(x_batch)
                val_loss = loss_fn(yhat,y_batch)
                cumulative_loss += val_loss / len(test_loader)

                val_losses.append(val_loss.item())
                
                ans = torch.sigmoid(yhat)
                ans = ans > 0.5
                misc = torch.sum(ans == y_batch)
                misc_sum += misc.item()
                num_samples += len(y_batch)
                # print(f"Accuracy: {misc.item() * 100 / len(y_batch)} %\n")

            # Record validation end time
            val_end_time = time.time_ns() / 1e6
            val_time = val_end_time - val_start_time
            
            epoch_test_losses.append({
                'op': 'test',
                'loss': float(cumulative_loss),
                'time': val_time
            })
            print('Epoch : {}, val loss : {}, val time : {:.2f}s'.format(epoch + 1, cumulative_loss, val_time))  
            print(f"Accuracy: {misc_sum * 100 / num_samples} %\n")
            
            # save best model
            if cumulative_loss < best_loss:
                best_model_wts = model.state_dict()
                best_loss = cumulative_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if (early_stopping_counter >= early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
                print("/nTerminating: early stopping")
                break # terminate training
    
    return best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses


def train_classifier(training_path: str, tile_size: int):
    print(f'Training Small CNN (width={tile_size})\n')
    model = SimpleCNN(tile_size).to('cuda')
    loss_fn = torch.nn.BCEWithLogitsLoss().to('cuda')
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

    # print("Training FC")
    best_model_wts, test_losses, train_losses, losses, val_losses = train(
        model, loss_fn, optimizer, train_loader, test_loader, n_epochs=100, device='cuda')

    assert best_model_wts is not None

    # Load best model
    model.load_state_dict(best_model_wts)

    print(str(test_losses) + '\n')
    print(str(train_losses) + '\n')

    # Calculate total training and validation times
    total_train_time = sum(epoch['time'] for epoch in train_losses)
    total_val_time = sum(epoch['time'] for epoch in test_losses)
    print(f'Total training time: {total_train_time:.2f}s')
    print(f'Total validation time: {total_val_time:.2f}s')
    print(f'Total time: {total_train_time + total_val_time:.2f}s')

    # Create results directory
    results_dir = os.path.join(training_path, 'results', f'SimpleCNN_{tile_size}')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    with open(os.path.join(results_dir, 'model.pth'), 'wb') as f:
        torch.save(model, f)

    with open(os.path.join(results_dir, 'test_losses.json'), 'w') as f:
        f.write(json.dumps(test_losses))

    with open(os.path.join(results_dir, 'train_losses.json'), 'w') as f:
        f.write(json.dumps(train_losses))


def main(args):
    dataset_dir = os.path.join(CACHE_DIR, args.dataset)

    for video in sorted(os.listdir(dataset_dir)):
        video_path = os.path.join(dataset_dir, video)
        if not os.path.isdir(video_path) or video.endswith('.mp4'):
            continue

        print(f"Processing video {video_path}")
        for tile_size in TILE_SIZES:
            training_path = os.path.join(video_path, 'training') 
            train_classifier(training_path, tile_size)


if __name__ == '__main__':
    main(parse_args())