"""Shared classifier training loop used by p014 and p017 scripts."""

from __future__ import annotations

import copy
import json
import os
import time
import traceback
import typing
import multiprocessing as mp

import altair as alt
import numpy as np
import pandas as pd
import torch

from polyis.utilities import format_time


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


def calculate_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def plot_training_progress(
    train_history: History,
    val_history: History,
    results_dir: str,
    train_images_processed: int,
    val_images_processed: int,
    throughput_per_epoch: list[list[dict[str, float | int | str]]],
    frozen: bool,
):
    try:
        assert len(val_history.time) == len(train_history.time)
        epoch = len(train_history.loss)

        plot_data = []
        for i in range(len(train_history.loss)):
            train_precision, train_recall, train_f1 = calculate_metrics(
                train_history.tp[i], train_history.fp[i], train_history.fn[i]
            )
            val_precision, val_recall, val_f1 = calculate_metrics(
                val_history.tp[i], val_history.fp[i], val_history.fn[i]
            )
            plot_data.extend(
                [
                    {'Epoch': i, 'Value': train_history.loss[i], 'Metric': 'Train Loss', 'Type': 'Loss'},
                    {'Epoch': i, 'Value': val_history.loss[i], 'Metric': 'Validation Loss', 'Type': 'Loss'},
                    {'Epoch': i, 'Value': train_precision, 'Metric': 'Train Precision', 'Type': 'Precision'},
                    {'Epoch': i, 'Value': val_precision, 'Metric': 'Validation Precision', 'Type': 'Precision'},
                    {'Epoch': i, 'Value': train_recall, 'Metric': 'Train Recall', 'Type': 'Recall'},
                    {'Epoch': i, 'Value': val_recall, 'Metric': 'Validation Recall', 'Type': 'Recall'},
                    {'Epoch': i, 'Value': train_f1, 'Metric': 'Train F1', 'Type': 'F1'},
                    {'Epoch': i, 'Value': val_f1, 'Metric': 'Validation F1', 'Type': 'F1'},
                ]
            )
        df = pd.DataFrame(plot_data)

        df['Split'] = df['Metric'].str.split(' ').str[0]
        df['MetricType'] = df['Metric'].str.split(' ', n=1).str[1]

        chart1 = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x='Epoch:Q',
                y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 1])),
                color='MetricType:N',
                strokeDash='Split:N',
            )
            .properties(title=f'Training Progress - Epoch {epoch + 1}', width=400, height=300)
            .resolve_scale(color='independent', strokeDash='independent')
        )

        throughput_flat = []
        for epoch_throughput in throughput_per_epoch[1:]:
            for op_data in epoch_throughput:
                throughput_flat.append(op_data)
        if len(throughput_flat) == 0:
            return

        throughput_df = pd.DataFrame(throughput_flat)
        throughput_df['op'] = throughput_df['op'].astype(str)
        throughput_df['time'] = throughput_df['time'].astype(float)

        op_times_df = throughput_df.groupby('op')['time'].sum().reset_index()

        op_times_df['Phase'] = np.where(op_times_df['op'].str.startswith('train'), 'Training', 'Validation')

        op_times_df = op_times_df[~op_times_df['op'].str.endswith('load_data')]

        op_times_df['ms_per_frame'] = op_times_df.apply(
            lambda row: row['time'] / train_images_processed
            if row['Phase'] == 'Training'
            else row['time'] / val_images_processed,
            axis=1,
        )

        assert isinstance(op_times_df, pd.DataFrame)
        op_times_df['Operation'] = op_times_df['op'].str.replace(r'^(train_|test_)', '', regex=True)

        bar_df = op_times_df[['Phase', 'Operation', 'ms_per_frame']]
        assert isinstance(bar_df, pd.DataFrame)

        assert not bar_df.empty
        chart2 = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X('Phase:N', axis=alt.Axis(labelAngle=0)),
                y='ms_per_frame:Q',
                color='Operation:N',
                tooltip=['Phase', 'Operation', alt.Tooltip('ms_per_frame:Q', format='.2f')],
            )
            .properties(title='Milliseconds per Frame by Operation', width=150, height=300)
            .resolve_scale(color='independent')
        )

        combined_chart = alt.hconcat(chart1, chart2, spacing=20)

        plot_path = os.path.join(results_dir, f'training_progress_{"frozen" if frozen else "finetuned"}.png')
        combined_chart.save(plot_path, scale_factor=2)
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error plotting training progress: {e}")
        print("\n" * 20)
        return


def normalize_output_shape(outputs: torch.Tensor, model_type: str) -> torch.Tensor:
    if model_type.startswith('Yolo'):
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
    return outputs


def calculate_confusion_matrix(
    predictions: torch.Tensor, labels: torch.Tensor
) -> tuple[int, int, int, int]:
    tp = torch.sum((predictions == 1) & (labels == 1)).item()
    tn = torch.sum((predictions == 0) & (labels == 0)).item()
    fp = torch.sum((predictions == 1) & (labels == 0)).item()
    fn = torch.sum((predictions == 0) & (labels == 1)).item()
    return int(tp), int(tn), int(fp), int(fn)


def _forward(model: torch.nn.Module, inputs: torch.Tensor, pos_inputs: torch.Tensor, pos_in_batch: bool) -> torch.Tensor:
    if pos_in_batch:
        return model(inputs, pos_inputs)
    return model(inputs)


def train_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    pos_inputs: torch.Tensor,
    labels: torch.Tensor,
    model_type: str,
    pos_in_batch: bool,
    max_grad_norm: float = 10.0,
):
    optimizer.zero_grad()

    outputs: torch.Tensor = _forward(model, inputs, pos_inputs, pos_in_batch)
    outputs = normalize_output_shape(outputs, model_type)

    loss: torch.Tensor = loss_fn(outputs, labels)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    optimizer.step()

    return loss.item(), outputs


def run_training_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: str,
    model_type: str,
    command_queue: mp.Queue,
    description: str,
    pos_in_batch: bool,
    ema_model: torch.nn.Module | None = None,
    ema_decay: float = 0.999,
):
    data_loading_time: float = 0.0
    gpu_transfer_time: float = 0.0
    step_time: float = 0.0

    cumulative_loss = 0.0
    num_samples: int = 0
    train_tp: int = 0
    train_tn: int = 0
    train_fp: int = 0
    train_fn: int = 0

    model.train()
    batch_start_time = time.time_ns() / 1e6

    command_queue.put((device, {'description': description.format('T  0%')}))

    for idx, (x_batch, y_batch, pos_batch) in enumerate(train_loader):
        data_loading_time += (time.time_ns() / 1e6) - batch_start_time

        gpu_transfer_start_time = time.time_ns() / 1e6
        x_batch = x_batch.to(device)
        if pos_in_batch:
            pos_batch = pos_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1).float()
        gpu_transfer_time += (time.time_ns() / 1e6) - gpu_transfer_start_time

        step_start_time = time.time_ns() / 1e6
        loss, y_hat = train_step(
            model, loss_fn, optimizer, x_batch, pos_batch, y_batch, model_type, pos_in_batch
        )
        step_time += (time.time_ns() / 1e6) - step_start_time

        if ema_model is not None:
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        cumulative_loss += loss

        with torch.no_grad():
            y_hat = normalize_output_shape(y_hat, model_type)
            y_hat_probs = torch.sigmoid(y_hat)
            predictions = y_hat_probs > 0.5
            num_samples += len(y_batch)

            batch_tp, batch_tn, batch_fp, batch_fn = calculate_confusion_matrix(predictions, y_batch)

            train_tp += batch_tp
            train_tn += batch_tn
            train_fp += batch_fp
            train_fn += batch_fn

        desc = f'T {int(idx * 100 / len(train_loader)):>2}%'
        command_queue.put((device, {'description': description.format(desc)}))
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


def run_validation_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.modules.loss._Loss,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    model_type: str,
    command_queue: mp.Queue,
    description: str,
    pos_in_batch: bool,
):
    data_loading_time: float = 0.0
    gpu_transfer_time: float = 0.0
    inference_time: float = 0.0
    loss_time: float = 0.0

    cumulative_loss = 0.0
    num_samples = 0
    val_tp: int = 0
    val_tn: int = 0
    val_fp: int = 0
    val_fn: int = 0

    model.eval()
    val_batch_start_time = time.time_ns() / 1e6

    command_queue.put((device, {'description': description.format('V 00%')}))

    for idx, (x_batch, y_batch, pos_batch) in enumerate(test_loader):
        data_loading_time += (time.time_ns() / 1e6) - val_batch_start_time

        val_gpu_transfer_start_time = time.time_ns() / 1e6
        x_batch = x_batch.to(device)
        if pos_in_batch:
            pos_batch = pos_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1).float()
        gpu_transfer_time += (time.time_ns() / 1e6) - val_gpu_transfer_start_time

        val_inference_start_time = time.time_ns() / 1e6
        yhat = _forward(model, x_batch, pos_batch, pos_in_batch)
        inference_time += (time.time_ns() / 1e6) - val_inference_start_time

        yhat = normalize_output_shape(yhat, model_type)

        val_loss_start_time = time.time_ns() / 1e6
        val_loss = loss_fn(yhat, y_batch)
        cumulative_loss += val_loss
        loss_time += (time.time_ns() / 1e6) - val_loss_start_time

        yhat_probs = torch.sigmoid(yhat)
        ans = yhat_probs > 0.5
        num_samples += len(y_batch)

        batch_tp, batch_tn, batch_fp, batch_fn = calculate_confusion_matrix(ans, y_batch)

        val_tp += batch_tp
        val_tn += batch_tn
        val_fp += batch_fp
        val_fn += batch_fn

        desc = f'V {int(idx * 100 / len(test_loader)):>2}%'
        command_queue.put((device, {'description': description.format(desc)}))
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


def check_early_stopping(
    val_loss: float,
    best_loss: float,
    early_stopping_counter: int,
    early_stopping_tolerance: int,
    early_stopping_threshold: float,
) -> tuple[bool, int]:
    if val_loss < best_loss:
        return False, 0

    early_stopping_counter += 1

    if (early_stopping_counter >= early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
        return True, early_stopping_counter

    return False, early_stopping_counter


def save_training_results(
    results_dir: str,
    epoch_train_losses: list[dict],
    epoch_test_losses: list[dict],
    throughput_per_epoch: list[list[dict[str, float | int | str]]],
):
    with open(os.path.join(results_dir, 'test_losses.json'), 'w') as f:
        f.write(json.dumps(epoch_test_losses))

    with open(os.path.join(results_dir, 'train_losses.json'), 'w') as f:
        f.write(json.dumps(epoch_train_losses))

    with open(os.path.join(results_dir, 'throughput_per_epoch.jsonl'), 'w') as f:
        for t in throughput_per_epoch:
            f.write(json.dumps(t) + '\n')


def format_throughput(throughput: dict[str, float | int | str], prefix: str):
    return format_time(
        **{
            f'{prefix}_{k[:-len("_time")]}': float(v)
            for k, v in throughput.items()
            if k.endswith('_time')
        }
    )


def save_model(
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    results_dir: str,
    name: str | None = None,
):
    if ema_model is not None:
        best_model_wts = ema_model.state_dict()
        best_raw_model_wts = model.state_dict()
        ema_model.zero_grad(set_to_none=True)
        with open(os.path.join(results_dir, name or 'model_best.pth'), 'wb') as f:
            torch.save(ema_model, f)
    else:
        best_model_wts = model.state_dict()
        best_raw_model_wts = model.state_dict()
        model.zero_grad(set_to_none=True)
        with open(os.path.join(results_dir, name or 'model_best.pth'), 'wb') as f:
            torch.save(model, f)

    return best_model_wts, best_raw_model_wts


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    results_dir: str,
    model_type: str,
    device: str,
    command_queue: mp.Queue,
    dataset: str,
    tile_size: int,
    pos_in_batch: bool,
    max_model_name_length: int,
    visualize: bool = False,
    frozen: bool = False,
    initial_best_loss: float = float('inf'),
):
    early_stopping_tolerance = 3
    early_stopping_threshold = 0.001

    epoch_train_losses: list[dict] = []
    epoch_test_losses: list[dict] = []

    train_history: History = History([], [], [], [], [], [])
    val_history: History = History([], [], [], [], [], [])

    cumulative_train_images: int = 0
    cumulative_val_images: int = 0

    best_model_wts: dict[str, torch.Tensor] = model.state_dict()
    best_raw_model_wts: dict[str, torch.Tensor] = model.state_dict()
    best_loss: float = initial_best_loss
    early_stopping_counter: int = 0

    throughput_per_epoch: list[list[dict[str, float | int | str]]] = []

    ema_model = None
    ema_decay = 0.999
    ema_warmup_epochs = 3

    description = f"{dataset} {tile_size:>3} {model_type:>{max_model_name_length}} {test_loader.batch_size:>4} {{}}"
    command_queue.put((device, {'description': description.format('T'), 'total': n_epochs, 'completed': 0}))

    for epoch in range(n_epochs):
        if epoch == ema_warmup_epochs and ema_model is None:
            ema_model = copy.deepcopy(model)
            ema_model.eval()

        train_result = run_training_epoch(
            model,
            loss_fn,
            optimizer,
            train_loader,
            device,
            model_type,
            command_queue,
            description,
            pos_in_batch,
            ema_model=ema_model,
            ema_decay=ema_decay,
        )

        epoch_train_losses.append(
            {
                'op': 'train',
                'loss': train_result['cumulative_loss'],
                'tp': train_result['tp'],
                'tn': train_result['tn'],
                'fp': train_result['fp'],
                'fn': train_result['fn'],
            }
        )

        throughput: list[dict[str, float | int | str]] = []
        throughput.extend(format_throughput(train_result, 'train'))

        cumulative_train_images += train_result['num_samples']

        validation_model = ema_model if ema_model is not None else model
        with torch.no_grad():
            val_result = run_validation_epoch(
                validation_model,
                loss_fn,
                test_loader,
                device,
                model_type,
                command_queue,
                description,
                pos_in_batch,
            )

            epoch_test_losses.append(
                {
                    'op': 'test',
                    'loss': val_result['cumulative_loss'],
                    'tp': val_result['tp'],
                    'tn': val_result['tn'],
                    'fp': val_result['fp'],
                    'fn': val_result['fn'],
                }
            )

            throughput.extend(format_throughput(val_result, 'val'))

            throughput_per_epoch.append(throughput)

            cumulative_val_images += val_result['num_samples']

            train_history.add(train_result)
            val_history.add(val_result)

            if results_dir and visualize:
                plot_training_progress(
                    train_history,
                    val_history,
                    results_dir,
                    cumulative_train_images,
                    cumulative_val_images,
                    throughput_per_epoch,
                    frozen,
                )

            if val_result['cumulative_loss'] < best_loss:
                best_model_wts, best_raw_model_wts = save_model(model, ema_model, results_dir)
                best_loss = val_result['cumulative_loss']
                early_stopping_counter = 0
            else:
                should_stop, early_stopping_counter = check_early_stopping(
                    val_result['cumulative_loss'],
                    best_loss,
                    early_stopping_counter,
                    early_stopping_tolerance,
                    early_stopping_threshold,
                )

                if should_stop:
                    break

        command_queue.put((device, {'completed': epoch + 1}))

    save_training_results(results_dir, epoch_train_losses, epoch_test_losses, throughput_per_epoch)

    assert best_model_wts is not None, "Best model weights are None"
    return best_model_wts, best_raw_model_wts, best_loss
