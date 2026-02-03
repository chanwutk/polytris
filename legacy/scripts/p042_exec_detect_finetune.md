# p042_exec_detect_finetune

Train object detection models on fine-tuning datasets created by `p041_exec_detect_finetune_dataset.py`.

## Overview

This script trains detection models on compressed/packed frame datasets to improve detection accuracy on polyomino-collaged images. It supports multiple training frameworks.

## Supported Models

| Model | Framework | Description |
|-------|-----------|-------------|
| `5x6` | Ultralytics | YOLOv5x6 - Large YOLO model optimized for high resolution |
| `11x` | Ultralytics | YOLO11x - Latest YOLO architecture |
| `detectron2` | Detectron2 | Faster R-CNN R50-FPN from Facebook AI Research |
| `darknet` | Darknet | YOLOv3 - Original YOLO implementation |

## Command Line Arguments

```
--dataset         Dataset name (required)
--tilesize        Tile size (required)
--tilepadding     Tile padding mode (required)
--model           Model type: '5x6', '11x', 'detectron2', 'darknet' (default: '5x6')
--epochs          Training epochs (default: 100)
--imgsz           Image size for training (default: 1280)
--batch           Batch size (-1 for auto, default: -1)
--device          Device to use (e.g., "0" or "0,1,2")
--data-dir        Override data directory path
--darknet-cfg     Path to Darknet config file (.cfg) for YOLOv3 training
--darknet-weights Path to pretrained Darknet weights for transfer learning
```

## Usage Examples

### Train with Ultralytics YOLO
```bash
# YOLOv5x6
./run scripts/p042_exec_detect_finetune --dataset caldot1 --tilesize 60 --tilepadding none --model 5x6 --epochs 100

# YOLO11x
./run scripts/p042_exec_detect_finetune --dataset caldot1 --tilesize 60 --tilepadding none --model 11x --epochs 100
```

### Train with Detectron2
```bash
./run scripts/p042_exec_detect_finetune --dataset caldot1 --tilesize 60 --tilepadding none --model detectron2 --epochs 50
```

### Train with Darknet YOLOv3
```bash
# Default config
./run scripts/p042_exec_detect_finetune --dataset caldot1 --tilesize 60 --tilepadding none --model darknet

# Custom config and pretrained weights
./run scripts/p042_exec_detect_finetune --dataset caldot1 --tilesize 60 --tilepadding none --model darknet \
    --darknet-cfg /path/to/yolov3-custom.cfg \
    --darknet-weights /path/to/darknet53.conv.74
```

## Data Directory Structure

The script expects datasets created by `p041_exec_detect_finetune_dataset.py`:

```
/polyis-data/training/finetune/{dataset}_{tilesize}_{tilepadding}/
├── intermediate.jsonl          # Intermediate dataset
├── ultralytics/                # For --model 5x6 or 11x
│   ├── data.yaml
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── coco/                       # For --model detectron2
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
├── darknet/                    # For --model darknet
│   ├── images/
│   ├── labels/
│   ├── train.txt
│   ├── val.txt
│   ├── obj.data
│   ├── obj.names
│   └── backup/
└── weights/                    # Training outputs
    ├── yolov5x6/
    ├── yolo11x/
    ├── detectron2/
    └── darknet/
```

## Output

Trained model weights are saved to:
- **Ultralytics**: `{data_dir}/weights/yolov{model}/`
- **Detectron2**: `{data_dir}/weights/detectron2/`
- **Darknet**: `{data_dir}/weights/darknet/`

## Prerequisites

1. Run `p041_exec_detect_finetune_dataset.py` first to create the dataset
2. Ensure the appropriate format is created:
   - `ultralytics` format for `--model 5x6` or `--model 11x`
   - `coco` format for `--model detectron2`
   - `darknet` format for `--model darknet`

## Implementation Details

### Ultralytics Training
- Uses pretrained weights (`yolov5x6.pt` or `yolo11x.pt`)
- Configurable image size, batch size, and epochs
- Saves training plots and metrics

### Detectron2 Training
- Uses Faster R-CNN R50-FPN pretrained on COCO
- Registers custom COCO-format datasets
- Single class (car) detection

### Darknet Training
- Runs Darknet binary with `detector train` command
- Supports transfer learning from `darknet53.conv.74`
- Saves periodic weight checkpoints and best weights
