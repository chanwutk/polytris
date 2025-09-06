from typing import Dict, Any
import torch
import ultralytics.nn.tasks
from ultralytics import YOLO

# Map model sizes to actual model names
MODEL_SIZE_MAP = {
    'n': 'yolo11n-cls.pt',
    's': 'yolo11s-cls.pt', 
    'm': 'yolo11m-cls.pt',
    'l': 'yolo11l-cls.pt',
    'x': 'yolo11x-cls.pt'
}


def YoloN(_tile_size: int):
    return YoloClassifier(model_size='n')

def YoloS(_tile_size: int):
    return YoloClassifier(model_size='s')

def YoloM(_tile_size: int):
    return YoloClassifier(model_size='m')

def YoloL(_tile_size: int):
    return YoloClassifier(model_size='l')

def YoloX(_tile_size: int):
    return YoloClassifier(model_size='x')


class YoloClassifier(torch.nn.Module):
    def __init__(self, model_size: str = "s"):
        """
        YOLOv11-based classifier for car detection.
        
        Args:
            model_size: Size of the YOLOv11 model ('n', 's', 'm', 'l', 'x')
            num_classes: Number of output classes (default: 1000 for ImageNet)
        """
        super().__init__()
        
        if model_size not in MODEL_SIZE_MAP:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of {list(MODEL_SIZE_MAP.keys())}")
        
        # Load the YOLOv11 classification model
        model_name = MODEL_SIZE_MAP[model_size]
        model = YOLO(model_name).model
        assert isinstance(model, ultralytics.nn.tasks.ClassificationModel)
        classifier: "ultralytics.nn.tasks.ClassificationModel" = model
        self.yolo_model = classifier.model
        self.yolo_model[-1].export = True
        self.classifiction_layer = torch.nn.Linear(model.model[-1].linear.out_features, 1)
        
        # Store model configuration
        self.model_size = model_size
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the YOLOv11 classifier.
        
        Args:
            img: Input image tensor of shape (batch_size, 3, height, width) with dtype uint8 and values in [0, 255]
            
        Returns:
            Car probabilities as a tensor of shape (batch_size,)
        """
        # Convert uint8 [0, 255] to float32 [0, 1] for YOLO input
        img_float = img.float() / 255.0
        
        # Run batch inference with YOLOv11 - pass tensor directly
        results = self.yolo_model(img_float)
        results = self.classifiction_layer(results)
        return torch.sigmoid(results)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        return {
            'model_size': self.model_size,
            'model_name': f'yolo11{self.model_size}-cls.pt',
            'num_classes': 1,
            'car_class_index': 0
        }
