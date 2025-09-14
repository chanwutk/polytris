from typing import Dict, Any
import torch
import ultralytics.nn.tasks
from ultralytics import YOLO
import ultralytics.nn.modules.head
import ultralytics.nn.tasks

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
        """
        super().__init__()
        
        if model_size not in MODEL_SIZE_MAP:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of {list(MODEL_SIZE_MAP.keys())}")

        yolo_model = YOLO(MODEL_SIZE_MAP[model_size], verbose=False)
        yolo_model.fuse()
        yolo_model.requires_grad_(True)
        classifier = yolo_model.model
        assert isinstance(classifier, ultralytics.nn.tasks.ClassificationModel)

        self.convs = torch.nn.Sequential(*classifier.model[:-1])
        self.classify = classifier.model[-1]
        assert isinstance(self.classify, ultralytics.nn.modules.head.Classify)

        self.classify.linear = torch.nn.Linear(self.classify.linear.in_features, 1)

        self.train()
        
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
        x = img
        x = self.convs(x)
        c = self.classify
        x = c.linear(c.drop(c.pool(c.conv(x)).flatten(1)))
        return torch.sigmoid(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        return {
            'model_size': self.model_size,
            'model_name': f'yolo11{self.model_size}-cls.pt',
            'num_classes': 1,
            'car_class_index': 0
        }
