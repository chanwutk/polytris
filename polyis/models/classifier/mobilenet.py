import torch
import torchvision


class MobileNetL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = torch.nn.Linear(last_layer.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class MobileNetLQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.quantization.mobilenet_v3_large(weights='DEFAULT', quantize=True)
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = torch.nn.Linear(last_layer.in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class MobileNetS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = torch.nn.Linear(last_layer.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))