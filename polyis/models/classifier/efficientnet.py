import torch
import torchvision


class EfficientNetS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = torch.nn.Linear(last_layer.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class EfficientNetL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.efficientnet_v2_l(weights='DEFAULT')
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = torch.nn.Linear(last_layer.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))