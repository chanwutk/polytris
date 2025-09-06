import torch
import torchvision


class WideResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.wide_resnet50_2(weights='DEFAULT')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class WideResNet101(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.wide_resnet101_2(weights='DEFAULT')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))