import torch
import torchvision


class ResNet152(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet152(weights='DEFAULT')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class ResNet101(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(weights='DEFAULT')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class ResNet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(weights='DEFAULT')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class ResNet18Q(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.quantization.resnet18(weights='DEFAULT', quantize=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))