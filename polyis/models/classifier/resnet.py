import torch
import torchvision.models as models


class ResNet152(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT, progress=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class ResNet101(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, progress=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class ResNet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, progress=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


# class ResNet18Q(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.quantization.resnet18(
#             weights=models.quantization.ResNet18_QuantizedWeights.DEFAULT, quantize=True)
#         self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.sigmoid(self.model(x))