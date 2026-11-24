import torch
import torchvision.models as models

from polyis.models.classifier.utils import collapse_classifier


class ResNet152(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.ResNet152_Weights.DEFAULT
        self.model = models.resnet152(weights=weight, progress=False)
        self.model.fc = collapse_classifier(weight, self.model.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return features instead of classification


class ResNet101(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.ResNet101_Weights.DEFAULT
        self.model = models.resnet101(weights=weight, progress=False)
        self.model.fc = collapse_classifier(weight, self.model.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return features instead of classification


class ResNet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weight, progress=False)
        self.model.fc = collapse_classifier(weight, self.model.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return features instead of classification
