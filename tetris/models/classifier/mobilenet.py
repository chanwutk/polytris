import torch
import torchvision.models as models


class MobileNetL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.MobileNet_V3_Large_Weights.DEFAULT
        self.model = models.mobilenet_v3_large(weights=weight, progress=False)
        self.num_features = self.model.classifier[-1].in_features
        self.model.classifier = torch.nn.Sequential(*self.model.classifier[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return features instead of classification


class MobileNetS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weight, progress=False)
        self.num_features = self.model.classifier[-1].in_features
        self.model.classifier = torch.nn.Sequential(*self.model.classifier[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return features instead of classification
