import torch
import torchvision.models as models

from polyis.models.classifier.utils import collapse_classifier


class EfficientNetS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.EfficientNet_V2_S_Weights.DEFAULT
        self.model = models.efficientnet_v2_s(weights=weight, progress=False)
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = collapse_classifier(weight, last_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class EfficientNetL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.EfficientNet_V2_L_Weights.DEFAULT
        self.model = models.efficientnet_v2_l(weights=weight, progress=False)
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = collapse_classifier(weight, last_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))