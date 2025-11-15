import torch
import torchvision.models as models

from polyis.models.classifier.utils import collapse_classifier


class ShuffleNet05(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        self.model = models.shufflenet_v2_x0_5(weights=weight, progress=False)
        self.num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return features instead of classification


class ShuffleNet20(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.ShuffleNet_V2_X2_0_Weights.DEFAULT
        self.model = models.shufflenet_v2_x2_0(weights=weight, progress=False)
        self.num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return features instead of classification