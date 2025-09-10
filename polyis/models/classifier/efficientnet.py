import torch
import torchvision.models as models


class EfficientNetS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT, progress=False)
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = torch.nn.Linear(last_layer.in_features, 1,
                                                    bias=last_layer.bias is not None,
                                                    device=last_layer.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class EfficientNetL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT, progress=False)
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = torch.nn.Linear(last_layer.in_features, 1,
                                                    bias=last_layer.bias is not None,
                                                    device=last_layer.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))