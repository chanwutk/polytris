import torch
import torchvision.models as models


# class ShuffleNet05Q(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.quantization.shufflenet_v2_x0_5(
#             weights=models.quantization.ShuffleNet_V2_X0_5_QuantizedWeights.DEFAULT, quantize=True)
#         self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.sigmoid(self.model(x))


class ShuffleNet05(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT, progress=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class ShuffleNet20(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT, progress=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))