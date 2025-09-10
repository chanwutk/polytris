import torch
import torchvision.models as models


class MobileNetL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, progress=False)
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = torch.nn.Linear(last_layer.in_features, 1,
                                                    bias=last_layer.bias is not None,
                                                    device=last_layer.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


# class MobileNetLQ(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.quantization.mobilenet_v3_large(
#             weights=models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT, quantize=True)
#         last_layer = self.model.classifier[-1]
#         assert isinstance(last_layer, torch.nn.Linear), last_layer
#         self.model.classifier[-1] = torch.nn.QuantizedLinear(last_layer.in_features, 1,
#                                                     bias=last_layer.bias is not None,
#                                                     device=last_layer.weight.device)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.sigmoid(self.model(x))


class MobileNetS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT, progress=False)
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        self.model.classifier[-1] = torch.nn.Linear(last_layer.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))