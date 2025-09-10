import torch
import torchvision.models as models


class EfficientNetS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.EfficientNet_V2_S_Weights.DEFAULT
        categories = weight.meta["categories"]
        relevant_categories = [i for i, c in enumerate(categories) if c in ['car', 'bus', 'truck']]
        irrelevant_categories = [i for i in range(len(categories)) if i not in relevant_categories]
        self.model = models.efficientnet_v2_s(weights=weight, progress=False)
        last_layer = self.model.classifier[-1]

        r_weight = last_layer.weight[relevant_categories].sum(dim=1, keepdim=True)
        i_weight = last_layer.weight[irrelevant_categories].sum(dim=1, keepdim=True)
        r_bias = last_layer.bias[relevant_categories].sum(keepdim=True)
        i_bias = last_layer.bias[irrelevant_categories].sum(keepdim=True)

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