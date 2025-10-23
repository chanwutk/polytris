import torch
import torchvision.models as models


class WideResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT, progress=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return logits instead of sigmoid probabilities


class WideResNet101(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.DEFAULT, progress=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return logits instead of sigmoid probabilities