import torch
import torchvision.models as models

from polyis.models.classifier.utils import collapse_classifier


class SwinV2T(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = models.Swin_V2_T_Weights.DEFAULT
        self.model = models.swin_v2_t(weights=weight, progress=False)
        self.model.fc = collapse_classifier(weight, self.model.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Return logits instead of sigmoid probabilities
