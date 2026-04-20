import torch
import torchvision.models as models

from polyis.models.classifier.utils import collapse_classifier


class ShuffleNet05Baseline(torch.nn.Module):
    """ShuffleNet v2 x0.5 with pretrained FC folded to a single binary logit."""

    def __init__(self):
        super().__init__()
        weight = models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        self.model = models.shufflenet_v2_x0_5(weights=weight, progress=False)
        last_fc = self.model.fc
        assert isinstance(last_fc, torch.nn.Linear)
        self.model.fc = collapse_classifier(weight, last_fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_base_model(self, _keep_first_layer_trainable: bool = False):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze_base_model(self):
        for param in self.model.parameters():
            param.requires_grad = True
