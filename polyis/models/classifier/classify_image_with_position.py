import torch


class ClassifyImageWithPosition(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, pos_encode_size: int = 16):
        super().__init__()
        self.base_model = base_model
        num_features = base_model.num_features
        assert isinstance(num_features, int)
        num_features = int(num_features)

        self.adapter = torch.nn.Conv2d(in_channels=6, out_channels=3,
                                       kernel_size=1, stride=1, padding=0)
        adapter_weight = torch.zeros_like(self.adapter.weight)
        for i in range(3):
            adapter_weight[i, i, 0, 0] = 1.0
        blend_value = 1.0 / 3.0
        adapter_weight[:, 3:, 0, 0] = blend_value
        self.adapter.weight.data = adapter_weight
        assert self.adapter.bias is not None
        torch.nn.init.zeros_(self.adapter.bias)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features + pos_encode_size, num_features + pos_encode_size),
            torch.nn.BatchNorm1d(num_features + pos_encode_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(num_features + pos_encode_size, 1),
        )

        self.pos_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, pos_encode_size),
            torch.nn.BatchNorm1d(pos_encode_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )

    def forward2(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        x = self.base_model(x)
        pos = self.pos_encoder(pos)
        x = torch.cat((x, pos), dim=1)
        x = self.classifier(x)
        return x

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented")
