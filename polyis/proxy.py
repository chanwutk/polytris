import torch


class ClassifyRelevance(torch.nn.Module):
    def __init__(self, img_size: int, width=128):
        super().__init__()
        features = [3]
        while img_size > 1:
            if img_size % 2 != 0:
                raise ValueError(f"img_size must be 2^x, got {img_size}")
            features.append(width)
            img_size //= 2
        inOutFeatures = zip(features[:-1], features[1:])
        sequential = []
        for inFeatures, outFeatures in inOutFeatures:
            sequential.append(torch.nn.Conv2d(
                in_channels=inFeatures,
                out_channels=outFeatures,
                kernel_size=4,
                stride=2,
                padding=1,
                # padding='same'
            ))
            sequential.append(torch.nn.ReLU())
        self.encoder = torch.nn.Sequential(*sequential)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.Linear(width, 1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.encoder(img)
        x = x.flatten(1)
        x = self.decoder(x)
        # return torch.sigmoid(x)
        return x


def train():
    model = ClassifyRelevance()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

