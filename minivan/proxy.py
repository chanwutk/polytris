import torch


class ClassifyRelevance(torch.nn.Module):
    def __init__(self, width=128):
        super().__init__()
        features = [3, 32, 64, 64, 64, 64, 64, 64]
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
            torch.nn.Linear(64, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.Linear(width, 1),
        )

        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Conv2d(64, 64, 4, padding='same'),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(64, 1, 4, padding='same'),
        # )

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

