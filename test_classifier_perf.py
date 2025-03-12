import time
import json

import torch
import torch.utils.data
import torch.optim

from torchvision import datasets, transforms
from torchvision.models.efficientnet import efficientnet_v2_s
from torch.optim import Adam  # type: ignore

from minivan.proxy import ClassifyRelevance

from tqdm import tqdm


device = torch.device('cuda:5')
# fpp = open('./log.txt', 'w')
fpp = open('./log.json', 'w')
fpp.write('[{}')


def train_cnn(width: int, record=True):
    with torch.no_grad():
        # fpp.write(f'Testing Small CNN (width={width})\n')
        model = ClassifyRelevance(width).to(device)
        # model.load_state_dict(torch.load(f'cnn{width}_model.pth', weights_only=True))
        model.eval()
        # model = torch.compile(model)

        _train_data = datasets.ImageFolder('./train-proxy-data', transform=transforms.ToTensor())
        print(len(_train_data))

        generator = torch.Generator().manual_seed(42)
        for s in [2 ** x for x in range(1, 14)]:
            # split = int(s * len(_train_data))
            train_data, _ = torch.utils.data.random_split(
                dataset=_train_data,
                lengths=[s, len(_train_data) - s],
                generator=generator
            )

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
            # print(len(train_data))

            for x_batch, _ in train_loader:
                print(x_batch.shape)
                runtime: dict = {'split': len(x_batch), 'width': width}
                start = time.time()
                x_batch = x_batch.to(device)
                end = time.time()
                print(f"Data loading time: {end - start}")
                if record:
                    fpp.write(',\n' + json.dumps({
                        **runtime,
                        'action': 'load',
                        'runtime': int((end - start) * 1000 * 100)
                    }))

                start = time.time()
                yhat = model(x_batch)
                ans = torch.sigmoid(yhat)
                end = time.time()
                print(f"Model execution time: {end - start}")
                if record:
                    fpp.write(',\n' + json.dumps({
                        **runtime,
                        'action': 'predict',
                        'runtime': int((end - start) * 1000 * 100)
                    }))

                start = time.time()
                np_ans = ans.cpu().detach().numpy()
                print(int(sum(np_ans) / 10000000000))
                end = time.time()
                print(f"Data unloading time: {end - start}")
                runtime['unload'] = end - start
                if record:
                    fpp.write(',\n' + json.dumps({
                        **runtime,
                        'action': 'unload',
                        'runtime': int((end - start) * 1000 * 100)
                    }))


def main():
    train_cnn(32, record=False)
    for _ in range(10):
        train_cnn(32)
        train_cnn(64)
        train_cnn(128)
        train_cnn(256)
        train_cnn(512)
    fpp.write('\n]')


if __name__ == '__main__':
    main()

fpp.close()