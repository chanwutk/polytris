import time

import torch
import torch.utils.data
import torch.optim

from torchvision import datasets, transforms
from torchvision.models.efficientnet import efficientnet_v2_s
from torch.optim import Adam  # type: ignore

from polyis.proxy import ClassifyRelevance

from tqdm import tqdm


device = torch.device('cuda:5')
fpp = open('./log.txt', 'w')


def custom_efficientnet():
    model = efficientnet_v2_s(pretrained=True).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=1, bias=True).to(device)

    return model


def train_step(model, loss_fn, optimizer, inputs, labels):
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs):
    losses = []
    val_losses = []

    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()
        for x_batch, y_batch in tqdm(train_loader, total=len(train_loader)): #iterate ove batches
            x_batch = x_batch.to(device) #move to gpu
            y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
            y_batch = y_batch.to(device) #move to gpu

            loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)

            epoch_loss += loss / len(train_loader)
            losses.append(loss)
        
        epoch_train_losses.append(epoch_loss)
        fpp.write('\nEpoch : {}, train loss : {}\n'.format(epoch+1,epoch_loss))

        #validation doesnt requires gradient
        with torch.no_grad():
            model.eval()
            cumulative_loss = 0
            for x_batch, y_batch in test_loader:
                # print(y_batch)
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
                y_batch = y_batch.to(device)

                #model to eval mode
                model.eval()

                yhat = model(x_batch)
                val_loss = loss_fn(yhat,y_batch)
                cumulative_loss += loss / len(test_loader)

                val_losses.append(val_loss.item())
                
                ans = torch.sigmoid(yhat)
                # ans = yhat
                ans = ans > 0.5
                misc = torch.sum(ans == y_batch)
                fpp.write(f"Accuracy: {misc.item() * 100 / len(y_batch)} %\n")

            epoch_test_losses.append(cumulative_loss)
            fpp.write('Epoch : {}, val loss : {}\n'.format(epoch+1,cumulative_loss))  
            fpp.flush()
            
            best_loss = min(epoch_test_losses)
            
            #save best model
            if cumulative_loss <= best_loss:
                best_model_wts = model.state_dict()
            
            # #early stopping
            # early_stopping_counter = 0
            # if cum_loss > best_loss:
            #   early_stopping_counter +=1

            # if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
            #   print("/nTerminating: early stopping")
            #   break #terminate training
    
    return best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses


def train_cnn(width: int):
    start = time.time()

    try:
        fpp.write(f'Training Small CNN (width={width})\n')
        model = ClassifyRelevance(width).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        running_loss = 0.0


        train_data = datasets.ImageFolder('./train-proxy-data', transform=transforms.ToTensor())

        generator = torch.Generator().manual_seed(42)
        split = int(0.8 * len(train_data))
        train_data, test_data = torch.utils.data.random_split(
            dataset=train_data,
            lengths=[split, len(train_data) - split],
            generator=generator
        )

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=True)

        losses = []
        val_losses = []

        epoch_train_losses = []
        epoch_test_losses = []

        n_epochs = 10
        early_stopping_tolerance = 3
        early_stopping_threshold = 0.03

        # print("Training FC")
        best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=20)
        model.load_state_dict(best_model_wts)

        #load best model
        model.load_state_dict(best_model_wts)

        fpp.write(str(epoch_test_losses) + '\n')
        fpp.write(str(epoch_train_losses) + '\n')

        import json

        with open(f'cnn{width}_model.pth', 'wb') as f:
            torch.save(model, f)

        with open(f'cnn{width}_epoch_test_losses.json', 'w') as f:
            f.write(json.dumps(epoch_test_losses))

        with open(f'cnn{width}_epoch_train_losses.json', 'w') as f:
            f.write(json.dumps(epoch_train_losses))
    finally:
        fpp.write(f'Time cnn{width}:{time.time() - start}')


def main():
    train_cnn(32)
    train_cnn(64)
    train_cnn(128)
    train_cnn(256)
    train_cnn(512)

    start = time.time()

    try:
        fpp.write('Training Small EfficientNet\n')
        # model = ClassifyRelevance().to(device)
        model = custom_efficientnet()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        running_loss = 0.0


        train_data = datasets.ImageFolder('./train-proxy-data', transform=transforms.ToTensor())

        generator = torch.Generator().manual_seed(42)
        split = int(0.8 * len(train_data))
        train_data, test_data = torch.utils.data.random_split(
            dataset=train_data,
            lengths=[split, len(train_data) - split],
            generator=generator
        )

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=True)

        losses = []
        val_losses = []

        epoch_train_losses = []
        epoch_test_losses = []

        n_epochs = 10
        early_stopping_tolerance = 3
        early_stopping_threshold = 0.03

        fpp.write("Training FC\n")
        optimizer = Adam(model.classifier[-1].parameters(), lr=0.001)
        best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=20)
        model.load_state_dict(best_model_wts)

        fpp.write("Tuning all layers\n")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = Adam(model.parameters(), lr=0.001)
        best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=30)
        model.load_state_dict(best_model_wts)

        #load best model
        model.load_state_dict(best_model_wts)

        fpp.write(str(epoch_test_losses) + '\n')
        fpp.write(str(epoch_train_losses) + '\n')

        import json

        with open('eff_model.pth', 'wb') as f:
            torch.save(model, f)

        with open('eff_epoch_test_losses.json', 'w') as f:
            f.write(json.dumps(epoch_test_losses))

        with open('eff_epoch_train_losses.json', 'w') as f:
            f.write(json.dumps(epoch_train_losses))
    finally:
        fpp.write(f'Time eff:{time.time() - start}')


if __name__ == '__main__':
    main()

fpp.close()