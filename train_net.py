import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import data_transforms, DatasetImageNet
from net import DeepRank


# -- parameters
BATCH_SIZE = 20
LEARNING_RATE = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -- path info
TRIPLET_PATH = "triplet.csv"
MODEL_PATH = 'deeprank.pt'  # model will save at this path

# -- dataset loader & device setting
train_dataset = DatasetImageNet(TRIPLET_PATH, transform=data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)


def train_model(num_epochs, optim_name=""):
    model = DeepRank()

    if torch.cuda.is_available():
        model.to(device)

    if optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optim_name == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)
    print(f'==> Selected optimizer : {optim_name}')

    model.train()  # set to training mode

    start_time = time.time()
    for epoch in range(num_epochs):

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        running_loss = []
        for batch_idx, (Q, P, N) in enumerate(train_loader):
            if torch.cuda.is_available():
                Q, P, N = Q.to(device), P.to(device), N.to(device)

            # set gradient to 0
            optimizer.zero_grad()

            Q_embedding, P_embedding, N_embedding = model(Q), model(P), model(N)

            # get triplet loss
            loss = F.triplet_margin_loss(anchor=Q_embedding, positive=P_embedding, negative=N_embedding)

            # back-propagate & optimize
            loss.backward()
            optimizer.step()

            # calculate loss
            running_loss.append(loss.item())

            print(f'\t--> epoch{epoch+1} {100 * batch_idx / len(train_loader):.2f}% done... loss : {loss:.4f}')

        epoch_loss = np.mean(running_loss)
        print(f'epoch{epoch+1} average loss: {epoch_loss:.2f}')

    finish_time = time.time()
    print(f'elapsed time : {time.strftime("%H:%M:%S", time.gmtime(finish_time - start_time))}')

    torch.save(model.state_dict(), MODEL_PATH)  # save model parameters


def main():
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--epochs', help='Number of epochs')
    parser.add_argument('--optim', help='Optimizer to choose')

    args = parser.parse_args()

    epochs = 0
    if int(args.epochs) < 0:
        print('This should be a positive value')
        quit()
    else:
        epochs = int(args.epochs)

    optim_type = 'sgd'  # default optimizer
    if str(args.optim) in ["adam", "rms"]:
        optim_type = args.optim.lower()

    train_model(num_epochs=epochs, optim_name=optim_type)


if __name__ == '__main__':
    main()

