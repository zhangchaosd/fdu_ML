import torch
from torch._C import device
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

DATAPATH = 'D:/9709/Desktop/works/fdu_ML/MLpj/天纺标数据/'
BATCHSIZE = 64
EPOCH = 100

class Dataset3(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_dir, train = True):
        self.x_train = np.load(dataset_dir + 'x_train_PE.npy').astype(np.float32)
        self.y_train = np.load(dataset_dir + 'y_train_PE.npy').astype(np.float32)
        self.x_test = np.load(dataset_dir + 'x_test_PE.npy').astype(np.float32)
        self.y_test = np.load(dataset_dir + 'y_test_PE.npy').astype(np.float32)
        self.train = train

    def __len__(self):
        
        if self.train:
            return len(self.x_train)
        return len(self.x_test)

    def __getitem__(self, idx):
        if self.train:
            return torch.from_numpy(self.x_train[idx]), torch.from_numpy(self.y_train[idx])
        return torch.from_numpy(self.x_test[idx]), torch.from_numpy(self.y_test[idx])

#-----------------------------------------------------------
class Net3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features = 1307, out_features = 2560),
            nn.ReLU(),
            nn.Linear(2560,512),
            nn.ReLU(),
            nn.Linear(512,2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        return self.fc(x)
#------------------------------------------------------------

def mission3():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    training_data = Dataset3(DATAPATH, train = True)
    test_data = Dataset3(DATAPATH, train = False)
    train_dataloader = DataLoader(training_data, batch_size = BATCHSIZE, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle = True) #676
    print('mission 3 start')
    model = Net3()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0000001)
    for i in range(EPOCH):
        print('EPOCH: ', i)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if batch % 10 == 0:
            #    print("loss: ", loss)

        # test
        model.eval()
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            MAE = torch.mean(torch.abs(pred[:, 0]-y[:, 0]))
            print('EPOCH: ', i, 'MAE: ', MAE)


if __name__ == '__main__':
    mission3()
    exit()