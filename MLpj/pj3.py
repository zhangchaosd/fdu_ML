import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
import random

DATAPATH = 'D:/9709/Desktop/works/fdu_ML/MLpj/天纺标数据/'
#DATAPATH = 'C:/Users/97090/Desktop/fdu_ML/MLpj/天纺标数据/'
BATCHSIZE = 256
EPOCH = 250

class AddNoise(object):
    def __init__(self, p = 0.2, alpha = 0.999):
        self.p = p
        self.alpha = alpha
 
    def __call__(self, x):
        if random.uniform(0, 1) < self.p:
            rd = torch.randn_like(x) - 0.5
            rd[rd < 0] = self.alpha - 1
            rd[rd > 0] = 1 - self.alpha
            rd = rd * x
            x += rd
            return x
        else:
            return x

class Dataset3(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_dir, train = True, transform = None):
        self.x_train = torch.from_numpy(np.load(dataset_dir + 'x_train_PE.npy').astype(np.float32))
        self.y_train = torch.from_numpy(np.load(dataset_dir + 'y_train_PE.npy').astype(np.float32))
        self.x_test = torch.from_numpy(np.load(dataset_dir + 'x_test_PE.npy').astype(np.float32))
        self.y_test = torch.from_numpy(np.load(dataset_dir + 'y_test_PE.npy').astype(np.float32))
        self.train = train
        self.transform = transform

    def __len__(self):
        if self.train:
            return len(self.x_train)
        return len(self.x_test)

    def __getitem__(self, idx):
        if self.train:
            if self.transform:
                return self.transform(self.x_train[idx]), self.y_train[idx]
            return self.x_train[idx], self.y_train[idx]
        return self.x_test[idx], self.y_test[idx]

#-----------------------------------------------------------
class Net32(nn.Module):
    def __init__(self, in_feature = 1307, hidden_dim = 2048) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_feature, hidden_dim),
            nn.LeakyReLU(0.01)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.LeakyReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU()
        )
        self.fc7 = nn.Sequential(
            nn.Linear(128, 2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2+out1)
        out4 = self.fc4(out3+out2+out1)
        out5 = self.fc5(out4)
        out6 = self.fc6(out5)
        return self.fc7(out6)

def mission3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    training_data = Dataset3(DATAPATH, train = True, transform = AddNoise())
    test_data = Dataset3(DATAPATH, train = False)
    train_dataloader = DataLoader(training_data, batch_size = BATCHSIZE, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle = True) #676
    print('mission 3 start')
    model = Net32().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.000003)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    lambda1 = lambda epoch:np.sin(epoch) / epoch
    scheduler = StepLR(optimizer, step_size = 5, gamma = 0.8)
    f = 0.1
    for i in range(EPOCH):
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #scheduler.step()
        
        # test
        model.eval()
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            MAE = torch.mean(torch.abs(pred[:,0] - y[:,0])).data.cpu().numpy()
            print('EPOCH: ', i, 'test MAE: ', MAE )
            if MAE < f:
                f *= 0.8
                nam = 'model_weights MAE ' + str(MAE) + '.pth'
                torch.save(model.state_dict(), nam)
                print('update lr, saved model')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1


if __name__ == '__main__':
    mission3()
    exit()