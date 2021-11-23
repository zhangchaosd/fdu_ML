import torch
import torch.nn as nn
import json
import numpy as np
from torch.nn.modules import normalization
from torch.nn.modules.activation import Softmax
from torch.utils.data import DataLoader

import os

#DATAPATH = 'D:/9709/Desktop/works/fdu_ML/MLpj/布眼数据集.json'
DATAPATH = 'C:/Users/97090/Desktop/fdu_ML/MLpj/布眼数据集fixed.json'
DATAPATH0 = 'C:/Users/97090/Desktop/fdu_ML/MLpj/一条数据.json'
# 布眼数据集fixed.json: https://drive.google.com/file/d/1SBX_lIny3KfxX2JOaePPnQWNBavSvGs5/view?usp=sharing
BATCHSIZE = 512
EPOCH = 500

class Dataset2(torch.utils.data.dataset.Dataset):
    def __init__(self, data, train = True, val = False):
        self.testData = data[:10000]
        self.valData = data[10000:20000]
        self.trainData = data[20000:]
        self.train = train
        self.val = val

    def __len__(self):
        if self.train:
            return len(self.trainData) # 50026
        if self.val:
            return len(self.valData) # 10000
        return len(self.testData) # 10000

    def __getitem__(self, idx):
        data = self.trainData[idx] if self.train else self.valData[idx] if self.val else self.testData[idx]
        label = int(data[-1])
        x = data[:-1]
        oh = [0., 0.]
        oh[label] = 1.
        return x, torch.tensor(oh)

    def ns(self, data):
        return self.normalization(self.standardization(data)).tolist()

    def normalization(self, data):
        _range = np.max(abs(data))
        return data / _range

    def standardization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma

class Net2(nn.Module):
    def __init__(self, in_feature = 912 * 2, hidden_dim = 4096) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(num_features = in_feature),
            nn.Linear(in_feature, hidden_dim),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
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


class Net22(nn.Module): #33epoch 8528 512 0.00001
    def __init__(self, in_feature = 912*2, hidden_dim = 4096) -> None:
        super().__init__()
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(num_features = in_feature),
            nn.Linear(in_feature, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        return self.fc2(x)

class Net23(nn.Module):
    def __init__(self, hidden_dim = 4096) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # nn.Conv1d(in_channels = 1, out_channels=2, kernel_size=11, padding = 5),
            # nn.LeakyReLU(),
            # nn.Conv1d(in_channels = 2, out_channels=4, kernel_size=9, padding = 4),
            # nn.LeakyReLU(),
            # nn.Conv1d(in_channels = 4, out_channels=8, kernel_size=7, padding = 3),
            # #nn.MaxPool1d(kernel_size =2, stride=2),
            # nn.LeakyReLU(),
            # nn.Conv1d(in_channels = 8, out_channels=16, kernel_size=5, padding = 2),
            # nn.MaxPool1d(kernel_size =2, stride=2),
            # nn.LeakyReLU(),
            # nn.Conv1d(in_channels = 1, out_channels=8, kernel_size=3, padding = 1),
            # #nn.MaxPool1d(kernel_size =2, stride=2),
            # nn.LeakyReLU(),
            # nn.Conv1d(in_channels = 8, out_channels=4, kernel_size=3, padding = 1),
            # nn.MaxPool1d(kernel_size =2, stride=2),
            # nn.LeakyReLU(),
            nn.Conv1d(in_channels = 1, out_channels=2, kernel_size=3, padding = 1),
            #nn.MaxPool1d(kernel_size =2, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels = 2, out_channels=1, kernel_size=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=1824,out_features=32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            nn.LeakyReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = torch.unsqueeze(input=x,dim=1)
        out = self.conv(x)
        out = torch.squeeze(input=out,dim=1)
        return self.fc(out)

def mission2(model, data, batchsize, lr, device, addfactor = 3): #data.shape = 70026, 1825
    print("Using {} device".format(device))
    print('mission 2 start')
    training_data = Dataset2(data=data, train = True)
    val_data = Dataset2(data=data, train = False, val = True)
    # test_data = Dataset2(data=data, train = False, val = False)
    train_dataloader = DataLoader(training_data, batch_size = batchsize, shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = 10, shuffle = True)
    # test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle = True)

    weight=torch.tensor([0.28, 0.72]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight = weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    tmax=0
    for i in range(EPOCH):
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('loss ', loss.item())
        # val
        model.eval()
        total = len(val_data)
        correct = 0
        Ctotal = 0
        CXtotal = 0
        C = 0
        CX = 0
        for batch, (x, y) in enumerate(val_dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            labs = torch.argmax(pred, dim = 1)
            gt = torch.argmax(y, dim = 1)
            for j in range(x.shape[0]):
                if labs[j] == gt[j]:
                    correct += 1
                    if gt[j] == 1:
                        C += 1
                    else:
                        CX += 1
                if gt[j] == 1:
                    Ctotal += 1
                else:
                    CXtotal += 1
        acc = correct / total
        if acc>tmax:
            tmax=acc
        Cacc = C / Ctotal
        CXacc = CX / CXtotal
        print('EPOCH:', i,'total:', acc, 'C:', round(Cacc, 2), 'CX:', round(CXacc, 2))
        if (i+1) % 7 == 0 or C == 0 or CX == 0:
            print('update lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.3
        if acc < 0.8 and C !=0 and CX != 0:
            print('add lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= addfactor
    print('tmax: ', tmax)

def loadData(datapath = DATAPATH):
    print('Now loading data')
    cache_dir = os.getcwd() + '/data_fixed.npy'
    print(cache_dir)
    if os.path.exists(cache_dir):
        print('use cache')
        return torch.from_numpy(np.load(cache_dir).astype(np.float32))
    print('reading...')
    jsf = json.load(open(datapath))
    datal = []
    datar = []
    labels = []
    for re in jsf:
        label = 1 if len(re['l']) == 1 else 0
        labels.append([label])
        w = torch.tensor(re['w'])
        r = torch.tensor(re['r'])
        a = torch.tensor(re['a'])
        i = torch.tensor(re['i'])
        wrair = torch.cat((w,r,a,i), dim = 0)
        datar.append(wrair.numpy())
        w = nn.functional.normalize(w, p=2.0, dim=0, eps=1e-12, out=None)
        r = nn.functional.normalize(r, p=2.0, dim=0, eps=1e-12, out=None)
        a = nn.functional.normalize(a, p=2.0, dim=0, eps=1e-12, out=None)
        i = nn.functional.normalize(i, p=2.0, dim=0, eps=1e-12, out=None)
        wrail = torch.cat((w,r,a,i), dim = 0)
        wrail = nn.functional.normalize(wrail, p=2.0, dim=0, eps=1e-12, out=None) #??????????????
        datal.append(wrail.numpy())
    datal = torch.tensor(datal)
    datar = torch.tensor(datar)
    datar = nn.functional.normalize(datar, p=2.0, dim=0, eps=1e-12, out=None)
    data = torch.cat((datal, datar, torch.tensor(labels)), dim = 1).numpy()

    np.random.shuffle(data)
    np.save(cache_dir, data)
    print('cache saved')
    return torch.tensor(data)

if __name__ == '__main__':
    data = loadData(DATAPATH)
    print(data.shape)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #mission2(model = Net22().to(device), data = data, batchsize=512, device=device, lr = 0.00003) #23 0.8972 0.3 0.7
    #mission2(model = Net23().to(device), data=data, batchsize=1024, device=device, lr = 100)
    mission2(model = Net2().to(device), data=data, batchsize=512, device=device, lr = 0.00003, addfactor = 3)

    exit()