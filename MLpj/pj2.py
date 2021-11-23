import torch
import torch.nn as nn
import json
import numpy as np
from torch.nn.modules import normalization
from torch.nn.modules.activation import Softmax
from torch.utils.data import DataLoader

#DATAPATH = 'D:/9709/Desktop/works/fdu_ML/MLpj/布眼数据集.json'
DATAPATH = 'C:/Users/97090/Desktop/fdu_ML/MLpj/布眼数据集fixed.json'
# 布眼数据集fixed.json: https://drive.google.com/file/d/1SBX_lIny3KfxX2JOaePPnQWNBavSvGs5/view?usp=sharing
BATCHSIZE = 512
EPOCH = 500

class Dataset2(torch.utils.data.dataset.Dataset):
    def __init__(self, jsf, train = True, val = False):
        self.jsf = jsf # 70027 fixed:70026
        self.testData = self.jsf[:10000]
        self.valData = self.jsf[10000:20000]
        self.trainData = self.jsf[20000:]
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
        label = 1 if len(data['l']) == 1 else 0
        #x = [data['h']] + [data['t']] + self.ns(data['w']) + self.ns(data['r']) + self.ns(data['a']) + self.ns(data['i'])
        x = self.ns(data['w']) + self.ns(data['r']) + self.ns(data['a']) + self.ns(data['i'])
        x2 = data['w'] + data['r'] + data['a'] + data['i']
        x = self.ns(x)
        oh = [0., 0.]
        oh[label] = 1.
        return torch.tensor(x), torch.tensor(x2), torch.tensor(oh)

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
    def __init__(self, in_feature = 912*2, hidden_dim = 2048) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            #nn.BatchNorm1d(num_features = in_feature),
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
            nn.Conv1d(in_channels = 1, out_channels=1, kernel_size=11, padding = 5),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels = 1, out_channels=1, kernel_size=9, padding = 4),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels = 1, out_channels=1, kernel_size=7, padding = 3),
            nn.MaxPool1d(kernel_size =2, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels = 1, out_channels=1, kernel_size=5, padding = 2),
            nn.MaxPool1d(kernel_size =2, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels = 1, out_channels=1, kernel_size=3, padding = 1),
            nn.MaxPool1d(kernel_size =2, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels = 1, out_channels=1, kernel_size=3, padding = 1),
            nn.MaxPool1d(kernel_size =2, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels = 1, out_channels=1, kernel_size=3, padding = 1),
            nn.MaxPool1d(kernel_size =2, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels = 1, out_channels=1, kernel_size=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=57,out_features=32),
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

def mission2(model, batchsize, lr, device):
    print("Using {} device".format(device))
    print('mission 2 start')
    jsf = json.load(open(DATAPATH))
    np.random.shuffle(jsf)
    training_data = Dataset2(jsf = jsf, train = True)
    val_data = Dataset2(jsf = jsf, train = False, val = True)
    # test_data = Dataset2(jsf = jsf, train = False, val = False)
    train_dataloader = DataLoader(training_data, batch_size = batchsize, shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = 10, shuffle = True)
    # test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle = True)

    model = Net22().to(device)
    weight=torch.tensor([0.28, 0.72]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight = weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    tmax=0
    for i in range(EPOCH):
        model.train()
        for batch, (x, x2, y) in enumerate(train_dataloader):
            nn.functional.normalize(x2, p=2.0, dim=0, eps=1e-12, out=None)
            x = torch.cat([x, x2], dim = 1)
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss ', loss.item())
        # val
        model.eval()
        total = len(val_data)
        correct = 0
        Ctotal = 0
        CXtotal = 0
        C = 0
        CX = 0
        for batch, (x, x2, y) in enumerate(val_dataloader):
            nn.functional.normalize(x2, p=2.0, dim=0, eps=1e-12, out=None)
            x = torch.cat([x, x2], dim = 1)
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
        print('EPOCH: ', i,'total: ', acc, 'C: ', Cacc, 'CX: ', CXacc)
        if (i+1) % 7 == 0 or C == 0 or CX == 0:
            print('update lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.3
        if acc < 0.8 and C !=0 and CX != 0:
            print('add lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 3
    print('tmax: ', tmax)

if __name__ == '__main__':
    #ck()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #mission2(model = Net22().to(device), batchsize=512, device=device, lr = 0.00003) #23 0.8972 0.3 0.7
    mission2(model = Net23().to(device), batchsize=1024, device=device, lr = 0.0000003)

    exit()