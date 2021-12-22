import torch
import torch.nn as nn
import json
import numpy as np
from torch.nn.modules import normalization
from torch.nn.modules.activation import Softmax
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os

#DATAPATH = 'D:/9709/Desktop/works/fdu_ML/MLpj/布眼数据集.json'
DATAPATH = 'C:/Users/97090/Desktop/fdu_ML/MLpj/布眼数据集fixed.json'
DATAPATH0 = 'C:/Users/97090/Desktop/fdu_ML/MLpj/一条数据.json'
# 布眼数据集fixed.json: https://drive.google.com/file/d/1SBX_lIny3KfxX2JOaePPnQWNBavSvGs5/view?usp=sharing
BATCHSIZE = 512
EPOCH = 70

class Dataset2(torch.utils.data.dataset.Dataset):
    def __init__(self, dic, ids, isTrain = True):
        np.random.shuffle(ids)
        self.dic = dic
        num_total = len(ids)
        num_train = int(num_total * 0.9)
        self.train_ids = ids[:num_train]
        self.val_ids = ids[num_train:]
        self.train = []
        self.val = []
        for id in self.train_ids:
            self.train = self.train + dic[id]
        for id in self.val_ids:
            self.val = self.val + dic[id]
        self.isTrain = isTrain

    def __len__(self):
        if self.isTrain:
            return len(self.train)
        return len(self.val_ids)

    def __getitem__(self, idx):
        if self.isTrain:
            data = self.train[idx]
            label = int(data[-1])
            x = data[:-1]
            oh = [0., 0.]
            oh[label] = 1.
            return x, torch.tensor(oh)
        else:
            x = []
            id = self.val_ids[idx]
            datas = self.dic[id]
            for data in datas:
                x = x + data
        return x, torch.tensor([0., 0.])

'''
fc 1824 4096
bn
leakyrelu 0.01

fc 4096 4096
bn
leakyrelu 0.01
x3

fc 4096 1024
bn
leakyrelu 0.01

fc 1024 128
bn
leakyrelu 0.01

fc 128 2
softmax

'''

class Net2(nn.Module):
    def __init__(self, in_feature = 912*2, hidden_dim = 4096) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(num_features = in_feature),
            nn.Linear(in_feature, hidden_dim),
            nn.BatchNorm1d(num_features = hidden_dim),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features = hidden_dim),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features = hidden_dim),
            nn.LeakyReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features = hidden_dim),
            nn.LeakyReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(num_features = 1024),
            nn.LeakyReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(num_features = 128),
            nn.LeakyReLU()
        )
        self.fc7 = nn.Sequential(
            nn.Linear(128, 2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        # out3 = self.fc3(out2 + out1)
        out3 = self.fc3(out2)
        # out4 = self.fc4(out3 + out2 + out1)
        out4 = self.fc4(out3)
        out5 = self.fc5(out4)
        out6 = self.fc6(out5)
        return self.fc7(out6)

def mission2(model, data, batchsize, lr, device, addfactor = 3): #data.shape = 70026, 1825
    print("Using {} device".format(device))
    print('mission 2 start')
    training_data = Dataset2(data = data, train = True)
    val_data = Dataset2(data = data, train = False, val = True)
    # test_data = Dataset2(data=data, train = False, val = False)
    train_dataloader = DataLoader(training_data, batch_size = batchsize, shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = 10, shuffle = True)
    # test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle = True)

    weight = torch.tensor([0.28, 0.72]).to(device) #965
    loss_fn = nn.CrossEntropyLoss(weight = weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    tmax = 0
    for i in range(EPOCH):
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print('loss ', loss.item())

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
        if acc > tmax:
            tmax = acc
        Cacc = C / Ctotal
        CXacc = CX / CXtotal
        print('EPOCH:', i,'total:', acc, 'C:', round(Cacc, 2), 'CX:', round(CXacc, 2))
        if (i + 1) % 7 == 0 or C == 0 or CX == 0:
            print('update lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.3
        if acc < 0.8 and C != 0 and CX != 0:
            print('add lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= addfactor
    print('tmax: ', tmax)
    return tmax


def mission25(model, dic, ids, batchsize, lr, device, addfactor = 3): #data.shape = 70026, 1825
    print("Using {} device".format(device))
    print('mission 2 start')
    training_data = Dataset2(dic = dic,ids = ids, train = True)
    val_data = Dataset2(dic = dic,ids = ids, train = False, val = True)
    # test_data = Dataset2(data=data, train = False, val = False)
    train_dataloader = DataLoader(training_data, batch_size = batchsize, shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = 10, shuffle = True) # batchsize
    # test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle = True)

    weight = torch.tensor([0.28, 0.72]).to(device) #965
    loss_fn = nn.CrossEntropyLoss(weight = weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    tmax = 0
    for i in range(EPOCH):
        model.train()
        for _, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print('loss ', loss.item())
        continue
        # val
        model.eval()
        total = len(val_data)
        correct = 0
        Ctotal = 0
        CXtotal = 0
        C = 0
        CX = 0
        for _, (x, y) in enumerate(val_dataloader):
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
        if acc > tmax:
            tmax = acc
        Cacc = C / Ctotal
        CXacc = CX / CXtotal
        print('EPOCH:', i,'total:', acc, 'C:', round(Cacc, 2), 'CX:', round(CXacc, 2))
        if (i + 1) % 7 == 0 or C == 0 or CX == 0:
            print('update lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.3
        if acc < 0.8 and C != 0 and CX != 0:
            print('add lr')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= addfactor
    print('tmax: ', tmax)
    return tmax

def loadData(datapath = DATAPATH):
    print('Now loading data')
    cache_dir = os.getcwd() + '/data_fixed.npy'
    print(cache_dir)
    if os.path.exists(cache_dir):
        print('use cache')
        data = np.load(cache_dir).astype(np.float32)
        np.random.shuffle(data)
        return torch.from_numpy(data)
    print('reading...')
    jsf = json.load(open(datapath))
    datal = []
    datar = []
    labels = []
    for re in jsf:
        label = 1 if len(re['l']) == 1 else 0
        labels.append([label])
        r = torch.tensor(re['r'])
        a = torch.tensor(re['a'])
        i = torch.tensor(re['i'])
        rair = torch.cat((r,a,i), dim = 0)
        datar.append(rair.numpy())
        r = nn.functional.normalize(r, p=2.0, dim=0, eps=1e-12, out=None)
        a = nn.functional.normalize(a, p=2.0, dim=0, eps=1e-12, out=None)
        i = nn.functional.normalize(i, p=2.0, dim=0, eps=1e-12, out=None)
        rail = torch.cat((r,a,i), dim = 0)
        datal.append(rail.numpy())
    datal = torch.tensor(datal)
    datar = torch.tensor(datar)
    datar = nn.functional.normalize(datar, p=2.0, dim=0, eps=1e-12, out=None)
    data = torch.cat((datal, datar, torch.tensor(labels)), dim = 1)
    # data = torch.cat((datal, torch.tensor(labels)), dim = 1)
    data = data.numpy()
    np.random.shuffle(data)
    np.save(cache_dir, data)
    print('cache saved')
    return torch.tensor(data)

def standardization(data):
    mu = torch.mean(data, axis=0)
    sigma = torch.std(data, axis=0)
    return (data - mu) / sigma

def loadData2(datapath):
    print('Now loading data')
    cache_dir = os.getcwd() + '/data2_fixed.npy'
    cacheids_dir = os.getcwd() + '/data2ids_fixed.npy'
    print(cache_dir)
    if os.path.exists(cache_dir):
        print('use cache')
        return np.load(cache_dir), np.load(cacheids_dir)
    print('reading...')
    jsf = json.load(open(datapath))
    data = []
    ids = []
    dic = {}
    for re in jsf:
        label = 1 if len(re['l']) == 1 else 0
        r = torch.tensor(re['r'])
        rl = standardization(r)
        a = torch.tensor(re['a'])
        al = standardization(a)
        i = torch.tensor(re['i'])
        il = standardization(i)
        l = torch.tensor([label])
        id = re['id']
        rail = torch.cat((rl,al,il), dim = 0)
        rair = torch.cat((r,a,i), dim = 0)
        rair = standardization(rair)
        data = torch.cat((rail, rair, l), dim = 0)
        if id in dic:
            dic[id].append(data)
        else:
            dic[id] = [data]
            ids.append(id)
    np.save(cache_dir, dic)
    np.save(cacheids_dir, ids)
    print('cache saved')
    return dic, ids

if __name__ == '__main__':
    ret = []
    datapath = os.getcwd() + '/MLpj/布眼数据集fixed.json'
    for i in range(10):
        dic, ids = loadData2(datapath)
        # print(data.shape)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # ret.append(mission2(model = Net2().to(device), data = data, batchsize = 128, device = device, lr = 0.00003, addfactor = 3)) # 96.5
        # ret.append(mission2(model = Net2().to(device), data = data, batchsize = 512, device = device, lr = 0.00003, addfactor = 3)) # 96.5
        ret.append(mission25(model = Net2().to(device), dic = dic, ids = ids, batchsize = 512, device = device, lr = 0.00003, addfactor = 3)) # 96.5
    print('final ret:')
    print(ret)
    exit()


'''
无残差
[0.97, 0.9718, 0.971, 0.9718, 0.9675, 0.9722, 0.9699, 0.9726, 0.9704, 0.9704]
[0.9693, 0.969, 0.9698, 0.9717, 0.968, 0.9713, 0.9694, 0.9693, 0.9708, 0.9736]

带残差
[0.9694, 0.9679, 0.9667, 0.9678, 0.966, 0.9671, 0.9665, 0.9665, 0.9675, 0.9665]
'''