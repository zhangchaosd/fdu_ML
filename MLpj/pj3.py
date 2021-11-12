import torch
from torch._C import device
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *

#DATAPATH = 'D:/9709/Desktop/works/fdu_ML/MLpj/天纺标数据/'
DATAPATH = 'C:/Users/97090/Desktop/fdu_ML/MLpj/天纺标数据/'
BATCHSIZE = 256
EPOCH = 2000

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
    def __init__(self, in_feature = 1307, hidden_dim = 2048) -> None:
        super().__init__()
        self.cv161024 = nn.Sequential(
            nn.Linear(in_feature, hidden_dim),
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2), #1024
        )
        self.cv32512 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2), #512
        )
        self.cv64256 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2), #256
        )
        self.cv6464 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2), #128

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2), #64
        )
        self.cv3232 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2), #32
        )
        self.cv116 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            #nn.MaxPool1d(kernel_size = 2, stride = 2) #16
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32, 2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        out = x.unsqueeze(1)
        out1 = self.cv161024(out) #n,16,1024
        out2 = self.cv32512(out1) #n,32,512
        out3 = self.cv64256(out2) #n,64,256
        out4 = self.cv6464(out3) #n,64,64
        #out4 = torch.cat([out4,out3], dim = 1) #n,64,320
        out5 = self.cv3232(out4) #n,32,32
        #out5 = torch.cat([out5,out2], dim = 1) #n,32,544
        #print('out5', out5.shape)
        out6 = self.cv116(out5) #n,1,16
        ret = out6.squeeze(1)
        return self.fc2(ret)
        '''
        self.wq = nn.Linear(in_features = in_feature, out_features = hidden_dim)
        self.wk = nn.Linear(in_features = in_feature, out_features = hidden_dim)
        self.wv = nn.Linear(in_features = in_feature, out_features = hidden_dim)
        self.softmax = nn.Softmax(dim = 0)
        self.fc = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048,512),
            nn.LeakyReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        A = torch.matmul(K.t(), Q)
        A = self.softmax(A)
        b = torch.matmul(V, A)
        return self.fc(b)
        '''
#------------------------------------------------------------

def mission3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    training_data = Dataset3(DATAPATH, train = True)
    test_data = Dataset3(DATAPATH, train = False)
    train_dataloader = DataLoader(training_data, batch_size = BATCHSIZE, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle = True) #676
    print('mission 3 start')
    model = Net3().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00003)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    lambda1 = lambda epoch:np.sin(epoch) / epoch
    scheduler = StepLR(optimizer,step_size=5,gamma = 0.8)
    f = 0.1
    for i in range(EPOCH):
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
        #scheduler.step()
        
        # test
        model.eval()
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            MAE = torch.mean(torch.abs(pred[:,0] - y[:,0]))
            print('EPOCH: ', i, 'test MAE: ', MAE )
            if MAE < f:
                f*=0.8
                print('update lr')
                nam = 'model_weights'+str(i)+'.pth'
                torch.save(model.state_dict(), nam)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

#最低0.1150 分类 AdamW(model.parameters(), lr = 0.0000001) batchsize 64 net3
#0.08
#0.075  epoch250  batchsize 256 lr 0.000003 
if __name__ == '__main__':
    mission3()
    exit()