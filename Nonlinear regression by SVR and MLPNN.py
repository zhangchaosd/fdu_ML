####
#Design a regression system to predict housing prices. 
#The data are available at: https://www.kaggle.com/vikrishnan/boston-house-prices  (also available at Sklearn).
#The regression algorithms should contain support vector regression and MLPNN.  
####

import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np

#运行前设置数据路径
DATAPATH = "D:/DATASETS/housing.csv"

#fc-relu-fc-relu-fc
class regression(nn.Module):
    def __init__(self,eps):
        super(regression,self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(14,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.w = nn.Parameter(torch.ones(2))
        self.b = nn.Parameter(torch.zeros(1))
        self.eps = eps
    def forward(self, x):
        num = x.shape[0]
        x = self.layers(x)
        x = x[x > self.eps].sum() - x[x < -self.eps].sum() #SVM Loss
        return x/num



if __name__ == '__main__':
    with open(DATAPATH, newline='') as csvfile:
        #Read data
        datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        num_total = 506
        X = np.zeros((num_total, 14))
        r = 0
        for row in datareader:
            c = 0
            for i in range(len(row)):
                if len(row[i]) > 0:
                    X[r, c] = float(row[i])
                    c += 1
            r += 1
        np.random.shuffle(X) #Shuffle the data to split
        #Split the dataset to training set, val set and test set
        num_val = num_total // 6
        num_test = num_val
        num_train = num_total - num_val - num_test
        X_train = X[0:num_train].copy()
        X_val = X[num_train:num_train + num_val].copy()
        X_test = X[num_train + num_val:num_train + num_val + num_test].copy()

        X_train = torch.tensor(X_train).float()
        X_val = torch.tensor(X_val).float()
        X_test = torch.tensor(X_test).float()
        model = regression(0.1)
        opt = optim.SGD(model.parameters(),lr = 0.00003)
        for i in range(50):
            loss = model(X_train)
            print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        valloss = model(X_val)
        print("valloss: ", valloss)
        testloss = model(X_test)
        print("testloss", testloss)
        print("done")