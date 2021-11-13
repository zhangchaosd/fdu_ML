import torch
import json
import numpy as np

DATAPATH = 'D:/9709/Desktop/works/fdu_ML/MLpj/布眼数据集.json'
#DATAPATH = 'C:/Users/97090/Desktop/fdu_ML/MLpj/布眼数据集.json'


#filedir = 'D:/9709/Desktop/works/fdu_ML/MLpj/一条数据.json' #一条数据  布眼数据集
#    jsf = json.load(open(filedir))
#    print(len(jsf))

class Dataset2(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_dir = DATAPATH, train = True, val = False, shuffle = True):
        self.jsf = json.load(open(dataset_dir)) #70027
        if shuffle:
            np.random.shuffle(self.jsf)
        self.trainData = self.jsf[:10000]
        self.valData = self.jsf[10000:20000]
        self.testData = self.jsf[20000:]
        self.train = train
        self.val = val

    def __len__(self):
        if self.train:
            return len(self.trainData)
        if self.val:
            return len(self.valData)
        return len(self.testData)

    def __getitem__(self, idx):
        if self.train:
            return torch.from_numpy(self.x_train[idx]), torch.from_numpy(self.y_train[idx])
        return torch.from_numpy(self.x_test[idx]), torch.from_numpy(self.y_test[idx])


def mission2():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    training_data = Dataset3(DATAPATH, train = True)
    test_data = Dataset3(DATAPATH, train = False)
    train_dataloader = DataLoader(training_data, batch_size = BATCHSIZE, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle = True) #676
    print('mission 3 start')







if __name__ == '__main__':
    mission2()
    exit()