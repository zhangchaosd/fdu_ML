import csv
import matplotlib.pyplot as plt
import numpy as np

#运行前设置数据路径
DATAPATH = "D:/DATASETS/housing.csv"

class LinearRegression():
    def __init__(self, X, Y, lr = 0.000003, alpha1 = 0.0000001, alpha2 = 0.00000002) -> None:
        self.x = X
        self.labels = Y
        self.lr = lr
        self.num_train = X.shape[0]
        self.D = X.shape[1]
        self.w = np.ones((self.D, 1))
        #self.w = np.random.rand(self.D, 1) #Another way to init self.w
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def train(self):
        self.preds = np.matmul(self.x, self.w).reshape(-1)
        return self.getloss(preds = self.preds, labels = self.labels)

    def test(self, X, Y):
        preds = np.matmul(X, self.w).reshape(-1)
        return self.getloss(preds = preds, labels = Y)

    def backward(self):
        #First I need to compute these gradients manually

        #The gradient of Quadratic loss
        dw = np.matmul(self.x.T, self.preds).reshape((-1, 1))

        #The gradient of LASSO regression
        tdw = self.w.copy()
        tdw[tdw > 0] = 0.5
        tdw[tdw < 0] = -0.5
        dw += tdw #L1

        #The gradient of ridge regression
        dw += self.w #L2

        dw /= self.num_train
        #Update w
        self.w -= self.lr * dw

    def getloss(self, preds, labels):
        num = len(preds)
        loss = np.sum((preds - labels) ** 2) #Quadratic loss
        loss += self.alpha1 * np.sum(np.abs(self.w)) #L1 LASSO regression
        loss += self.alpha2 * np.sum(self.w ** 2) #L2 Ridge regression
        loss /= 2 * num
        return loss


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
        Y = X[:, -1].copy()
        X[:, -1] = 1 #Use last column as bias

        #Split the dataset to training set, val set and test set
        num_val = num_total // 6
        num_test = num_val
        num_train = num_total - num_val - num_test
        X_train = X[0:num_train].copy()
        Y_train = Y[0:num_train].copy()
        X_val = X[num_train:num_train + num_val].copy()
        Y_val = Y[num_train:num_train + num_val].copy()
        X_test = X[num_train + num_val:num_train + num_val + num_test].copy()
        Y_test = Y[num_train + num_val:num_train + num_val + num_test].copy()

        model = LinearRegression(X_train, Y_train, lr = 0.000003)

        epoch = 50001
        xs_train = []
        ys_train = []
        xs_val = []
        ys_val = []
        for i in range(epoch): #main loop
            loss = model.train()
            xs_train.append(i)
            ys_train.append(loss)
            if i % 500 == 0:
                print('tarin:  epoch:', i, '/50000    LOSS:', loss)
                xs_val.append(i)
                val_loss = model.test(X_val, Y_val)
                ys_val.append(val_loss)
                print('val loss:  ', val_loss)
            model.backward()

        test_loss = model.test(X_test, Y_test)
        print('Final test loss:   ', test_loss)

        #Visualization
        plt.xlabel('epoch')
        plt.ylabel('loss')
        xs_test = np.linspace(0, epoch, epoch)
        ys_test = np.linspace(test_loss, test_loss,epoch)
        plt.plot(xs_train, ys_train, color = 'red', label = 'train')
        plt.plot(xs_val, ys_val, color = 'blue', label = 'val')
        plt.plot(xs_test, ys_test, color = 'green', label = 'test')
        plt.legend(loc = 'upper right')
        plt.ylim((0, 1000))
        plt.show()

        #save model