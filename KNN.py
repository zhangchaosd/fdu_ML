from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#为了运行速度，只取一部分数据
tX = np.array(test_data.data[0:1000], dtype = int)
y_test = np.array(test_data.test_labels.data[0:1000])
tX_train = np.array(training_data.data[0:6000], dtype = int)
y_train = np.array(training_data.train_labels.data[0:6000])

num_test = tX.shape[0]
num_train = tX_train.shape[0]

#flatten
X = tX.reshape(num_test,-1)
X_train = tX_train.reshape(num_train, -1)
dists = np.zeros((num_test, num_train))

def GetDists(dists):
    dists += np.sum(X_train ** 2, axis = 1).reshape(1, num_train)
    dists += np.sum(X ** 2, axis = 1).reshape(num_test, 1)
    dists -= 2 * np.matmul(X, X_train.T)

def KNNcore(k=1):
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = np.argsort(dists[i])
        vt = [0,0,0,0,0,0,0,0,0,0,0]
        for j in range(k):
            vt[y_train[closest_y[j]]] += 1
        tmax = 0
        for j in range(len(vt)):
            if tmax <= vt[j]:
                tmax = vt[j]
                y_pred[i] = j
    num_correct = np.sum(y_test == y_pred)
    accuracy = float(num_correct) / num_test
    print('k = %d :Got %d / %d correct => accuracy: %f' % (k, num_correct, num_test, accuracy))

GetDists(dists)
Ks = {1, 3, 5, 7}
for k in Ks:
    KNNcore(k)
'''
result:
k = 1 :Got 904 / 1000 correct => accuracy: 0.904000
k = 3 :Got 918 / 1000 correct => accuracy: 0.918000
k = 5 :Got 919 / 1000 correct => accuracy: 0.919000
k = 7 :Got 913 / 1000 correct => accuracy: 0.913000
'''

