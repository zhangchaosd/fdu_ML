import struct
import numpy as np

DATA_PATH = 'D:/DATASETS/MNIST/'

train_images_idx3_ubyte_file = DATA_PATH + 'train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = DATA_PATH + 'train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = DATA_PATH + 't10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = DATA_PATH + 't10k-labels.idx1-ubyte'

######读取数据#####
def load_images(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def load_labels(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    labels = [int(i) for i in labels]
    return labels
###读取数据结束######


def fc_forward(x, w, b):
    out = x.dot(w) + b
    cache = (x, w, b)
    return out, cache

def fc_backward(dout, cache):
    x, w, b = cache
    dx = dout.dot(w.T)
    dw = x.T.dot(dout)
    db = np.sum(dout, axis = 0)
    return dx, dw, db

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    dx = (x > 0) * dout
    return dx

def sigmod_forward(x):
    out = 1. / (1 + np.exp(-x))
    cache = x
    return out, cache

def sigmod_backward(dout, cache):
    dx, x = None, cache
    s = 1. / (1 + np.exp(-x))
    dx = dout * (1 - s) * (s)
    return dx

def fc_relu_forward(x, w, b):
    a, fc_cache = fc_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def fc_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = fc_backward(da, fc_cache)
    return dx, dw, db

def fc_sigmod_forward(x, w, b):
    a, fc_cache = fc_forward(x, w, b)
    out, sigmod_cache = sigmod_forward(a)
    cache = (fc_cache, sigmod_cache)
    return out, cache


def fc_sigmod_backward(dout, cache):
    fc_cache, sigmod_cache = cache
    da = sigmod_backward(dout, sigmod_cache)
    dx, dw, db = fc_backward(da, fc_cache)
    return dx, dw, db

def softmax_loss(x, y):
    loss, dx = None, None

    N = x.shape[0]
    C = x.shape[1]
    shifted_x = x - np.max(x, axis = 1, keepdims = True)
    Z = np.sum(np.exp(shifted_x), axis = 1, keepdims = True)
    loss = np.sum(np.log(Z)) - np.sum(shifted_x[range(N), y])
    loss /= N

    dx = np.exp(shifted_x) / Z
    dx[range(N),y] -= 1
    dx /= N

    return loss, dx

class MLPNN():
    def __init__(self,input_dim = 28 * 28, hidden_dim = 100, num_classes = 10, act = 'relu', weight_scale = 1e-3, reg = 0.0):
        self.params = {}
        self.act = act
        self.reg = reg
        self.params['W1'] = np.random.normal(0, weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y = None):
        scores = None
        if self.act == 'relu':
            hidden, fc_relu_cache = fc_relu_forward(X, self.params['W1'], self.params['b1'])
        elif self.act == 'sigmod':
            hidden, fc_sigmod_cache = fc_sigmod_forward(X, self.params['W1'], self.params['b1'])
        scores, fc_cache = fc_forward(hidden, self.params['W2'], self.params['b2'])
        #没有传 y，则直接返回预测值
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dout = softmax_loss(scores, y)
        dhidden, dw2, grads['b2'] = fc_backward(dout, fc_cache)
        grads['W2'] = dw2

        if self.act == 'relu':
            dx, dw1, grads['b1'] = fc_relu_backward(dhidden, fc_relu_cache)
        elif(self.act == 'sigmod'):
            dx, dw1, grads['b1'] = fc_sigmod_backward(dhidden, fc_sigmod_cache)
        grads['W1'] = dw1

        return loss, grads


def train(train_images, train_labels, test_images, test_labels, epoch = 100, act = 'sigmod', lr = 0.0001):
    model = MLPNN()
    for i in range(epoch):
        loss, grads = model.loss(train_images, train_labels)
        print(loss)
        for p, w in model.params.items():
            dw = grads[p]
            next_w = w - lr * dw
            model.params[p] = next_w

    #训练完成，查看在训练集和测试集上的正确率
    pred_train = model.loss(train_images)
    pred = np.argmax(pred_train, axis = 1)
    num_train = train_images.shape[0]
    correct = np.sum(pred == train_labels)
    print("train acc: ", correct/num_train)

    pred_test = model.loss(test_images)
    pred = np.argmax(pred_test, axis = 1)
    num_test = test_images.shape[0]
    correct = np.sum(pred == test_labels)
    print("test acc: ", correct/num_test)


if __name__=='__main__':
    #只取一部分数据训练，全部数据运行时间太长
    train_images = load_images(train_images_idx3_ubyte_file)[:6000]
    train_labels = load_labels(train_labels_idx1_ubyte_file)[:6000]
    test_images = load_images(test_images_idx3_ubyte_file)
    test_labels = load_labels(test_labels_idx1_ubyte_file)
    train(train_images, train_labels, test_images, test_labels, epoch = 100, act = 'sigmod', lr = 0.001) #激活函数可选 relu 或 sigmod