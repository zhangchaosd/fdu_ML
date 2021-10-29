import struct
import numpy as np
from sklearn import svm
import sklearn
import matplotlib.pyplot as plt
import matplotlib

DATA_PATH = 'D:/DATASETS/MNIST/'

train_images_idx3_ubyte_file = DATA_PATH + 'train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = DATA_PATH + 'train-labels.idx1-ubyte'

test_images_idx3_ubyte_file = DATA_PATH + 't10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = DATA_PATH + 't10k-labels.idx1-ubyte'

######读取数据#####
def decode_idx3_ubyte(idx3_ubyte_file):
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

def decode_idx1_ubyte(idx1_ubyte_file):
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
    return labels

def load_images(idx_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_labels(idx_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)
###读取数据结束######

def trainAndTest(train_images, train_labels, test_images, test_labels, kernal = 'rbf'):
    classifier=svm.SVC(C=1,kernel=kernal,decision_function_shape='ovr')
    classifier.fit(train_images, train_labels)
    #计算分类器的准确率
    print(kernal+':')
    print("训练集：",classifier.score(train_images,train_labels))
    print("测试集：",classifier.score(test_images,test_labels))


if __name__=='__main__':
    #只取一部分数据训练，全部数据运行时间太长
    train_images = load_images(train_images_idx3_ubyte_file)[:6000]
    train_labels = load_labels(train_labels_idx1_ubyte_file)[:6000]
    test_images = load_images(test_images_idx3_ubyte_file)
    test_labels = load_labels(test_labels_idx1_ubyte_file)
    trainAndTest(train_images,train_labels,test_images,test_labels,'linear')
    trainAndTest(train_images,train_labels,test_images,test_labels,'rbf')

'''
运行结果：
linear:
训练集： 1.0
测试集： 0.91
rbf:
训练集： 0.9863333333333333
测试集： 0.953
'''