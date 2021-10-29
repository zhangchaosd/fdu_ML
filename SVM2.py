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


if __name__=='__main__':
    train_images = load_images(train_images_idx3_ubyte_file) #(num_rows*num_cols,num_images)
    train_labels = load_labels(train_labels_idx1_ubyte_file)
    
    test_images = load_images(test_images_idx3_ubyte_file)
    test_labels = load_labels(test_labels_idx1_ubyte_file)
    print(train_images.shape)

    classifier=svm.SVC(C=1,kernel='rbf',gamma=10,decision_function_shape='ovr') # ovr:一对多策略  'linear'
    classifier.fit(train_images, train_labels.ravel()) #ravel函数在降维时默认是行序优先   CHECK SHAPE

    #4.计算svc分类器的准确率
    print("训练集：",classifier.score(train_images,train_labels))
    print("测试集：",classifier.score(test_images,test_labels))