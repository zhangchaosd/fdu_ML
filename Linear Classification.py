import numpy as np
import struct
import matplotlib.pyplot as plt

DATA_PATH = 'D:/DATASETS/MNIST/'

train_images_idx3_ubyte_file = DATA_PATH + 'train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = DATA_PATH + 'train-labels.idx1-ubyte'

test_images_idx3_ubyte_file = DATA_PATH + 't10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = DATA_PATH + 't10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'   #'>IIII'是说使用大端法读取4个unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'   # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


if __name__ == '__main__':
    train_images = load_train_images() #(num_rows*num_cols,num_images)
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    train_labels[train_labels>0]=0
    Y = np.zeros((train_labels.shape[0], 2))
    for i in range(train_labels.shape[0]):
        Y[int(train_labels[i])] = 1
    X = np.ones((train_images.shape[0], train_images.shape[1] + 1))
    X[:,0:-1]=train_images

    #带正则化项的最小二乘法
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X) + 0.01*np.identity(X.shape[1])), X.T), Y)
    #
    Xt = np.ones((test_images.shape[0], test_images.shape[1] + 1))
    Xt[:,0:-1]=test_images
    pred = np.matmul(Xt, W)
    #logistic regression
    pred = 1 / (1 + np.exp(-pred))
    tt = Xt.shape[0]
    cor = 0
    for i in range(Xt.shape[0]):
        if test_labels[i]==0:
            if pred[i][0]==1:
                cor += 1
        elif pred[i][0]!=0:
            cor += 1


