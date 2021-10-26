from numpy import * 

import struct

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
    print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'
    images = empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        images[i] = array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_images(idx_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_labels(idx_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

def selectJrand(i,m): #在0-m中随机选择一个不是i的整数
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  #保证a在L和H范围内（L <= a <= H）
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def kernelTrans(X, A, kTup): #核函数，输入参数,X:支持向量的特征树；A：某一行特征数据；kTup：('lin',k1)核函数的类型和参数
    #X (29, 2)
    #A (1, 2)
    m,n = shape(X) #29, 2
    K = mat(zeros((m,1))) #(29, 1)
    if kTup[0]=='lin': #线性函数
        K = X * A.T
    elif kTup[0]=='rbf': # 径向基函数(radial bias function)
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #返回生成的结果
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    #(29, 1)
    return K


#定义类，方便存储数据
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
        self.X = dataMatIn  #数据特征 (100,2)
        self.labelMat = classLabels #数据类别 (100,1)
        self.C = C #软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler #停止阀值
        self.m = shape(dataMatIn)[0] #数据行数 100
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 #初始设为0
        self.eCache = mat(zeros((self.m,2))) #缓存
        self.K = mat(zeros((self.m,self.m))) #核函数的计算结果
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


def calcEk(oS, k): #计算Ek（参考《统计学习方法》p127公式7.105）
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#随机选取aj，并返回其E值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  #返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): #返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k): #更新os数据
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

#首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i, oS): #输入参数i和所有参数数据
    Ei = calcEk(oS, i) #计算E值
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)): #检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
        j,Ej = selectJ(i, oS, Ei) #随机选取aj，并返回其E值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]): #以下代码的公式参考《统计学习方法》p126
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            #print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #参考《统计学习方法》p127公式7.107
        if eta >= 0:
            #print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta #参考《统计学习方法》p127公式7.106
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) #参考《统计学习方法》p127公式7.108
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol): #alpha变化大小阀值（自己设定）
            #print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#参考《统计学习方法》p127公式7.109
        updateEk(oS, i) #更新数据
        #以下求解b的过程，参考《统计学习方法》p129公式7.114-7.116
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


#SMO函数，用于快速求解出alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)): #输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m): #遍历所有数据
                alphaPairsChanged += innerL(i,oS)
                #print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)) #显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: #遍历非边界的数据
                alphaPairsChanged += innerL(i,oS)
                #print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        #print("iteration number: %d" % iter)
    return oS.b,oS.alphas

if __name__=='__main__':
    train_images = load_images(train_images_idx3_ubyte_file) #(num_rows*num_cols,num_images)
    train_labels = load_labels(train_labels_idx1_ubyte_file)
    train_labels[train_labels != 0] = -1
    train_labels[train_labels == 0] = 1

    dataArr = train_images
    labelArr = train_labels

    #kernel = 'lin'
    kernel = 'rbf'

    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, (kernel, 1.3)) #通过SMO算法得到b和alpha
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas)[0]  #选取不为0数据的行数（也就是支持向量）
    sVs=datMat[svInd] #支持向量的特征数据
    labelSV = labelMat[svInd] #支持向量的类别（1或-1）
    print("there are %d Support Vectors" % shape(sVs)[0]) #打印出共有多少的支持向量
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],(kernel, 1.3))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))

    test_images = load_images(test_images_idx3_ubyte_file)
    test_labels = load_labels(test_labels_idx1_ubyte_file)
    test_labels[test_labels != 0] = -1
    test_labels[test_labels == 0] = 1

    dataArr_test = test_images
    labelArr_test = test_labels


    errorCount_test = 0
    datMat_test=mat(dataArr_test) #(100.2)
    labelMat = mat(labelArr_test).transpose() #(100,1)
    m,n = shape(datMat_test) #100, 2
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat_test[i,:],('rbf', 1.3)) #(29, 1)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr_test[i]):
            errorCount_test += 1
    print("the test error rate is: %f" % (float(errorCount_test)/m))
