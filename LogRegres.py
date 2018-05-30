from numpy import *
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat=[]
    classLabels=[]
    f=open('D:\\学习资料\\machinelearninginaction\\Ch05\\testSet.txt')
    for line in f.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])  # 类似float 只能一个参数 不能一个序列
        classLabels.append(int(lineArr[-1]))
    return dataMat,classLabels

def sigmoid(VecZ):
    return 1.0/(1+exp(-VecZ))

def gradAscent(dataMat,classLabels):
    dataMatrix=mat(dataMat)
    labelMatrix=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    weight=ones((n,1))
    alpha=0.001
    iternum=500
    for i in range(iternum):
        h=sigmoid(dataMatrix*weight)
        error=labelMatrix-h
        weight+=alpha*(dataMatrix.transpose()*error)
    return weight

def stogradAscent(dataMat,classLabels):
    dataMatrix=mat(dataMat)
    labelMatrix=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    weights=ones((n,1))
    alpha=0.01
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=labelMatrix[i]-h
        weights+=alpha*dataMatrix[i].transpose()*error
    return weights

# def stogradAscent1(dataMat,classLabels,numiter=150):
#     dataMatrix=mat(dataMat)
#     labelMatrix=mat(classLabels).transpose()
#     m,n=shape(dataMatrix)
#     weights=ones((n,1))
#     for i in range(numiter):
#         trainMatrix=dataMatrix
#         trainlabelMatrix=labelMatrix
#         for j in range(m):
#             alpha=4/(1+i+j)+0.001
#             randIndex=int(random.uniform(0,len(trainMatrix)))
#             h = sigmoid(sum(trainMatrix[randIndex] * weights))
#             error = trainlabelMatrix[randIndex] - h
#             weights += alpha * trainMatrix[randIndex].transpose() * error
#             delete(trainMatrix,randIndex,axis=0);delete(trainlabelMatrix,randIndex,axis=0)
#     return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# dataMat,classLabels=loadDataSet()
# weights=gradAscent(dataMat,classLabels)
# Stoweights=stogradAscent(dataMat,classLabels)
# Stoweights1=stogradAscent1(dataMat,classLabels)
# print(Stoweights1)

def plotBestFit(dataMat,classLabels,weights):
    xcord0=[];ycord0=[]
    xcord1=[];ycord1=[]
    dataArr=array(dataMat)
    for i in range(shape(dataArr)[0]):
        if classLabels[i]==0:
            xcord0.append(dataArr[i][1]);ycord0.append(dataArr[i][2])
        else:
            xcord1.append(dataArr[i][1]);ycord1.append(dataArr[i][2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord0,ycord0,color='red',marker='s')
    ax.scatter(xcord1,ycord1,color='yellow')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2] # W2*X2+W1*X1+W0=0
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

# plotBestFit(dataMat,classLabels,weights)
# plotBestFit(dataMat,classLabels,Stoweights1)

def classifyVector(inX,weights):
    prob=sum(inX*weights)
    if sigmoid(prob)>0.5:
        return 1
    else:
        return 0

def horseTest():
    filetrain=open("D:\学习资料\machinelearninginaction\Ch05\horseColicTraining.txt")
    dataMat=[]
    labelMat=[]
    for line in filetrain.readlines():
        lineArr=line.strip().split()
        dataMat.append([float(x) for x in lineArr[:-1]])
        labelMat.append(float(lineArr[-1]))
    weights=stocGradAscent1(array(dataMat),labelMat,100)
    errorcount=0
    index=1
    filetest=open("D:\学习资料\machinelearninginaction\Ch05\horseColicTest.txt")
    for line in filetest.readlines():
        lineArr=line.strip().split('\t')
        testclass=classifyVector([float(x) for x in lineArr[:-1]],weights)
        if testclass != float(lineArr[-1]):
            errorcount+=1
            print("the %d is wrong, test result is %f,but real result is %f"%(index,testclass,float(lineArr[-1])))
        index+=1
    print("the classify error rate is %f"%(errorcount/67))

horseTest()