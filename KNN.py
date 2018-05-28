from numpy import *
import operator

def createDataSet():
    group=array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dis=[]
    for i in range(len(dataSet)):
        # distance=linalg.norm(inX,dataSet[i])
        distance=sqrt(sum(square(inX - dataSet[i])))
        dis.append(distance)
    dic={}
    for i in range(len(dataSet)):
        dic[dis[i]]=labels[i]
    dic2=sorted(dic.items(),key=operator.itemgetter(0))
    List=[x[1] for x in dic2][:k]
    max=0
    label=None
    for x in set(labels):
        if List.count(x)>max:
            max=List.count(x)
            label=x
    return  label

def file2matrix(filename):
    f=open(filename)
    labels=[]
    rf=f.readlines()
    Mat=zeros([len(rf),3])
    index=0
    for line in rf:
        Mat[index,:]=line.strip().split()[:-1]
        label=line.strip().split()[-1]
        index+=1
        labels.append(label)
    return Mat,labels

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw3D(dataSet,labels):
    ax=plt.subplot(111,projection='3d')
    for i in range(len(labels)):
        if labels[i]=='1':
            ax.scatter(dataSet[i][0],dataSet[i][1],dataSet[i][2],c='r')
        if labels[i]=='2':
            ax.scatter(dataSet[i][0],dataSet[i][1],dataSet[i][2],c='g')
        if labels[i]=='3':
            ax.scatter(dataSet[i][0],dataSet[i][1],dataSet[i][2],c='b')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

def autonorm(dataSet):
    minValue=dataSet.min(0)
    maxValue=dataSet.max(0)
    m=dataSet.shape[0]
    normDataSet=(dataSet-tile(minValue,[m,1]))/tile(maxValue-minValue,[m,1])
    return normDataSet

import random

def datingClassTest():
    hoRaito=0.1
    dataSet,labels=file2matrix("D:\学习资料\machinelearninginaction\Ch02\datingTestSet2.txt")
    normDataSet=autonorm(dataSet)
    m=normDataSet.shape[0]
    TestNum=int(m*hoRaito)
    count=0
    for i in range(TestNum):
        Result=classify0(normDataSet[i,:],normDataSet[TestNum:m],labels,20)
        print("Result is %s,but the True is %s" % (Result, labels[i]))
        if Result!=labels[i]:
            count+=1
    print("The total error Rate is %f"%(count/TestNum))

# datingClassTest()
# print(file2matrix("D:\学习资料\machinelearninginaction\Ch02\datingTestSet2.txt"))
# dataSet,labels=file2matrix("D:\学习资料\machinelearninginaction\Ch02\datingTestSet2.txt")
# print(autonorm(dataSet))
# draw3D(dataSet,labels)
# print(classify0(array([26052,1.441871,0.805124]),dataSet,labels,20))