import random
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    f=open(filename)
    dataMat=[]
    labelMat=[]
    for line in f.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

def smosimple(dataMat,labelMat,C,SlackVar,maxIter):
    dataMatrix=mat(dataMat)  # m*n
    labelMatrix=mat(labelMat).T # m*1
    m,n=shape(dataMatrix)
    alpha=zeros((m,1))  #m*1
    b=0
    iter=0
    while(iter<maxIter):
        alphaPairschange=0
        for i in range(m):
            # temp1=multiply(alpha, labelMatrix).T
            # temp2=dataMatrix*dataMatrix[i,:].T
            # ui=temp1*temp2+b
            ui=multiply(alpha,labelMatrix).T*(dataMatrix*dataMatrix[i,:].T)+b
            Ei=ui-labelMatrix[i]
        # 判断 alpha需要调整
            if labelMatrix[i]*ui<=SlackVar and alpha[i]<C or labelMatrix[i]*ui>=SlackVar and alpha[i]>0:
                j=selectJrand(i,m)
                uj=multiply(alpha,labelMatrix).T*(dataMatrix*dataMatrix[j,:].T)+b
                Ej=uj-labelMatrix[j]
                alphaIold=alpha[i].copy()
                alphaJold=alpha[j].copy()
                if labelMatrix[i]!=labelMatrix[j]:
                    L=max(0,alphaJold-alphaIold)  # alpha 下限 与 上限
                    H=min(C,C+alphaJold-alphaIold)
                else:
                    L=max(0,alphaIold+alphaJold-C)
                    H=min(C,alphaJold+alphaIold)
                if L==H:
                    print("L=H \n");continue
                eta=2*(dataMatrix[i,:]*dataMatrix[j,:].T)-(dataMatrix[i,:]*dataMatrix[i,:].T)-(dataMatrix[j,:]*dataMatrix[j,:].T)
                if eta>=0:
                    print("eta>=0");continue
                alphaJnew=alphaJold-labelMatrix[j]*(Ei-Ej)/eta
                alphaJnew=clipAlpha(alphaJnew,H,L)
                if abs(alphaJnew-alphaJold)<0.00001:
                    print("alpha J is not moving");continue
                alphaInew=alphaIold+labelMatrix[i]*labelMatrix[j]*(alphaJold-alphaJnew)

                alpha[i] = alphaInew
                alpha[j] = alphaJnew

                bi=b-Ei-labelMatrix[i]*(alphaInew-alphaIold)*(dataMatrix[i,:]*dataMatrix[i,:].T)-labelMatrix[j]*(alphaJnew-alphaJold)*(dataMatrix[j,:]*dataMatrix[j,:].T)
                bj=b-Ej-labelMatrix[i]*(alphaInew-alphaIold)*(dataMatrix[i,:]*dataMatrix[i,:].T)-labelMatrix[j]*(alphaJnew-alphaJold)*(dataMatrix[j,:]*dataMatrix[j,:].T)
                if alphaInew>=0 and alphaInew<=C:b=bi
                elif alphaJnew>=0 and alphaJnew<=C:b=bj
                else: b=(bi+bj)/2
                alphaPairschange+=1
                print("iter %d, i %d, alphapairs have changed %d"%(iter,i,alphaPairschange))
        if alphaPairschange==0:
            iter+=1
        else:iter=0
        print("iteration number:%d"%iter)
    w = multiply(alpha,labelMatrix).T * dataMatrix
    return alpha,b,w

def draw(dataSet,labels,alpha,w,b):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i in range(len(labels)):
        if labels[i]==1:
            if alpha[i]>0:
                ax.scatter(dataSet[i][0],dataSet[i][1],c='r',marker='^')
            else:
                ax.scatter(dataSet[i][0],dataSet[i][1],c='r',marker='.')
        else:
            if alpha[i]>0:
                ax.scatter(dataSet[i][0],dataSet[i][1],c='b',marker='^')
            else:
                ax.scatter(dataSet[i][0],dataSet[i][1],c='b',marker='.')
    x=arange(0,10,1)
    w1=w[0,0]
    w2=w[0,1]
    y=(-float(b)-w1*x)/w2
    plt.plot(x,y)
    plt.show()

# dataMat,labelMat=loadDataSet("D:\学习资料\machinelearninginaction\Ch06\\testSet.txt")
# alpha,b,w=smosimple(dataMat,labelMat,0.6,0.001,40)
# draw(dataMat,labelMat,alpha,w,b)





