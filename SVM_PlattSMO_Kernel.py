from numpy import *
import matplotlib.pyplot as plt

def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('The kernel is not recongnized')
    return K

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alpha = zeros((self.m, 1))
        self.b = 0
        self.eCache = zeros((self.m, 2))
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def calcEk(oS,k):
    uk = multiply(oS.alpha, oS.labelMat).T * oS.K[:,k] + oS.b
    Ek = uk - oS.labelMat[k]
    return Ek

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def selectJ(oS,Ei,i):
    maxK=-1;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]
    vaildEchacheList=nonzero(array(oS.eCache[:,0]))[0]
    if len(vaildEchacheList)>1:
        for k in vaildEchacheList:
            if k==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if deltaE>maxDeltaE:
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
        return j,Ej

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

def innerL(oS,i):
    Ei=calcEk(oS,i)
    if oS.labelMat[i] * Ei <= -oS.tol and oS.alpha[i] < oS.C or oS.labelMat[i] * Ei >= oS.tol and oS.alpha[i] > 0:
        j,Ej = selectJ(oS,Ei,i)
        alphaIold = oS.alpha[i].copy()
        alphaJold = oS.alpha[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, alphaJold - alphaIold)  # alpha 下限 与 上限
            H = min(oS.C, oS.C + alphaJold - alphaIold)
        else:
            L = max(0, alphaIold + alphaJold - oS.C)
            H = min(oS.C, alphaJold + alphaIold)
        if L == H:
            print("L=H \n")
            return 0
        eta = 2 * oS.K[i,j] - oS.K[i,i]-oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        alphaJnew = alphaJold - oS.labelMat[j] * (Ei - Ej) / eta
        alphaJnew = clipAlpha(alphaJnew, H, L)
        if abs(alphaJnew - alphaJold) < 0.00001:
            print("alpha J is not moving")
            return 0
        alphaInew = alphaIold + oS.labelMat[i] * oS.labelMat[j] * (alphaJold - alphaJnew)

        oS.alpha[i] = alphaInew
        oS.alpha[j] = alphaJnew

        bi = oS.b - Ei - oS.labelMat[i] * (alphaInew - alphaIold) * oS.K[i,i] - oS.labelMat[
            j] * (alphaJnew - alphaJold) * oS.K[i,j]
        bj = oS.b - Ej - oS.labelMat[i] * (alphaInew - alphaIold) * oS.K[i,j] - oS.labelMat[
            j] * (alphaJnew - alphaJold) * oS.K[j,j]
        if alphaInew >= 0 and alphaInew <= oS.C:
            oS.b = bi
        elif alphaJnew >= 0 and alphaJnew <= oS.C:
            oS.b = bj
        else:
            oS.b = (bi + bj) / 2
        return 1
    else:
        return 0

def loadDataSet(filename):
    f=open(filename)
    dataMat=[]
    labelMat=[]
    for line in f.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def smoP(dataMatIn,classLabels,toler,C,maxIter,kTup):
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter=0
    entireSet=True;alphaPairsChanged=0
    while (iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(oS,i)
                print("fullSet,iter:%d,i:%d,pairschanged:%d"%(iter,i,alphaPairsChanged))
            iter+=1
        else:
            nonBoundIs=nonzero((array(oS.alpha)>0)*(array(oS.alpha)<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(oS,i)
                print("non-Bound,iter:%d,i:%d,pairschanged:%d"%(iter,i,alphaPairsChanged))
            iter+=1
        if entireSet:
            entireSet=False
        elif (alphaPairsChanged==0):
            entireSet=True
        print('iteration number:%d'%iter)
    return oS.b,oS.alpha

def draw(dataSet,labels,alpha):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i in range(len(labels)):
        if labels[i]==1:
            if alpha[i]>0:
                ax.scatter(dataSet[i,0],dataSet[i,1],c='r',marker='^')
            else:
                ax.scatter(dataSet[i,0],dataSet[i,1],c='r',marker='.')
        else:
            if alpha[i]>0:
                ax.scatter(dataSet[i,0],dataSet[i,1],c='b',marker='^')
            else:
                ax.scatter(dataSet[i,0],dataSet[i,1],c='b',marker='.')
    # x=arange(0,10,1)
    # w1=w[0,0]
    # w2=w[0,1]
    # y=(-float(b)-w1*x)/w2
    # plt.plot(x,y)
    plt.show()

def testRbf(k1=1.3):
    dataArr,labelArr=loadDataSet('D:\学习资料\machinelearninginaction\Ch06\\testSetRBF.txt')
    b,alphas=smoP(dataArr,labelArr,toler=0.001,C=20,maxIter=40,kTup=('rbf',k1))
    dataMat=mat(dataArr);labelMat=mat(labelArr).T
    draw(dataMat,labelMat,alphas)
    m,n=shape(dataMat)
    errorcount=0
    for i in range(m):
        trainK=kernelTrans(dataMat,dataMat[i,:],('rbf',k1))
        ui=multiply(alphas,labelMat).T * trainK + b
        if sign(ui)!=sign(labelMat[i]):
            errorcount+=1
    print('trainSetRBF ErrorRate is %f'%(errorcount/m))
    dataArr2,labelArr2=loadDataSet('D:\学习资料\machinelearninginaction\Ch06\\testSetRBF2.txt')
    errorcount2=0
    dataMat2=mat(dataArr2);labelMat2=mat(labelArr2).T
    draw(dataMat2,labelMat2,alphas)
    m2=shape(dataMat2)[0]
    for i in range(m2):
        trainK = kernelTrans(dataMat, dataMat[i, :], ('rbf', k1))
        ui = multiply(alphas, labelMat2).T * trainK + b
        if sign(ui) != sign(labelMat[i]):
            errorcount2 += 1
    print('trainSetRBF2 ErrorRate is %f' % (errorcount2 / m))


# testRbf()