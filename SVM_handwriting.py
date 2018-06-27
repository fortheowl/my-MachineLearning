from SVM_PlattSMO_Kernel import *
import os

def img2vector(filename):
    f=open(filename)
    NewVect=zeros((1,1024))
    for i in range(32):
        lineStr=f.readline()
        for j in range(32):
            NewVect[0,32*i+j]=int(lineStr[j])
    return NewVect

def handwritingClassTest():
    filenameList=os.listdir("D:\学习资料\machinelearninginaction\Ch02\\trainingDigits")
    TrainDataSet=zeros((len(filenameList),1024))
    index=0
    Trainlabels=[]
    for filename in filenameList:
        TrainDataSet[index,:]=img2vector("D:\学习资料\machinelearninginaction\Ch02\\trainingDigits\\"+filename)
        Trainlabels.append(int(filename[0]))
        index+=1
    TrainMat=mat(TrainDataSet);TrainLabelMat=mat(Trainlabels)
    b,alphas=smoP(TrainDataSet,Trainlabels,toler=0.001,C=20,maxIter=40,kTup=('rbf',1.3))
    w = multiply(alphas,TrainLabelMat.T).T * TrainMat
    filenameList2=os.listdir("D:\学习资料\machinelearninginaction\Ch02\\testDigits")
    TestDataSet=zeros((len(filenameList2),1024))
    index2=0
    # Testlabels=[]
    count=0
    for filename2 in filenameList2:
        TestDataSet[index2,:]=img2vector("D:\学习资料\machinelearninginaction\Ch02\\testDigits\\"+filename2)
        # Testlabels.append(int(filename2[0]))
        # index2+=1
        # Testlabel=KNN.classify0(TestDataSet[index2,:],TrainDataSet,Trainlabels,3)
        Testlabels=int(mat(TestDataSet[index2,:])*w.T)
        Reallabel=int(filename2[0])
        print("The classifier came back with: %d,The real answer is %d"%(Testlabels,Reallabel))
        if Testlabels!=Reallabel:
            count+=1
        index2+=1
    print("The total error rate is %f"%(count/len(filenameList2)))

handwritingClassTest()

