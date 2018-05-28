import os
import re
import Bayes
import random

regEx=re.compile('\\W*')

def loadDataSet(dirname):
    filenameList=os.listdir(dirname)
    EmailDataSet=[]
    for filename in filenameList:
        f=open(dirname+"\\"+filename)
        Email=[tok.lower() for tok in regEx.split(f.read()) if len(tok)>2 ]
        EmailDataSet.append(Email)
    return EmailDataSet

# EmailDataSet=loadDataSet("D:\学习资料\machinelearninginaction\Ch04\email\ham")
# print(EmailDataSet)

def spamTest():
    hamemail=loadDataSet("D:\学习资料\machinelearninginaction\Ch04\email\ham")
    hamclassList=[0]*len(hamemail)
    spamemail=loadDataSet("D:\学习资料\machinelearninginaction\Ch04\email\spam")
    spamclassList=[1]*len(spamemail)
    Allemail=[]
    Allemail.extend(hamemail);Allemail.extend(spamemail)
    AllList=[]
    AllList.extend(hamclassList);AllList.extend(spamclassList)
    VocalbList=Bayes.createVocabList(Allemail)
    # print(VocalbList)
    testMat=[]
    realclass=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(Allemail)))
        testMat.append(Bayes.bagOfWords2Vec(VocalbList,Allemail[randIndex]))
        del(Allemail[randIndex])
        realclass.append(AllList[randIndex])
        del(AllList[randIndex])
    trainMat=[]
    for i in range(len(Allemail)):
        trainMat.append(Bayes.bagOfWords2Vec(VocalbList,Allemail[i]))
    p0vect,p1vect,pA=Bayes.trainNB0(trainMat,AllList)
    # print(p0vect,'\n',p1vect,'\n',pA)
    for i in range(10):
        print("test_result=",Bayes.classifyNB(testMat[i],p0vect,p1vect,pA),",real_result=",realclass[i])

# spamTest()





