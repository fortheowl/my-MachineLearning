import numpy

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    retVec=[]
    for word in dataSet:
        retVec.extend(set(word))
    finalList=sorted(set(retVec),key=lambda x:x[0])
    return finalList

def setOfWords2Vec(vocablist,inputSet):  #词集模型
    vec=[0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            vec[vocablist.index(word)]=1
        else:
            print("this word (%s) is not in vocabulary"%word)
    return vec

def bagOfWords2Vec(vocablist,inputSet):  #词袋模型
    vec=[0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            vec[vocablist.index(word)]+=1
        else:
            print("this word (%s) is not in vocabulary" % word)
    return vec

def trainNB0(trainMatrix,trainCategory):
    numTrainMat=len(trainMatrix)
    numTrainCat=len(trainCategory)
    p0Vect=numpy.zeros(len(trainMatrix[0]))
    #p0Vect=numpy.ones(len(trainMatrix[0]))
    p1Vect=numpy.zeros(len(trainMatrix[0]))
    #p1Vect=numpy.ones(len(trainMatrix[0]))
    p0Sum=0
    #p0Sum=2
    p1Sum=0
    #p1Sum=2
    pAbusive=sum(trainCategory)/numTrainCat # 类别为1的概率
    for i in range(numTrainMat):
        if trainCategory[i]==1:
            p1Vect+=trainMatrix[i]
            p1Sum+=sum(trainMatrix[i])
            # p1Sum+=1
        else:
            p0Vect+=trainMatrix[i]
            p0Sum+=sum(trainMatrix[i])
            # p0Sum+=1
    p1Vect=p1Vect/p1Sum
    #p1Vect=log(p1Vect/p1Sum)
    p0Vect=p0Vect/p0Sum
    #p0Vect=log(p0Vect/p0Sum)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec,p0Vect,p1Vect,pClass1):
    p1=sum(p1Vect*vec)*pClass1
    p0=sum(p0Vect*vec)*(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

# postingList,classVec=loadDataSet()
# VocabList=createVocabList(postingList)
# # Vec0=setOfWords2Vec(VocabList,postingList[0])
# # print(Vec0)
# # print(VocabList)
# trainMat=[]
# for i in range(len(postingList)):
#     trainMat.append(bagOfWords2Vec(VocabList,postingList[i]))
# p0Vect,p1Vect,pClass1=trainNB0(trainMat,classVec)
# print(p0Vect,'\n',p1Vect,'\n',pClass1)
# Test1=['fuck','garbage','stupid','dog','my']
# Test2=['love','my','dalmation']
# Test1Vec=setOfWords2Vec(VocabList,Test1)
# Test2Vec=setOfWords2Vec(VocabList,Test2)
# Result=classifyNB(Test1Vec,p0Vect,p1Vect,pClass1)
# Result2=classifyNB(Test2Vec,p0Vect,p1Vect,pClass1)
# print(Result,Result2)