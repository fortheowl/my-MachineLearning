from math import log
import operator

def calcShannonEnt(dataSet):
    L=len(dataSet)
    dic={}
    for featvec in dataSet:
        if featvec[-1] not in dic.keys():
            dic[featvec[-1]]=0
        dic[featvec[-1]]+=1
    Ent=0
    for key in dic.keys():
        prob=dic[key]/L
        Ent-=prob*log(prob,2)
    return Ent

def createData():
    myDat=[[1,1,'yes'],[1,1,'yes'],[0,1,'no'],[1,0,'no'],[0,0,'no']]
    labels=['no surfacing','flippers']
    return myDat,labels

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featvec in dataSet:
        if featvec[axis]==value:
            reducedFeatVec=featvec[:axis]
            reducedFeatVec.extend(featvec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    L=len(dataSet[0])-1
    baseEnt=calcShannonEnt(dataSet)
    bestGain=0;bestFeature=-1
    for i in range(L):
        featVec=[example[i] for example in dataSet]
        featValue=set(featVec)
        NewEnt=0
        for value in featValue:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/len(dataSet)
            NewEnt+=prob*calcShannonEnt(subDataSet)
        Gain=baseEnt-NewEnt
        if Gain>bestGain:
            bestGain=Gain
            bestFeature=i
    return bestFeature

def majorityCnt(classlist): #如果已处理完所有特征，仍然有叶节点类标签非唯一，按多数的标签决定该叶节点标签类
    classCount={}
    for vote in classlist:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataset,label):
    classlist=[example[-1] for example in dataset]
    # if len(set(classlist))==1:
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]
    if len(dataset[0])==1:
        return majorityCnt(classlist)
    bestFeat=chooseBestFeatureToSplit(dataset)
    bestlabel=label[bestFeat]
    del(label[bestFeat])
    myTree={bestlabel:{}}
    value=[example[bestFeat] for example in dataset]
    for i in set(value):
        sublabel=label[:]
        subDataSet=splitDataSet(dataset,bestFeat,i)
        myTree[bestlabel][i]=createTree(subDataSet,sublabel)
    return myTree

def getNumLeafs(myTree):
    numLeafs=0
    firstLeaf=list(myTree.keys())[0]
    secondDict=myTree[firstLeaf]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(myTree):
    maxdepth=0
    firstLeaf=list(myTree.keys())[0]
    secondDict=myTree[firstLeaf]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            maxdepth=1+getTreeDepth(secondDict[key])
        else:
            maxdepth=1
    return maxdepth

def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if key==testVec[featIndex]:
            if type(secondDict[key]).__name__=='dict':
                classifylabel=classify(secondDict[key],featLabels,testVec)
            else:
                classifylabel=secondDict[key]
        else:
            continue
    return classifylabel

# labels=['age','prescript','astigmatic','tearRate']
# fr=open("D:\学习资料\machinelearninginaction\Ch03\lenses.txt")
# myDat=[line.strip().split('\t') for line in fr.readlines()]
# myTree=createTree(myDat,labels)
# print(myTree)
# classlabel=classify(myTree,['age','prescript','astigmatic','tearRate'],['pre','myope','yes','normal'])
# print(classlabel)

#pickle dump() load() Python格式序列化存储和读取
def storeTree(inputTree,filename):
     import pickle
     fw=open(filename)
     pickle.dump(inputTree,fw)
     fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)

