import matplotlib.pyplot as plt
import DecisionTree

decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    plt.annotate(nodeTxt,xy=parentPt,xycoords="axes fraction",xytext=centerPt,
                 textcoords="axes fraction",va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(DecisionTree.getNumLeafs(inTree))
    plotTree.totalD=float(DecisionTree.getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff=-1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2+cntrPt[1]
    plt.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=DecisionTree.getNumLeafs(myTree)
    depth=DecisionTree.getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.xOff+(1+float(numLeafs))/2/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1/plotTree.totalD

fr=open("D:\学习资料\machinelearninginaction\Ch03\lenses.txt")
myDat=[line.strip().split('\t') for line in fr.readlines()]
labels=['age','prescript','astigmatic','tearRate']
myTree=DecisionTree.createTree(myDat,labels)
createPlot(myTree)