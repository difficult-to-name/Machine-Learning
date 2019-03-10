import json, operator
import treePlotter
from math import log


def clacShannonEnt(dataSet):
    """
    计算给定数据集的信息熵
    :param dataSet:
    :return:
    """
    # 实例总数
    numEntries = len(dataSet)
    labelCounts = {}
    # 记录每个类别的出现次数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算信息熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    :param dataSet:
    :param axis:划分数据集的特征索引
    :param value:需要返回的特征值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        # 根据特征划分，并剔除该特征
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好数据集划分方式
    :param dataSet:
    :return: 最好的划分特征
    """
    # 统计数据集特征个数，计算原始熵
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = clacShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1

    for i in range(numFeatures):
        # 取数据集中每个实例的第i个特征值
        featList = [example[i] for example in dataSet]
        # 去重
        uniqueVals = set(featList)
        # 设置新熵
        newEntropy = 0.0
        # 按照第i个特征划分数据集后的熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * clacShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 取信息增益最高的特征值
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    当遍历完所有特征后类标签并不唯一，则取最多数
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    构建决策树
    :param dataSet:
    :param labels:
    :return: 树结构
    """
    # 获取数据集中标签列表
    classList = [example[-1] for example in dataSet]

    """以下代码在递归中被调用"""
    if classList.count(classList[0]) == len(classList):     # 当前数据集只有一个分类，返回该标签
        return classList[0]
    if len(dataSet[0]) == 1:        # 当前数据集只剩一个特征，返回次数最多的分类标签
        return majorityCnt(classList)
    """以上代码在递归中被调用"""

    # 选取最佳特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #del (labels[bestFeat])

    # 获取最佳特征值列表
    featVaules = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featVaules)

    # 构造树
    for value in uniqueVals:
        subLabels = labels[0:bestFeat] + labels[bestFeat+1:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    使用决策树执行分类
    :param inputTree:决策树
    :param featLabels:标签列表
    :param testVec:测试数据
    :return:分类
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 确定特征在数据集中的位置
    featIndex = featLabels.index(firstStr)
    # 遍历树寻找相同特征值的分类
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            # 若分类是字典（判断节点）则调用递归
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel


def storeTree(inputTree, filename):
    """
    存储决策树
    """
    with open(filename, 'w') as f:
        f.write(json.dumps(inputTree, indent=3))

def grabTree(filename):
    """
    读取决策树
    """
    with open(filename, 'r') as f:
        return  str(json.loads(f.read()))

def dataDispose(filename):
    """
    数据处理
    """
    with open(filename, 'r') as f:
        dataSet = f.readlines()
    dataSet = [inst.strip().split('\t') for inst in dataSet]
    lenseLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenseTree = createTree(dataSet, lenseLabels)
    return  lenseTree


myTree = dataDispose('lenses.txt')
storeTree(myTree, 'classifierStorage.json')
treePlotter.createPlot(myTree)