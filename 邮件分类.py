import re, feedparser
from numpy import *
from math import log


def createVocaList(dataSet):
    # 创建空集
    vocabSet = set([])
    for document in dataSet:
        # 求两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    检查单词是否出现在词汇表中
    :param vocabList: 词汇表
    :param inputSet: 单词列表
    :return: 词向量，1表示出现
    """
    #  创建一个所含元素都是0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 训练文档矩阵
    :param trainCategory:类别标签向量
    :return:
    """
    # 训练文档数量
    numTrainDocs = len(trainCategory)
    # 词汇表所含词汇数量
    numWords = len(trainMatrix[0])
    # 任意文档属于类别1的概率
    proabusive = sum(trainCategory)/float(numTrainDocs)
    # 初始化
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    # 统计已知类别文档中各词出现的次数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 计算条件概率:P(某词出现在这类别的次数|已知类别)
    p1Vect = list(map(log,p1Num/p1Denom))
    p0Vect = list(map(log,p0Num/p0Denom))
    return p0Vect, p1Vect, proabusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    :param vec2Classify:要分类的向量
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return: 类别
    """
    # 根据贝叶斯公式P(C|W)=[P(W|C)P(C)]/P(W)， 这里P(W)=1,即文档已确定
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
    测试分类器(封装好所有函数)
    :return:
    """
    listOPosts, listClasses = loadDataSet()
    myvocabList = createVocaList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myvocabList, postinDoc))
    p0V, p1V, pab = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myvocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pab))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myvocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pab))


def textParse(bigString):
    listOfTokens = re.split('\W',bigString)
    listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 2]
    return listOfTokens


def spamTest():
    """
    朴素贝叶斯分类器过滤邮件
    :return:
    """
    docList = [];classList = [];fullText = []
    # 导入并解析文件
    for i in range(1,26):
        wordList = textParse(open('spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocaList(docList)
    # 创建空的训练集和测试集
    trainingSet = list(range(50));testSet= []
    # 随机选取10个邮件作测试集
    for i in range(10):
        rangIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[rangIndex])
        del(trainingSet[rangIndex])
    # 计算两类邮件词汇的条件概率
    trainMat =[];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 对测试集进行分类，并统计错误率
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: ", float(errorCount/len(testSet)))


spamTest()

