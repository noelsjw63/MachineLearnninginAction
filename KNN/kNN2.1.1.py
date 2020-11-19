from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def createDataSet():
    """

    :return:
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
    pass


def classify0(inX, dataSet, labels, k):
    '''

    :param inX: input vector
    :param dataSet:
    :param labels:
    :param k:
    :return:
    '''
    dataSetSize = dataSet.shape[0]  # the shape return the number of examples
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # tile(A, rep): Construct an array by repeating A the number of times given by reps.

    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndics = distances.argsort()

    # argsort : return the index after sorted
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndics[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # get(): find voteIlable, if not find then return 0

    sortedClassacount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # items: 以列表的形式返回字典中的key-value对，储存在元组中、
    # itemgetter(1) 以classCount 中的第二个域进行排序

    return sortedClassacount[0][0]

def file2Matrix(fname):
    fr = open(fname)
    arrayOLines = fr.readlines()
    numofLines = len(arrayOLines)
    returnMat = zeros((numofLines, 3)) # 为什么是3行呢
    label = []
    index = 0
    for line in arrayOLines:
        line = line.strip() # 删除首位空白字符
        splitedLine = line.split('\t')
        returnMat[index, :] = splitedLine[0:3] # [m:n] 取m至n-1
        label.append(int(splitedLine[-1]))
        index += 1
    return returnMat, label

def img2vector(filename):
    returnVec = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i + j] = int(lineStr[j])
    return returnVec


def digitsClassifier():
    label = []
    trainingFileList = listdir('digits/trainingDigits')
    trainSize = len(trainingFileList)
    trainingMat = zeros((trainSize, 1024))
    for i in range(trainSize):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNameStr = int(fileStr.split('_')[0])
        label.append(classNameStr)
        trainingMat[i, :]  = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    testSize = len(testFileList)
    for i in range(testSize):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNameStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifyResult = classify0(vectorUnderTest, trainingMat, label, 3
                                   )
        print("The classifier came back with : %d, the real answer iss :%d" %(classifyResult, classNameStr))
        if (classifyResult != classNameStr):
            errorCount += 1.0

    print("\nThe total number of error is %d" % errorCount)
    print("\nThe total error rate is: %f" % (errorCount/float(testSize)))









if __name__ == '__main__':
    # group, label = createDataSet()

    # Section 2.1
    # print(classify0([0, 0], group, label, 3))

    # Section 2.2
    # datingData, datingLabel = file2Matrix('datingTestSet2.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingData[:, 1], datingData[:, 2], 15.0*array(datingLabel), 15.0*array(datingLabel)) # 第二列第三列
    # plt.show()

    # Section 2.3 Digit classifier
    test = img2vector('digits/trainingDigits/0_0.txt')
    digitsClassifier()
