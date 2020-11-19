from math import log



def calEntropy(dataset):
    numEntries = len(dataset)
    labelCount = {}
    for features in dataset:
        if features[-1] not in labelCount.keys():
            labelCount[features[-1]] = 0
        labelCount[features[-1]] += 1
    entropy = 0
    for key, freq in labelCount.items():
        p_k = float(freq) / numEntries
        entropy -= p_k * log(p_k,2)
    return entropy

def splitDataSet(dataset, axis, )

if __name__ == '__main__':
    # data = [[1,1,'yes'], [1,0,'no'], [0,1,'no'], [0,1, 'no']]
    # print( calEntropy(data))