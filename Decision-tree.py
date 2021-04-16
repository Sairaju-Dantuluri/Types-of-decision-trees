import numpy as np
import pandas as pd
import random
from featureranker import FeatureRanker

alter = 0
fr = FeatureRanker()


class DecisionTree:
    left = None
    right = None
    splitvalue = 0.0
    f_index = 0
    family = -1


def splitPoint(data, feature_index):
    # print("splitpoint : ", data[:, feature_index])
    return np.median(data[:, feature_index])


def split(data, measure):
    feature_index = (fr.rank_features(data, measure))[0]
    splitpoint = splitPoint(data, feature_index)
    return feature_index, splitpoint


def buildDecisionTree(data, measure, alter):
    # print(data.shape)
    root = DecisionTree()
    av = np.average(data[:, -1])
    # if the data size is 0
    if (np.shape(data))[0] == 0:
        root = None
        return root
    # if the feature set is exhausted
    elif len(data[0]) == 1:
        if av >= 0.5:
            root.family = 1
        else:
            root.family = 0
        return root
    # if the data is correctly classified
    elif av == 0 or av == 1:
        root.family = av
        return root

    root.f_index, root.splitvalue = split(data, measure)
    leftData = None
    rightData = None
    flag1 = 0
    flag2 = 0
    # print("split : "+str(root.splitvalue))
    # print("feature index : ", root.f_index)
    for d in data:

        # print("dval ", d[root.f_index])
        if d[root.f_index] < root.splitvalue:
            if flag1 != 0:
                leftData = np.vstack((leftData, d))

            else:
                leftData = d
                flag1 = 1
        elif d[root.f_index] > root.splitvalue:
            if flag2 != 0:
                rightData = np.vstack((rightData, d))
            else:
                rightData = d
                flag2 = 1
        else:  # divide equal elements equally among left and right using alter flag
            if alter == 0:
                if flag1 != 0:
                    leftData = np.vstack((leftData, d))
                    alter = 1

                else:
                    leftData = d
                    flag1 = 1
                    alter = 1

            else:
                if flag2 != 0:
                    rightData = np.vstack((rightData, d))
                    alter = 0

                else:
                    rightData = d
                    flag2 = 1
                    alter = 0

    leftData = np.atleast_2d(leftData)
    rightData = np.atleast_2d(rightData)

    # print(leftData.shape, rightData.shape)

    leftData = np.delete(leftData, axis=1, obj=root.f_index)
    rightData = np.delete(rightData, axis=1, obj=root.f_index)
    root.left = buildDecisionTree(leftData, measure, alter)
    root.right = buildDecisionTree(rightData, measure, alter)
    return root


def predict(data, root):
    if root.family < 0:
        feat = data[root.f_index]
        data = np.delete(data, axis=0, obj=root.f_index)
        if root.splitvalue > feat:
            return predict(data, root.left)
        else:
            return predict(data, root.right)
    else:
        return root.family


def predictData(data, root):
    a, d = 0, 0
    for row in data:
        if row[-1] == 1 == predict(row, root):
            d += 1
        elif row[-1] == 0 == predict(row, root):
            a += 1
    bc = data.shape[0] - (a+d)
    acc = 0.0
    acc = (a+d)/data.shape[0]
    fsqr = 0.0
    if a != 0:
        fsqr = (2*a/((2*a) + bc))
    return acc, fsqr

# 10 fold cross validation


def CrossValidate(data):
    acclis = [0, 0, 0, 0, 0]
    flis = [0, 0, 0, 0, 0]
    for i in range(10):
        test = None
        train = None
        lenx = int((data.shape[0])/10)

        if (i != 9):
            test = (data[i*lenx:(i+1)*lenx])
        else:
            test = (data[i*lenx:])

        if (i != 0):
            if type(train) != type(None):
                train = np.vstack((train, data[0:i*lenx]))
            else:
                train = (data[0:i*lenx])
        if (i != 9):
            if type(train) != type(None):
                train = np.vstack((train, data[(i+1)*lenx:]))
            else:
                train = data[(i+1)*lenx:]
        # chisquare, ginisplit, gainratio, infogain
        root = buildDecisionTree(train, 'chisquare', 0)
        print("testdata shape : ", test.shape)
        acc = predictData(test, root)
        acclis[0] += acc[0]
        flis[0] += acc[1]

        root = buildDecisionTree(train, 'ginisplit', 0)
        acc = predictData(test, root)
        acclis[1] += acc[0]
        flis[1] += acc[1]

        root = buildDecisionTree(train, 'gainratio', 0)
        acc = predictData(test, root)
        acclis[2] += acc[0]
        flis[2] += acc[1]

        root = buildDecisionTree(train, 'infogain', 0)
        acc = predictData(test, root)
        acclis[3] += acc[0]
        flis[3] += acc[1]

        root = buildDecisionTree(train, 'misclass', 0)
        acc = predictData(test, root)
        acclis[4] += acc[0]
        flis[4] += acc[1]
    return np.divide(acclis, 10), np.divide(flis, 10)


output = "chisquareacc,chisquarefsqr,ginisplitacc,ginisplitfsqr,gainratioacc,gainratiofsqr,infogainacc,infogainfsqr,miscalssacc,miscalssfsqr\n"
for i in range(51, 52):
    filename = "data/"+str(i)+".csv"
    df = pd.read_csv(filename, header=None).sample(frac=1)
    data = np.array(df)
    for row in data:
        if row[-1] > 1:
            row[-1] = 1
    acc, fsqr = CrossValidate(data)
    dataoutput = ''
    for j in range(5):
        dataoutput += (str(acc[j]) + "," + str(fsqr[j])+",")
    dataoutput = dataoutput[:-1]
    dataoutput += "\n"
    output += dataoutput
    resultfile = "results/res"+str(i)+".csv"
    file = open(resultfile, 'a')
    file.write(output)
    print("done : ", i, ".csv")
