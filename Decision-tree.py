import numpy as np
import pandas as pd
import random
from featureranker import FeatureRanker

fr = FeatureRanker()


def splitPoint(data, feature_index):
    return np.median(data[:][feature_index])


def split(data, measure):
    feature_index = (fr.rank_features(data, measure))[0]
    splitpoint = splitPoint(data, feature_index)
    return feature_index, splitpoint


def buildDecisionTree(data, measure):
    root = DecisionTree()
    av = np.average(data[:][-1])
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
    leftData = np.array(0)
    rightData = np.array(0)
    for d in data:
        if d[root.f_index] <= root.splitvalue:
            leftData = np.vstack(leftData, d)
        else:
            rightData = np.vstack(rightData, d)
    leftData = np.delete(leftData, axis=1, obj=root.f_index)
    rightData = np.delete(rightData, axis=1, obj=root.f_index)
    root.left = buildDecisionTree(leftData, measure)
    root.right = buildDecisionTree(rightData, measure)
    return root


class DecisionTree:
    left = DecisionTree()
    right = DecisionTree()
    splitvalue = 0.0
    f_index = 0
    family = -1
