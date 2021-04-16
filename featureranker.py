import numpy as np
import pandas as pd
import math


class FeatureRanker:

    def chi_square(self, arr):
        shape = arr.shape

        vlist = []
        hlist = []
        for i in range(shape[0]):
            sum = 0
            for j in range(shape[1]):
                sum = sum + arr[i][j]
            vlist.append(sum)

        for i in range(shape[1]):
            sum = 0
            for j in range(shape[0]):
                sum = sum + arr[j][i]
            hlist.append(sum)

        total = 0
        for i in range(len(hlist)):
            total = total+hlist[i]

        arr2 = np.zeros(shape)
        for i in range(len(hlist)):
            for j in range(len(vlist)):
                value = hlist[i]*vlist[j]
                value = value/total
                arr2[j][i] = value

        for i in range(shape[0]):
            for j in range(shape[1]):
                if(arr2[i][j] != 0):
                    value = ((arr[i][j]-arr2[i][j])**2)/arr2[i][j]
                else:
                    value = 0
                arr[i][j] = value

        sum = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                sum = sum + arr[i][j]

        return sum

    def gini_split(self, arr):

        g1 = 0
        g2 = 0

        for i in range(len(arr)):
            if sum(arr[0]) != 0:
                g1 = g1 + (arr[0][i]/sum(arr[0]))**2
            if sum(arr[1]) != 0:
                g2 = g2 + (arr[1][i]/sum(arr[1]))**2

        g1 = 1 - g1
        g2 = 1 - g2

        total = sum(arr[0]) + sum(arr[1])

        gini = g1*(sum(arr[0])/total) + g2*(sum(arr[1])/total)

        return gini

    def entropy(self, arr):
        total = 0
        if(sum(arr) == 0):
            return 0
        for p in arr:
            p = p / sum(arr)
            if p != 0:
                total += p * math.log2(p)

        total *= -1
        return total

    def info_gain(self, arr):

        parent_node = np.zeros(arr.shape[-1])

        parent_node = arr.sum(axis=0)
        parent_entropy = self.entropy(parent_node)

        total = 0
        for i in range(arr.shape[0]):
            total += sum(arr[i])

        Sum = 0
        for a in arr:
            Sum += (sum(a)/total) * self.entropy(a)

        info = parent_entropy - Sum

        return info

    def gain_ratio(self, arr):

        gain = self.info_gain(arr)

        total = 0
        for i in range(arr.shape[0]):
            total += sum(arr[i])

        splitInfo = 0

        for i in range(arr.shape[0]):
            if(sum(arr[i]) != 0):
                p = sum(arr[i])/total
                splitInfo = splitInfo + (p*(math.log2(p)))

        splitInfo *= -1
        if(splitInfo != 0):
            gainRatio = gain/splitInfo
        else:
            gainRatio = 0

        return gainRatio

    def misclass_error(self, arr):
        error = 0
        total = 0
        pmax = 0
        for i in range(arr.shape[0]):
            total += sum(arr[i])

        for i in range(arr.shape[0]):
            if(sum(arr[i]) != 0):
                for j in range(arr.shape[1]):
                    p = arr[i][j]/sum(arr[i])
                    if(p > pmax):
                        pmax = p

            node_p = sum(arr[i])/total
            error += node_p*(1 - pmax)

        return error

    def rank_features(self, data, measure):
        # print(data.shape)
        avgs = np.mean(data, axis=0)

        values = np.zeros(data.shape[-1]-1)

        for i in range(data.shape[-1]-1):
            arr = np.zeros((2, 2))
            for j in range(data.shape[0]):

                if data[j][i] >= avgs[i]:
                    if(data[j][-1] == 0):
                        arr[0][1] += 1
                    else:
                        arr[0][0] += 1
                else:
                    if(data[j][-1] == 0):
                        arr[1][1] += 1
                    else:
                        arr[1][0] += 1

            # Here,replce gini_split with the required function
            if measure == 'ginisplit':
                values[i] = self.gini_split(arr)

            if measure == 'gainratio':
                values[i] = self.gain_ratio(arr)

            if measure == 'chisquare':
                values[i] = self.chi_square(arr)

            if measure == 'infogain':
                values[i] = self.info_gain(arr)

            if measure == 'misclass':
                values[i] = self.misclass_error(arr)

        if measure == 'ginisplit':
            values = -values
        rankwise_indices = (-values).argsort()[:len(values)]

        return rankwise_indices
