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
                value = ((arr[i][j]-arr2[i][j])**2)/arr2[i][j]
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

    def info_gain(self, arr):
        eLeft = 0
        eRight = 0

        for i in range(len(arr)):
            pLeft = arr[0][i]/sum(arr[0])
            eLeft = eLeft - (pLeft)*(math.log2(pLeft))
            pRight = arr[1][i]/sum(arr[1])
            eRight = eRight - (pRight)*(math.log2(pRight))

        total = sum(arr[0]) + sum(arr[1])
        info = eLeft*(sum(arr[0])/total) + eRight*(sum(arr[1])/total)

        return info

    def gain_ratio(self, arr):
        gain = self.info_gain(arr)

        total = sum(arr[0]) + sum(arr[1])

        splitInfo = 0

        for i in range(arr.shape[0]):
            splitInfo = splitInfo + \
                (sum(arr[i])/total)*(math.log2(sum(arr[i])/total))

        gainRatio = gain/splitInfo

        return gainRatio

    def rank_features(self, data, measure):

        no_of_unique = []

        for i in range(data.shape[1]):
            no_of_unique.append(len(pd.unique(data[i])))

        avgs = []

        for i in range(data.shape[1]):
            avgs.append(np.mean(data[i]))

        values = np.zeros(data.shape[-1]-1)

        for i in range(data.shape[-1]-1):
            arr = np.zeros((2, 2))
            for j in range(len(data[i])):

                if data[i][j] >= avgs[i]:
                    if(data[-1][j] == 0):
                        arr[0][1] = arr[0][1] + 1
                    else:
                        arr[0][0] = arr[0][0] + 1
                else:
                    if(data[-1][j] == 0):
                        arr[1][1] = arr[1][1] + 1
                    else:
                        arr[1][0] = arr[1][0] + 1

            # Here,replce gini_split with the required function
            if measure == 'ginisplit':
                values[i] = self.gini_split(arr)

            if measure == 'gainratio':
                values[i] = self.gain_ratio(arr)

            if measure == 'chisquare':
                values[i] = self.chi_square(arr)

            if measure == 'informationgain':
                values[i] = self.info_gain(arr)

        if measure == 'ginisplit':
            values = -values
        rankwise_indices = (-values).argsort()[:len(values)]

        return rankwise_indices
