"""
Program to rank the features using chi-squre values
"""
    
import numpy as np
import pandas as pd

def chi_square(arr):
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



def rank_features(data):
    
    no_of_unique = []
    
    for i in range(data.shape[1]):
        no_of_unique.append(len(pd.unique(data[i])))
        
    avgs = []
        
    for i in range(data.shape[1]):
        avgs.append(np.mean(data[i]))
        
    
    arr = np.zeros((2,no_of_unique[-1]))
    
    chi_values = np.zeros(data.shape[-1]-1)
    
    for i in range(data.shape[-1]-1):
        for j in range(len(data[i])):
            if data[i][j] >= avgs[i]:
                arr[0][data[20][j]] = arr[0][data[20][j]] + 1
            else:
                arr[1][data[20][j]] = arr[1][data[20][j]] + 1           
        chi_values[i] = chi_square(arr)
        
    rankwise_indices = (-chi_values).argsort()[:len(chi_values)]
    
    return rankwise_indices

data = pd.read_csv("1.csv",header= None)
print(rank_features(data))
               