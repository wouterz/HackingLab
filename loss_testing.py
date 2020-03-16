import numpy as np
import json
import os

import matplotlib.pyplot as plt
import glob
from math import log, inf
import itertools
import pickle
from datetime import datetime

flatten = lambda l: [item for sublist in l for item in sublist]
value_dict = dict()
key_dict = dict()

def results_to_matrix(results: {str:[]}) -> [[]]:
    keys = results.keys()
    for key in keys:
        if(key not in key_dict.keys()):
            key_dict[key]=len(key_dict.keys())
    values = set(flatten(results.values()))
    for val in values:
        if(val not in value_dict.keys()):
            value_dict[val]=len(value_dict.keys())
 #   if(isinstance(matrix, list)):
    matrix = np.zeros((len(keys), len(value_dict.keys())))
    for i in results.keys():
        for j in results[i]:
            matrix[key_dict[i]][value_dict[j]]+=1
    #for value in results.values():
    #    el=[]
    #    for c in classes: el.append(value.count(c))
    #    matrix.append(el)
    return matrix

def compute_matrix_loss(matrix: [[]]) -> float:
    matrix = matrix/matrix.sum(axis=1)[:,None]
    usedLabels = []
    loss=0
    matrix = matrix.transpose()
    for i in matrix:
        highest_probability = 0
        probability_id = inf
        for j in range(0,len(i)):
            if(j not in usedLabels):
                if(i[j]>=highest_probability):
                    highest_probability=i[j]
                    probability_id=j
        usedLabels.append(probability_id)
        if(highest_probability==0):
            loss = loss+10
        else:
            loss = loss - log(highest_probability)
    loss = loss+(len(matrix[1])-len(matrix))*10
    return loss

# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*log(q[i]) for i in range(len(p))])
 


if __name__ == "__main__":

    x = {'1': ['modem', 'modem', 'modem', 'modem', 'screen', 'puck', 'puck', 'puck', 'puck', 'modem'], '2': ['screen', 'modem', 'modem', 'modem', 'modem', 'modem', 'modem', 'modem', 'modem', 'modem'], '3': ['modem', 'modem', 'envelope', 'envelope', 'laptop', 'laptop', 'envelope', 'modem', 'modem', 'modem'], '4': ['envelope', 'envelope', 'modem', 'modem', 'modem', 'laptop', 'laptop', 'laptop', 'laptop', 'traffic_light'], '5': ['envelope', 'envelope', 'laptop', 'envelope', 'envelope', 'modem', 'envelope', 'laptop', 'laptop', 'laptop'], '6': ['envelope', 'envelope', 'envelope', 'modem', 'envelope', 'envelope', 'laptop', 'laptop', 'modem', 'envelope'], '7': ['envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope'], '8': ['envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope'], '9': ['envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope', 'envelope']}

# 
    matrix = results_to_matrix(x)
    print(matrix)
    print(compute_matrix_loss(matrix))


# 
    matrix2 = np.zeros((10, 10))
    np.fill_diagonal(matrix2, 10)
    print(matrix2)
    print(compute_matrix_loss(matrix2))

    matrix2[9][9] = 9
    matrix2[9][8] = 1
    print(matrix2)
    print(compute_matrix_loss(matrix2))


    # 
    matrix3 = np.zeros((10, 7))
    for i in range(6):
        matrix3[i][i] = 10
    
    for i in range(6,10):
        matrix3[i][6] = 10

    print(matrix3)
    print(compute_matrix_loss(matrix3))

    # 
    matrix4 = np.zeros((10, 10))
    for i in range(7):
        matrix4[6-i][i] = 10
    
    for i in range(7,10):
        matrix4[i][i] = 10

    print(matrix4)
    print(compute_matrix_loss(matrix4))

    # # define data
    # p = [0.10, 0.40, 0.50]
    # q = [0.80, 0.15, 0.05]
    # # calculate cross entropy H(P, Q)
    # ce_pq = cross_entropy(p, q)
    # print(ce_pq)


