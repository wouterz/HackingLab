import numpy as np

flatten = lambda l: [item for sublist in l for item in sublist]

matrix = [];
valueDict = dict()
keyDict = dict()

def results_to_matrix(results):
    global matrix
    keys = results.keys()
    for key in keys:
        if(key not in keyDict.keys()):
            keyDict[key]=len(keyDict.keys());
    values = set(flatten(results.values()))
    if(isinstance(matrix, list)):
        matrix = np.zeros((len(keys), 0))
    for val in values:
        if(val not in valueDict.keys()):
            matrix = np.append(matrix, np.zeros((len(matrix),1)), axis=1)
            valueDict[val]=len(valueDict.keys())
    for i in results.keys():
        for j in results[i]:
            matrix[keyDict[i]][valueDict[j]]+=1
    #for value in results.values():
    #    el=[]
    #    for c in classes: el.append(value.count(c))
    #    matrix.append(el)
    return matrix
classes = ['switch', 'crossword_puzzle', 'digital_clock']
results = {'1': ['switch', 'switch', 'switch'], '2': ['switch', 'switch', 'switch'], '3': ['digital_clock', 'digital_clock', 'digital_clock'], '4': ['crossword_puzzle', 'digital_clock', 'digital_clock'], '5': ['crossword_puzzle', 'crossword_puzzle', 'crossword_puzzle'], '6': ['crossword_puzzle', 'crossword_puzzle', 'crossword_puzzle'], '7': ['crossword_puzzle', 'crossword_puzzle', 'crossword_puzzle'], '8': ['crossword_puzzle', 'crossword_puzzle', 'crossword_puzzle'], '9': ['crossword_puzzle']}
results2 = {'2': ['apple']}
print(results_to_matrix(results))
print(results_to_matrix(results2))
