import os
import glob
import math
import numpy as np
from performances import *

def is_illegal_num(x):
    if math.isinf(x) or math.isnan(x):
        return True
    return False

def get_scores_Protocol_1(path_scores):
    fileID = open(os.path.join(path_scores, 'Protocol_1','Dev_scores.txt'), 'r')
    lines = fileID.readlines()
    fileID.close()
    dev_scores = np.zeros([len(lines)])
    Dev_labels = np.zeros([len(lines)])
    for i in range(len(lines)):
        line = lines[i]
        file_name, file_score = line.split(',')
        dev_scores[i] = float(file_score)
        if is_illegal_num(dev_scores[i]):
            dev_scores[i] = 0.0
        if file_name[7] == '1':
            Dev_labels[i] = 1
        if file_name[7] == '2' or file_name[7] == '3':
            Dev_labels[i] = -1
        if file_name[7] == '4' or file_name[7] == '5':
            Dev_labels[i] = -2
    
    ## Evaluation on the Test set 
    fileID = open(os.path.join(path_scores, 'Protocol_1','Test_scores.txt'), 'r')
    lines = fileID.readlines()
    fileID.close()
    test_scores = np.zeros([len(lines)])
    Test_labels = np.zeros([len(lines)])
    for i in range(len(lines)):
        line = lines[i]
        file_name, file_score = line.split(',')
        file_name = hex2dec(file_name)
        test_scores[i] = float(file_score)
        if is_illegal_num(test_scores[i]):
            test_scores[i] = 0.0
        if file_name[1] == '1':
            Test_labels[i] = 1
        if file_name[1] == '2' or file_name[1] == '3':
            Test_labels[i] = -1
        if file_name[1] == '4' or file_name[1] == '5':
            Test_labels[i] = -2
    # Test_labels = (np.random.randint(2, size=np.shape(Test_labels)) - 0.5 )  * 2
    Performances_this = performances(dev_scores,Dev_labels,test_scores,Test_labels)
    return Performances_this
