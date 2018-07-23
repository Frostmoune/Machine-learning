import numpy as np
from numpy import sqrt
import random

def getCos(A, B):
    now_a = np.array(A)
    now_b = np.array(B)
    return np.linalg.norm(now_a - now_b, 2) / (sqrt(sum(now_a ** 2)) * sqrt(sum(now_b ** 2)))

def getDis(A ,B):
    now_a = np.array(A)
    now_b = np.array(B)
    return np.linalg.norm(now_a - now_b, 2)

def Knn(train_features, train_labels, test_feature, k):
    train_dict = [(getDis(train_features[i], test_feature), train_labels[i]) for i in range(len(train_labels))]
    sorted(train_dict, key = lambda x: x[0], reverse = True)
    k_labels = train_dict[0:k]
    res = 0
    counts = {}
    for x, y in k_labels:
        if y not in counts.keys():
            counts[y] = 1
        else:
            counts[y] += 1
        if res < counts[y]:
            res = y
    return res