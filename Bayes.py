import numpy as np 
import math
from math import exp, sqrt

class NaiveBayes(object):
    def __init__(self, all_features, all_labels, feature_len):
        self.N = len(all_labels)
        self.features = np.array(all_features)
        self.labels = np.array(all_labels)
        self.feature_len = feature_len

        self.setClassfy()
        self.mu = np.zeros((self.classfy_num, self.feature_len))
        self.sigma = np.zeros((self.classfy_num, self.feature_len))
        self.counts = [0 for i in range(self.classfy_num)]

        # 概率密度
        self.f = lambda x, i, j:1 / (sqrt(2 * math.pi) * self.sigma[i, j]) * exp(-((x[j] - self.mu[i, j]) ** 2) / 2 * (self.sigma[i, j] ** 2))

    # 得到每种类别对应的特征集
    def setClassfy(self):
        self.classfy = {}
        self.classfy_probability = {}
        self.classfy_num = 0
        for i in range(self.N):
            if self.labels[i] not in self.classfy:
                self.classfy[self.labels[i]] = [self.features[i]]
                self.classfy_probability[self.labels[i]] = 1
                self.classfy_num += 1
            else:
                self.classfy[self.labels[i]].append(self.features[i])
                self.classfy_probability[self.labels[i]] += 1

        for x in self.classfy.keys():
            self.classfy[x] = np.array(self.classfy[x])
            self.classfy_probability[x] /= self.N
    
    # 极大似然估计
    def maxLikelihood(self):
        i = 0
        for x in self.classfy.keys():
            self.mu[i, :] = sum([y for y in self.classfy[x]]) / len(self.classfy[x])
            self.sigma[i, :] = sum([(y - self.mu[i, :]) * (y - self.mu[i, :]).T for y in self.classfy[x]]) / len(self.classfy[x])
            self.counts[i] = x
            i += 1
    
    # 训练
    def train(self):
        self.maxLikelihood()
    
    # 预测
    def predict(self, test_feature):
        best_p, best_class = 0, -1
        for i in range(self.classfy_num):
            now_p = self.classfy_probability[self.counts[i]]
            for j in range(self.feature_len):
                temp_p = self.f(test_feature, i, j)
                now_p *= self.f(test_feature, i, j)
            if best_p < now_p:
                best_p = now_p
                best_class = self.counts[i]
        return best_class