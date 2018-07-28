import numpy as np
from math import exp 

class Logistic(object):
    def __init__(self, all_features, all_labels, feature_len, alpha = 1, tol = 0.001):
        self.N = len(all_features)
        self.features = np.array(all_features)
        self.features = np.column_stack((np.ones((self.N, 1)), self.features))
        self.calLabels(all_labels)
        self.feature_len = feature_len + 1

        self.theta = np.ones((self.feature_len, 1))
        self.sigmod = lambda x:np.array([1 / (1 + exp(-x[i])) for i in range(self.N)])
        self.tol = tol
        self.alpha = alpha

        self.predict_sigmod = lambda x:1 / (1 + exp(-x))
    
    # 处理labels
    def calLabels(self, all_labels):
        self.labels = np.zeros((1, self.N))
        self.counts = {}
        key = 0
        for i in range(self.N):
            if all_labels[i] not in self.counts:
                self.counts[all_labels[i]] = key
                key += 1
            self.labels[0, i] = self.counts[all_labels[i]]
    
    # 梯度上升
    def gradientAscent(self):
        while True:
            temp = self.theta
            now_y = np.dot(self.features, temp)
            now_z = self.sigmod(now_y).reshape((self.N, 1))
            now_error = self.labels.T - now_z
            self.theta += self.alpha * np.dot(self.features.T, now_error)
            now_tol = np.linalg.norm(temp - self.theta, np.inf)
            if now_tol < self.tol:
                break
    
    # 训练
    def train(self):
        self.gradientAscent()

    # 预测
    def predict(self, test_feature):
        x = np.array([1])
        x = np.column_stack((x, np.array(test_feature).reshape(1, len(test_feature))))
        y = np.dot(x, self.theta)
        if y < -700:
            res = 0
        else:
            res = 0 if self.predict_sigmod(y) <= 0.5 else 1
        for x in self.counts:
            if res == self.counts[x]:
                return x
        return None