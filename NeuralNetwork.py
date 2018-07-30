import numpy as np 
import random
from math import exp 

class Neuron(object):
    def __init__(self, is_input = 0):
        if is_input:
            self.f = lambda x:x
        else:
            self.f = lambda x:0 if x < -700 else 1 / (1 + exp(-x))
            self.thresold = -9 + random.random() * 20
    
    def setThresold(self, thresold):
        self.thresold = thresold
    
    def setInput(self, input_):
        self.input = input_
    
    def calOutput(self):
        self.output = self.f(self.input - self.thresold)
        return self.output

class NeuralNetwork(object):
    def __init__(self, all_features, all_labels, feature_len, hidden_num, output_num = 1, learn_rate = 0.5):
        self.input_num = feature_len
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.learn_rate = learn_rate

        self.N = len(all_features)
        self.features = np.array(all_features)
        self.calLabels(all_labels)

        self.input_layer = [Neuron(1) for i in range(self.input_num)]
        self.hidden_layer = [Neuron() for i in range(self.hidden_num)]
        self.output_layer = [Neuron() for i in range(self.output_num)]
        self.input_hidden_weight = -10 + np.random.rand(self.input_num, self.hidden_num) * 20
        self.hidden_output_weight = -10 + np.random.rand(self.hidden_num, self.output_num) * 20
        self.hidden_output_buffer = np.zeros((1, self.hidden_num))
        self.output_buffer = np.zeros((1, self.output_num))
    
    # 处理labels
    def calLabels(self, all_labels):
        self.count = {}
        self.labels = [0 for i in range(self.N)]
        flag = 0
        for i in range(self.N):
            if all_labels[i] not in self.count:
                self.count[all_labels[i]] = flag
                flag += 1
            self.labels[i] = self.count[all_labels[i]]
        self.labels = np.array(self.labels).reshape((self.N, self.output_num))
    
    # 计算神经网络的结果
    def calOutput(self, now_input):
        for i in range(self.hidden_num):
            self.hidden_layer[i].setInput(np.dot(now_input, self.input_hidden_weight[:,i]))
            self.hidden_output_buffer[0, i] = self.hidden_layer[i].calOutput()
        
        for j in range(self.output_num):
            self.output_layer[j].setInput(np.dot(self.hidden_output_buffer, self.hidden_output_weight[:,j]))
            self.output_buffer[0, j] = self.output_layer[j].calOutput()
    
    # 计算误差
    def calError(self, index):
        return 1 / 2 * np.linalg.norm(self.output_buffer - self.labels[index, :], 2)
    
    # 计算G
    def calG(self, index):
        res = self.output_buffer
        return np.array([res[0, i] * (1 - res[0, i]) * (self.labels[index, i] - res[0, i]) for i in range(self.output_num)])
    
    # 计算E
    def calE(self, g):
        res = self.hidden_output_buffer
        return np.array([res[0, i] * (1 - res[0, i]) * np.dot(self.hidden_output_weight[i, :], g) for i in range(self.hidden_num)])
    
    # 误差逆传播
    def BP(self, index):
        now_input = self.features[index].reshape((1, self.input_num))
        self.calOutput(now_input)
        g = self.calG(index).reshape((1, self.output_num))
        e = self.calE(g).reshape((1, self.hidden_num))

        self.hidden_output_weight = self.hidden_output_weight + self.learn_rate * np.dot(self.hidden_output_buffer.T, g)
        self.input_hidden_weight = self.input_hidden_weight + self.learn_rate * np.dot(now_input.T, e)

        for j in range(self.output_num):
            self.output_layer[j].thresold -= self.learn_rate * g[0, j]
        for h in range(self.hidden_num):
            self.hidden_layer[h].thresold -= self.learn_rate * e[0, h]
    
    # 训练
    def train(self):
        for i in range(self.N):
            self.BP(i)
    
    # 预测
    def predict(self, test_feature):
        self.calOutput(np.array(test_feature))
        res = 0 if self.output_buffer[0, 0] <= 0.5 else 1
        for x in self.count:
            if res == self.count[x]:
                return x
        return None
