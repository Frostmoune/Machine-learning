import numpy as np 
import random

class Node(object):
    def __init__(self, data_index, classfy = None, is_leaf = False):
        self.data_index = data_index
        self.classfy = classfy
        self.is_leaf = is_leaf
        if is_leaf:
            self.son = None
        else:
            self.son = []
    
    def setDivideFeatureAndValue(self, divide_feature_index, divide_value):
        self.divide_feature_index = divide_feature_index
        self.divide_value = divide_value
    

class Tree(object):
    # feature_len:特征维度；max_son:每个结点最多子节点数；max_leaf_data：叶节点数据最大维度
    def __init__(self, all_features, all_labels, feature_len, max_son, max_leaf_data):
        self.all_features = all_features
        self.all_labels = all_labels
        self.feature_len = feature_len
        self.max_son = max_son
        self.max_leaf_data = max_leaf_data
        self.root = Node(list(range(0, len(all_features))))
    
    # 设置max_son
    def setMaxSon(self, max_son):
        self.max_son = max_son
    
    # 设置max_leaf_data
    def setMaxLeafData(self, max_leaf_data):
        self.max_leaf_data = max_leaf_data
    
    # 清空root
    def deleteRoot(self):
        self.root = Node(list(range(0, len(self.all_features))))
    
    # 对数据根据其中一个特征进行划分
    def getSubIndex(self, data_index, feature_index):
        now_data = [self.all_features[x][feature_index] for x in data_index]
        max_value = max(now_data)
        min_value = min(now_data)
        sub_data_indexs = [[] for x in range(self.max_son)]
        step = (max_value - min_value) / self.max_son
        for i in data_index:
            for j in range(0, self.max_son):
                if self.all_features[i][feature_index] < min_value + (j + 1) * step:
                    sub_data_indexs[j].append(i)
                    break
        return (min_value, step), sub_data_indexs
    
    # 计算信息熵
    def getEntropy(self, count):
        return -sum([count[x] * np.log2(count[x]) for x in count])
    
    # 根据数据得到概率
    def getCount(self, data_index):
        count = {}
        for i in data_index:
            if self.all_labels[i] not in count:
                count[self.all_labels[i]] = 1
            else:
                count[self.all_labels[i]] += 1
        for x in count:
            count[x] /= len(data_index)
        return count
    
    # 计算信息增益
    def getGain(self, data_index, sub_data_indexs):
        now_entropy = self.getEntropy(self.getCount(data_index))
        sub_entropy_sum = sum([len(x) / len(data_index) * self.getEntropy(self.getCount(x)) for x in sub_data_indexs])
        return now_entropy - sub_entropy_sum
    
    # 计算属性的固有值
    def getIV(self, data_index, sub_data_indexs):
        return -sum([len(x) / len(data_index) * np.log2(len(x) / len(data_index)) for x in sub_data_indexs])
    
    # 计算属性的信息增益率
    def getGainRadio(self, data_index, sub_data_indexs):
        return self.getGain(data_index, sub_data_indexs) / self.getIV(data_index, sub_data_indexs)
    
    # 设定结点类别
    def setClassfy(self, now_root, now_counts):
        num = -1
        classfy = -1
        for x in now_counts:
            if num < now_counts[x]:
                classfy = x
        now_root.classfy = classfy
    
    # 建立新节点
    def buildNode(self, now_root):
        now_counts = self.getCount(now_root.data_index)
        if now_counts == None or len(now_counts) == 0:
            return
        if len(now_root.data_index) <= self.max_leaf_data or len(now_counts) == 1:
            now_root.is_leaf = True
            self.setClassfy(now_root, now_counts)
            return
        best_sub_data_indexs = []
        best_gain = -100000000
        best_divide_feature_index = -1
        best_value = ()
        for i in range(self.feature_len):
            now_value, now_sub_data_indexs = self.getSubIndex(now_root.data_index, i)
            now_gain = self.getGain(now_root.data_index, now_sub_data_indexs)
            if best_gain < now_gain:
                best_divide_feature_index = i
                best_gain = now_gain
                best_sub_data_indexs = now_sub_data_indexs
                best_value = now_value

        for i in range(len(best_sub_data_indexs)):
            new_son = Node(best_sub_data_indexs[i])
            new_son.setDivideFeatureAndValue(best_divide_feature_index, best_value[0] + best_value[1] * (i + 1))
            self.buildNode(new_son)
            now_root.son.append(new_son)
    
    # 训练
    def train(self):
        self.buildNode(self.root)
        
    # 预测的子函数
    def predictNode(self, now_node, test_feature):
        if now_node.is_leaf:
            return now_node.classfy
        for i in range(len(now_node.son)):
            if test_feature[now_node.son[i].divide_feature_index] < now_node.son[i].divide_value:
                return self.predictNode(now_node.son[i], test_feature)
            elif i == len(now_node.son) - 1 and test_feature[now_node.son[i].divide_feature_index] >= now_node.son[i].divide_value:
                return self.predictNode(now_node.son[i], test_feature)
        return self.all_labels[random.randint(0, len(self.all_labels) - 1)]

    # 预测新特征
    def predictTree(self, test_feature):
        return self.predictNode(self.root, test_feature)