import numpy as np 
from Tree import Node, Tree 
from math import log, exp

class Stump(Tree):
    def __init__(self, all_features, all_labels, feature_len, max_son = 3, max_leaf_data = 8):
        super(Stump, self).__init__(all_features, all_labels, feature_len, max_son, max_leaf_data)
    
    # 设置当前用于划分的特征值
    def setDivideFeatureIndex(self, divide_feature_index):
        self.tree_divide_feature_index = divide_feature_index
    
    # 建立树桩
    def buildStumpNode(self, now_root):
        now_counts = self.getCount(now_root.data_index)
        if now_counts == None or len(now_counts) == 0:
            return
        if len(now_root.data_index) <= self.max_leaf_data or len(now_counts) == 1:
            now_root.is_leaf = True
            self.setClassfy(now_root, now_counts)
            return

        best_value, best_sub_data_indexs = self.getSubIndex(now_root.data_index, self.tree_divide_feature_index)
        for i in range(len(best_sub_data_indexs)):
            new_son = Node(best_sub_data_indexs[i])
            new_son.setDivideFeatureAndValue(self.tree_divide_feature_index, best_value[0] + best_value[1] * (i + 1))
            self.buildStumpNode(new_son)
            now_root.son.append(new_son)
    
    # 训练
    def train(self):
        self.buildStumpNode(self.root)
    
    # 预测的子函数
    def predictNode(self, now_node, test_feature):
        if now_node.is_leaf:
            return now_node.classfy
        for x in now_node.son:
            if test_feature[x.divide_feature_index] < x.divide_value:
                return self.predictNode(x, test_feature)
        return 0

    # 预测新特征
    def predict(self, test_feature):
        return self.predictNode(self.root, test_feature)

class AdaBoost(object):
    def __init__(self, all_features, all_labels, feature_len, select_tree_num, mode = 1):
        self.N = len(all_features)
        self.calLabels(all_labels)
        self.features = all_features
        self.feature_len = feature_len
        self.mode = mode

        self.D = np.ones((1, self.N)) / self.N
        self.select_tree_num = select_tree_num
        self.H = [0 for i in range(select_tree_num)]
        self.alpha = [0 for i in range(select_tree_num)]
        self.trainTrees()
    
    # 处理labels
    def calLabels(self, all_labels):
        self.count = {}
        self.labels = [0 for i in range(self.N)]
        flag = 1
        for i in range(self.N):
            if all_labels[i] not in self.count:
                self.count[all_labels[i]] = flag
                flag *= -1
            self.labels[i] = self.count[all_labels[i]]
    
    # 计算每一棵树的误差
    def calError(self, new_tree):
        rate = 0
        error = np.zeros((self.N, 1))
        for i in range(self.N):
            if new_tree.predictTree(self.features[i]) != self.labels[i]:
                rate += 1
                error[i, 0] = 1
        return rate / self.N, error
    
    # 训练每一棵树
    def trainTrees(self):
        self.trees = []
        self.trees_error = []
        self.epsilon = []
        if self.mode == 1:
            self.tree_num = self.feature_len
            self.trees = [0 for i in range(self.tree_num)]
            self.trees_error = [0 for i in range(self.tree_num)]
            self.epsilon = [0 for i in range(self.tree_num)]
            for i in range(self.feature_len):
                self.trees[i] = Stump(self.features, self.labels, self.feature_len)
                self.trees[i].setDivideFeatureIndex(i)
                self.trees[i].train()
                self.epsilon[i], self.trees_error[i] = self.calError(self.trees[i])
        else:
            self.tree_num = 49
            self.trees = [0 for i in range(self.tree_num)]
            self.trees_error, self.epsilon = [0 for i in range(self.tree_num)], [0 for i in range(self.tree_num)] 
            i = 0
            for max_son in range(2, 9):
                for max_leaf_data in range(2, 9):
                    self.trees[i] = Tree(self.features, self.labels, self.feature_len, 0, 0)
                    self.trees[i].setMaxSon(max_son)
                    self.trees[i].setMaxLeafData(max_leaf_data)
                    self.trees[i].train()
                    self.epsilon[i], self.trees_error[i] = self.calError(self.trees[i])
                    i += 1
    
    # 选择当前最优的树
    def selectTree(self):
        best_tree_index, min_error = -1, 100000000
        for i in range(self.tree_num):
            now_error = np.dot(self.D, self.trees_error[i])
            if min_error > now_error:
                min_error = now_error
                best_tree_index = i
        return best_tree_index
    
    # 更新D
    def updateD(self, best_index, i):
        for j in range(self.N):
            if self.trees_error[best_index][j, 0] == 0:
                self.D[0, j] *= exp(-self.alpha[i])
            else:
                self.D[0, j] *= exp(self.alpha[i])
    
    # boosting主要过程
    def boosting(self):
        for i in range(self.select_tree_num):
            best_index = self.selectTree()
            now_epsilon = self.epsilon[best_index]
            self.alpha[i] = 1 / 2 * log((1 - now_epsilon) / now_epsilon)
            self.updateD(best_index, i)
            self.H[i] = self.trees[best_index]
            self.H[i].tree_index = best_index
    
    # 训练
    def train(self):
        self.boosting()
    
    # debug
    def showInfo(self):
        for i in range(len(self.H)):
            now_index = self.H[i].tree_index
            print(self.alpha[i], self.epsilon[now_index])
            
    # 预测
    def predict(self, test_feature):
        res = sum([self.alpha[i] * self.H[i].predictTree(test_feature) for i in range(self.select_tree_num)])
        if res < 0:
            res = -1
        else:
            res = 1
        for x in self.count:
            if res == self.count[x]:
                return x
        return None