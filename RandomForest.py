from Tree import Tree, Node
import numpy as np 
import random

class RandomTree(Tree):
    def __init__(self, all_features, all_labels, feature_len, max_son, max_leaf_data):
        super(RandomTree, self).__init__(all_features, all_labels, feature_len, max_son, max_leaf_data)
        self.divide_feature_len = int(np.log2(feature_len)) + 1
    
    def buildRandomNode(self, now_root):
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

        # 随机选择用于划分的特征值的子集
        divide_list = list(range(self.feature_len))
        random.shuffle(divide_list)
        divide_list = divide_list[0:self.divide_feature_len]

        for i in divide_list:
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
            self.buildRandomNode(new_son)
            now_root.son.append(new_son)

    def train(self):
        self.buildRandomNode(self.root) 

class RandomForest(object):
    def __init__(self, trees_num):
        self.trees_num = trees_num
        self.trees = [0 for i in range(trees_num)]
    
    # 得到每棵树用于训练的数据
    def getRandomData(self, all_features, all_labels):
        sub_features = []
        sub_labels = []
        for i in range(len(all_features)):
            if random.random() < 0.500000001:
                sub_features.append(all_features[i])
                sub_labels.append(all_labels[i])
        return sub_features, sub_labels
    
    # 建立每棵树
    def buildTrees(self, all_features, all_labels, feature_len, max_son, max_leaf_data):
        for i in range(self.trees_num):
            sub_features, sub_labels = self.getRandomData(all_features, all_labels)
            self.trees[i] = RandomTree(sub_features, sub_labels, feature_len, max_son, max_leaf_data)
            self.trees[i].train()
    
    # 用随机森林进行预测
    def predictForest(self, test_feature):
        res = -1
        temp = -1
        now_res = 0
        count = {}
        for i in range(self.trees_num):
            now_res = self.trees[i].predictTree(test_feature)
            if now_res not in count:
                count[now_res] = 1
            else:
                count[now_res] += 1
            if temp < count[now_res]:
                temp = count[now_res]
                res = now_res
        return res