from KNN import Knn
from Tree import Tree
from RandomForest import RandomForest
from DataProvider import DataProvider

# KNN测试
def knnTest(feature_len, all_lines, all_features, all_labels):
    for i in range(10):
        rate = 0
        print("Test %d:"%(i + 1))
        train_features = all_features[0:int(0.8 * len(all_features))]
        train_labels = all_labels[0:int(0.8 * len(all_features))]
        test_features = all_features[int(0.8 * len(all_features)):]
        test_labels = all_labels[int(0.8 * len(all_features)):]
        length = len(test_labels)
        for k in range(1, 10):
            rate = 0
            print("k = %d: "%k, end = " ")
            for j in range(0, length):
                res = Knn(train_features, train_labels, test_features[j], k)
                if res == test_labels[j]:
                    rate += 1
            print(rate / length)
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)

# 决策树测试
def treeTest(feature_len, all_lines, all_features, all_labels):
    best_max_son = 0
    temp = 0
    counts = {}
    for i in range(10):
        rate = 0
        print("Test %d:"%(i + 1))
        train_features = all_features[0:int(0.8 * len(all_features))]
        train_labels = all_labels[0:int(0.8 * len(all_features))]
        test_features = all_features[int(0.8 * len(all_features)):]
        test_labels = all_labels[int(0.8 * len(all_features)):]
        for max_son in range(2, 11):
            rate = 0
            if max_son not in counts:
                counts[max_son] = 0
            print("max_son:%d "%(max_son), end = " ")
            new_tree = Tree(train_features, train_labels, len(train_features[0]), max_son, 5)
            new_tree.train()
            length = len(test_labels)
            for j in range(0, length):
                res = new_tree.predictTree(test_features[j])
                if res == test_labels[j]:
                    rate += 1
            print(rate / length)
            counts[max_son] += rate / length
            if temp < counts[max_son]:
                temp = counts[max_son]
                best_max_son = max_son 
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)
    print("Best best_max_son:%d %f"%(best_max_son, counts[best_max_son] / 10))

# 随机森林测试
def randomForestTest(feature_len, all_lines, all_features, all_labels):
    best_trees_num = 0
    temp = 0
    counts = {}
    for i in range(10):
        rate = 0
        print("Test %d:"%(i + 1))
        train_features = all_features[0:int(0.8 * len(all_features))]
        train_labels = all_labels[0:int(0.8 * len(all_features))]
        test_features = all_features[int(0.8 * len(all_features)):]
        test_labels = all_labels[int(0.8 * len(all_features)):]
        for trees_num in range(5, 26):
            rate = 0
            if trees_num not in counts:
                counts[trees_num] = 0
            print("trees_num:%d "%(trees_num), end = " ")
            new_forest = RandomForest(trees_num)
            new_forest.buildTrees(train_features, train_labels, len(train_features[0]), 3, 5)
            length = len(test_labels)
            for j in range(0, length):
                res = new_forest.predictForest(test_features[j])
                if res == test_labels[j]:
                    rate += 1
            print(rate / length)
            counts[trees_num] += rate / length
            if temp < counts[trees_num]:
                temp = counts[trees_num]
                best_trees_num = trees_num
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)
    print("Best trees_num:%d %f"%(best_trees_num, counts[best_trees_num] / 10))

if __name__ == "__main__":
    feature_len = 60
    now_provider = DataProvider("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    all_lines = now_provider.read(feature_len)
    # feature_len = 4
    # now_provider = DataProvider("http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
    # all_lines = now_provider.read(feature_len)
    all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)

    cho = '3'
    if cho == '1':
        knnTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '2':
        treeTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '3':
        randomForestTest(feature_len, all_lines, all_features, all_labels)