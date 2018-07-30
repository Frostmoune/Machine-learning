from KNN import Knn
from Tree import Tree
from RandomForest import RandomForest
from SVM import SVM
from Bayes import NaiveBayes
from Logistic import Logistic
from AdaBoost import AdaBoost
from NeuralNetwork import NeuralNetwork
from DataProvider import DataProvider

# KNN测试
def knnTest(feature_len, all_lines, all_features, all_labels):
    counts = {}
    for i in range(10):
        rate = 0
        print("Test %d:"%(i + 1))
        train_features = all_features[0:int(0.8 * len(all_features))]
        train_labels = all_labels[0:int(0.8 * len(all_features))]
        test_features = all_features[int(0.8 * len(all_features)):]
        test_labels = all_labels[int(0.8 * len(all_features)):]
        length = len(test_labels)
        for k in range(1, 5):
            rate = 0
            print("k = %d: "%k, end = " ")
            for j in range(0, length):
                res = Knn(train_features, train_labels, test_features[j], k)
                if res == test_labels[j]:
                    rate += 1
            print(rate / length)
            if k not in counts:
                counts[k] = rate / length
            else:
                counts[k] += rate / length
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)
    for x in counts:
        print(x, counts[x])

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
        new_tree = Tree(train_features, train_labels, len(train_features[0]), 0, 8)
        for max_son in range(2, 10):
            rate = 0
            if max_son not in counts:
                counts[max_son] = 0
            print("max_son:%d "%(max_son), end = " ")
            new_tree.setMaxSon(max_son)
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
            new_tree.deleteRoot() 
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)
    print("Best best_max_leaf:%d %f"%(best_max_son, counts[best_max_son] / 10))
    for x in counts:
        print(x, counts[x])

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
        for trees_num in range(25, 36):
            rate = 0
            if trees_num not in counts:
                counts[trees_num] = 0
            print("trees_num:%d "%(trees_num), end = " ")
            new_forest = RandomForest(trees_num)
            new_forest.buildTrees(train_features, train_labels, len(train_features[0]), 3, 6)
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
    for x in counts:
        print(x, counts[x])

# SVM测试
def svmTest(feature_len, all_lines, all_features, all_labels):
    counts = {}
    for i in range(10):
        rate = 0
        print("Test %d:"%(i + 1))
        train_features = all_features[0:int(0.8 * len(all_features))]
        train_labels = all_labels[0:int(0.8 * len(all_features))]
        test_features = all_features[int(0.8 * len(all_features)):]
        test_labels = all_labels[int(0.8 * len(all_features)):]
        length = len(test_labels)
        for C in range(50, 61, 1):
            rate = 0
            new_svm = SVM(train_features, train_labels, C = C, function = 'RBF', d = 0.53)
            # print("Train:")
            new_svm.train()
            # print("\nPredict:", end = "\n")
            for j in range(0, length): 
                res = new_svm.predict(test_features[j])
                if res == test_labels[j]:
                    rate += 1
            print("C = %f: "%C, end = " ")
            print(rate / length)
            if C not in counts:
                counts[C] = rate / length
            else:
                counts[C] += rate / length
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)
    for x, y in counts:
        print(x, y)

# 朴素贝叶斯测试
def naiveBayesTest(feature_len, all_lines, all_features, all_labels):
    for i in range(10): 
        print("Test %d"%(i + 1))
        train_features = all_features[0:int(0.8 * len(all_features))]
        train_labels = all_labels[0:int(0.8 * len(all_features))]
        test_features = all_features[int(0.8 * len(all_features)):]
        test_labels = all_labels[int(0.8 * len(all_features)):]
        length = len(test_labels)

        rate = 0
        new_bayes = NaiveBayes(train_features, train_labels, feature_len)
        new_bayes.train()
        for j in range(0, length): 
            res = new_bayes.predict(test_features[j])
            if res == test_labels[j]:
                rate += 1
        print("Rate is", (rate / length))
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)

# logistic回归测试
def logisticTest(feature_len, all_lines, all_features, all_labels):
    for i in range(10): 
        print("Test %d"%(i + 1))
        train_features = all_features[0:int(0.8 * len(all_features))]
        train_labels = all_labels[0:int(0.8 * len(all_features))]
        test_features = all_features[int(0.8 * len(all_features)):]
        test_labels = all_labels[int(0.8 * len(all_features)):]
        length = len(test_labels)

        rate = 0
        for tol in [1 / (10 ** i) for i in range(1, 8)]:
            rate = 0
            new_logistic = Logistic(train_features, train_labels, feature_len, alpha = 5, tol = tol)
            new_logistic.train()
            for j in range(0, length): 
                res = new_logistic.predict(test_features[j])
                if res == test_labels[j]:
                    rate += 1
            print("tol = %f: %f"%(tol, rate / length))
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)

# AdaBoost森林测试
def adaBoostTest(feature_len, all_lines, all_features, all_labels):
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
        for trees_num in range(25, 36):
            rate = 0
            if trees_num not in counts:
                counts[trees_num] = 0
            print("trees_num:%d "%(trees_num), end = " ")
            new_boost = AdaBoost(train_features, train_labels, len(train_features[0]), trees_num, mode = 2)
            new_boost.train()
            # new_boost.showInfo()
            length = len(test_labels)
            for j in range(0, length):
                res = new_boost.predict(test_features[j])
                if res == test_labels[j]:
                    rate += 1
            print(rate / length)
            counts[trees_num] += rate / length
            if temp < counts[trees_num]:
                temp = counts[trees_num]
                best_trees_num = trees_num
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)
    print("Best trees_num:%d %f"%(best_trees_num, counts[best_trees_num] / 10))
    for x in counts:
        print(x, counts[x])

# 神经网络测试
def neuralNetworkTest(feature_len, all_lines, all_features, all_labels):
    best_hidden_num = 0
    temp = 0
    counts = {}
    for i in range(10):
        rate = 0
        print("Test %d:"%(i + 1))
        train_features = all_features[0:int(0.8 * len(all_features))]
        train_labels = all_labels[0:int(0.8 * len(all_features))]
        test_features = all_features[int(0.8 * len(all_features)):]
        test_labels = all_labels[int(0.8 * len(all_features)):]
        for hidden_num in range(30, 41):
            rate = 0
            if hidden_num not in counts:
                counts[hidden_num] = 0
            print("hidden_num:%d "%(hidden_num), end = " ")
            new_NN = NeuralNetwork(train_features, train_labels, len(train_features[0]), hidden_num, learn_rate = 100)
            new_NN.train()
            length = len(test_labels)
            for j in range(0, length):
                res = new_NN.predict(test_features[j])
                if res == test_labels[j]:
                    rate += 1
            print(rate / length)
            counts[hidden_num] += rate / length
            if temp < counts[hidden_num]:
                temp = counts[hidden_num]
                best_hidden_num = hidden_num
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)
    print("Best hidden_num:%d %f"%(best_hidden_num, counts[best_hidden_num] / 10))
    for x in counts:
        print(x, counts[x])


# 各算法比较测试
def compareTest(feature_len, all_lines, all_features, all_labels):
    count = {}
    for i in range(10): 
        print("\nTest %d"%(i + 1))
        train_features = all_features[0:int(0.8 * len(all_features))]
        train_labels = all_labels[0:int(0.8 * len(all_features))]
        test_features = all_features[int(0.8 * len(all_features)):]
        test_labels = all_labels[int(0.8 * len(all_features)):]
        length = len(test_labels)

        rate = 0
        print("NaiveBayes : ", end = "")
        new_bayes = NaiveBayes(train_features, train_labels, feature_len)
        new_bayes.train()
        for j in range(0, length): 
            res = new_bayes.predict(test_features[j])
            if res == test_labels[j]:
                rate += 1
        print(rate / length)
        if "NaiveBayes" not in count:
            count["NaiveBayes"] = rate / length
        else:
            count["NaiveBayes"] += rate / length

        rate = 0
        print("KNN : ", end = "")
        for j in range(0, length): 
            res = Knn(train_features, train_labels, test_features[j], 3)
            if res == test_labels[j]:
                rate += 1
        print(rate / length)
        if "KNN" not in count:
            count["KNN"] = rate / length
        else:
            count["KNN"] += rate / length

        rate = 0
        print("Logistic : ", end = "")
        new_logistic = Logistic(train_features, train_labels, feature_len, alpha = 5, tol = 0.000001)
        new_logistic.train()
        for j in range(0, length): 
            res = new_logistic.predict(test_features[j])
            if res == test_labels[j]:
                rate += 1
        print(rate / length)
        if "Logistic" not in count:
            count["Logistic"] = rate / length
        else:
            count["Logistic"] += rate / length
        
        rate = 0
        print("NeuralNetwork : ", end = "")
        new_NN = NeuralNetwork(train_features, train_labels, feature_len, hidden_num = 32, learn_rate = 100)
        new_NN.train()
        for j in range(0, length): 
            res = new_NN.predict(test_features[j])
            if res == test_labels[j]:
                rate += 1
        print(rate / length)
        if "NeuralNetwork" not in count:
            count["NeuralNetwork"] = rate / length
        else:
            count["NeuralNetwork"] += rate / length
        
        rate = 0
        print("Tree : ", end = "")
        new_tree = Tree(train_features, train_labels, len(train_features[0]), 3, 8)
        new_tree.train()
        for j in range(0, length): 
            res = new_tree.predictTree(test_features[j])
            if res == test_labels[j]:
                rate += 1
        print(rate / length)
        if "Tree" not in count:
            count["Tree"] = rate / length
        else:
            count["Tree"] += rate / length

        rate = 0
        print("AdaBoost : ", end = "")
        new_boost = AdaBoost(train_features, train_labels, len(train_features[0]), 28, mode = 2)
        new_boost.train()
        for j in range(0, length): 
            res = new_boost.predict(test_features[j])
            if res == test_labels[j]:
                rate += 1
        print(rate / length)
        if "AdaBoost" not in count:
            count["AdaBoost"] = rate / length
        else:
            count["AdaBoost"] += rate / length

        rate = 0
        print("RandomForest : ", end = "")
        new_forest = RandomForest(30)
        new_forest.buildTrees(train_features, train_labels, len(train_features[0]), 3, 6)
        for j in range(0, length): 
            res = new_forest.predictForest(test_features[j])
            if res == test_labels[j]:
                rate += 1
        print(rate / length)
        if "RandomForest" not in count:
            count["RandomForest"] = rate / length
        else:
            count["RandomForest"] += rate / length

        rate = 0
        print("SVM : ", end = "")
        new_svm = SVM(train_features, train_labels, C = 43, function = 'RBF', d = 0.53)
        new_svm.train()
        for j in range(0, length): 
            res = new_svm.predict(test_features[j])
            if res == test_labels[j]:
                rate += 1
        print(rate / length)
        if "SVM" not in count:
            count["SVM"] = rate / length
        else:
            count["SVM"] += rate / length

        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)
    
    print("\nAverage:")
    for x in count:
        print(x, end = ": ")
        print(count[x] / 10)

if __name__ == "__main__":
    feature_len = 60
    now_provider = DataProvider("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    all_lines = now_provider.read(feature_len)
    # feature_len = 4
    # now_provider = DataProvider("http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
    # all_lines = now_provider.read(feature_len)
    all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, feature_len)

    cho = '8'
    if cho == '1':
        knnTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '2':
        treeTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '3':
        randomForestTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '4':
        svmTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '5':
        naiveBayesTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '6':
        logisticTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '7':
        adaBoostTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '8':
        neuralNetworkTest(feature_len, all_lines, all_features, all_labels)
    elif cho == '9':
        compareTest(feature_len, all_lines, all_features, all_labels)