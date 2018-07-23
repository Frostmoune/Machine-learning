from KNN import Knn
from DataProvider import DataProvider

def knnTest(all_features, all_labels):
    for i in range(10):
        rate = 0
        print("Test %d:"%(i + 1))
        train_features = all_features[0:1200]
        train_labels = all_labels[0:1200]
        test_features = all_features[1200:]
        test_labels = all_labels[1200:]
        length = len(test_labels)
        for k in range(1, 10):
            rate = 0
            print("k = %d:"%k, end = " ")
            for i in range(0, length):
                res = Knn(train_features, train_labels, test_features[i], k)
                if res == test_labels[i]:
                    rate += 1
            print(rate / length)
        all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, 4)

if __name__ == "__main__":
    now_provider = DataProvider("http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
    all_lines = now_provider.read(4)
    all_features, all_labels = now_provider.getFeatureAndLabel(all_lines, 4)

    knnTest(all_features, all_labels)