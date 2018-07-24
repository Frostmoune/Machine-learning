from urllib import request
import urllib
import re
import random

class DataProvider(object):
    def __init__(self, url):
        self.__url__ = url
        self.header = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0',
                    'Accept-Encoding': 'gzip, deflate'}
        self.__data__ = None
        self.getData() # 当用新数据时取消这条注释

    def getData(self):
        req = request.Request(url = self.__url__, headers = self.header)
        response = request.urlopen(req, timeout=15)
        data = response.read().decode('utf-8', 'ignore')
        self.save(data)
    
    def save(self, data):
        f = open('Alldata.csv', 'w')
        f.write(data)
        f.close()
    
    def read(self, features_num):
        f = open('Alldata.csv', 'r')
        all_lines = []
        for line in f.readlines():
            now_line = []
            if line == "\n" or line == "\r":
                continue
            line = str(line).strip().split(',')
            for i in range(0, features_num):
                now_line.append(float(line[i]))
            now_line.append(line[features_num])
            all_lines.append(now_line)
        return all_lines
    
    def getFeatureAndLabel(self, all_lines, features_num):
        now_lines = all_lines
        random.shuffle(now_lines)
        features = [x[:-1] for x in now_lines]
        labels = [x[-1] for x in now_lines]
        return features, labels


# now = DataProvider("http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
# now.getData()
# features, labels = now.readFeatureAndLabel(4)