import numpy as np
import random 

class SVM(object):
    def __init__(self, all_features, all_labels, C, function = 'RBF', d = 1, tol = 0.00001):
        self.features = np.array(all_features)
        self.setSVMLabels(all_labels)
        # 只有RBF和线性核
        if function == 'RBF':
            self.kernel = lambda x, y: np.exp(-np.linalg.norm(x - y, 2) / (2 * (d ** 2)))
        else:
            self.kernel = lambda x, y: np.dot(x, y.T) ** d
        
        # SMO迭代所需的参数
        self.N = len(all_features) # 样本量
        self.B = 0 # 偏移量
        self.C = C # SVM参数
        self.alpha = np.zeros((self.N)) # alpha
        self.setKernelMat()
        self.eCache = np.zeros((self.N, 2)) # 误差缓存
        self.f = lambda x:sum([self.alpha[i] * self.labels[i] * self.kernel(x, self.features[i]) for i in range(self.N)]) + self.B
        self.tol = tol # KKT阈值
        self.alpha_tol = 0.000001 # alpha改变阈值
        self.iter_num = 150 # 最大迭代次数
    
    # 改变label以供SMO训练
    def setSVMLabels(self, all_labels):
        self.labels = []
        self.counts = {}
        flag = 1
        for x in all_labels:
            if x not in self.counts:
                self.counts[x] = flag
                flag *= -1
            self.labels.append(self.counts[x])
        self.labels = np.array(self.labels)
    
    # 得到核函数对应的核矩阵
    def setKernelMat(self):
        self.kernel_mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.kernel_mat[i, j] = self.kernel(self.features[i], self.features[j])
    
    # 更新函数和误差
    def updateFAndE(self, i, j):
        self.f = lambda x:sum([self.alpha[i] * self.labels[i] * self.kernel(x, self.features[i]) for i in range(self.N)]) + self.B
        self.eCache[j] = [1, self.getE(j)]
    
    # 得到E
    def getE(self, i):
        fXi = np.dot(self.alpha * self.labels, self.kernel_mat[i, :].T) + self.B
        return fXi - self.labels[i]
    
    # 更新上界和下界
    def getHAndL(self, i, j):
        H = min(self.C, self.C + self.alpha[j] - self.alpha[i]) if self.labels[i] != self.labels[j] else min(self.C, self.alpha[j] + self.alpha[i])
        L = max(0, self.alpha[j] - self.alpha[i]) if self.labels[i] != self.labels[j] else max(0, self.alpha[j] + self.alpha[i] - self.C)
        return H, L
    
    # 更新第二个alpha
    def updateAlphaJ(self, i, j, H, L, Ei, Ej):
        Yi, Yj = self.labels[i] , self.labels[j]
        Kii, Kij, Kjj = self.kernel_mat[i, i], self.kernel_mat[i, j], self.kernel_mat[j, j]
        Hj = H
        Lj = L
        eta = Kii + Kjj - 2 * Kij
        if eta > 0:
            alpha_j = self.alpha[j] + Yj * (Ei - Ej) / eta
            if Lj <= alpha_j and alpha_j <= Hj:
                self.alpha[j] = alpha_j
            elif alpha_j < Lj:
                self.alpha[j] = Lj
            else:
                self.alpha[j] = Hj
        else:
            s = Yi * Yj
            Hi = self.alpha[i] + s * (self.alpha[j] - Hj)
            Li = self.alpha[i] + s * (self.alpha[j] - Lj)
            fi = Yi * (Ei - self.B) - self.alpha[i] * Kii - s * self.alpha[j] * Kij
            fj = Yj * (Ej - self.B) - self.alpha[j] * Kjj - s * self.alpha[i] * Kij
            psiL = Li * fi + Lj * fj + 1 / 2 * (Li ** 2) * Kii + 1 / 2 * (Lj ** 2) * Kjj + s * Li * Lj * Kij
            psiH = Hi * fi + Hj * fj + 1 / 2 * (Hi ** 2) * Kii + 1 / 2 * (Hj ** 2) * Kjj + s * Hi * Hj * Kij
            if psiL < psiH:
                self.alpha[j] = Lj
            else:
                self.alpha[j] = Hj
    
    # 更新第一个alpha
    def updateAlphaI(self, i, j, old_alphaj):
        self.alpha[i] = self.alpha[i] + self.labels[i] * self.labels[j] * (self.alpha[j] - old_alphaj)
    
    # 更新B
    def updateB(self, i, j, old_alphai, old_alphaj, Ei, Ej):
        Yi, Yj = self.labels[i] , self.labels[j]
        Kii, Kij, Kjj = self.kernel_mat[i, i], self.kernel_mat[i, j], self.kernel_mat[j, j]
        b1 = self.B - Yi * Kii * (self.alpha[i] - old_alphai) - Yj * Kij * (self.alpha[j] - old_alphaj) - Ei
        b2 = self.B - Yj * Kjj * (self.alpha[j] - old_alphaj) - Yi * Kij * (self.alpha[i] - old_alphai) - Ej
        if 0 < self.alpha[i] and self.alpha[i] < self.C:
            self.B = b1
        elif 0 < self.alpha[j] and self.alpha[j] < self.C:
            self.B = b2
        else:
            self.B = (b1 + b2) / 2
    
    # 判断alpha是否在边界上
    def isBoundary(self, i):
        return self.alpha[i] <= 0 or self.alpha[i] >= self.C
    
    # 随机选择第二个参数
    def selectJRandom(self, i):
        j = i
        while j == i:
            j = int(random.uniform(0, self.N))
        return j
    
    # 选择第二个参数
    def selectJ(self, i, Ei):
        self.eCache[i] = [1, Ei]
        valid_indexs = np.nonzero(self.eCache[:, 0])[0]
        best_j, best_abs, best_Ej = -1, 0, 0
        if len(valid_indexs) > 1:
            for j in valid_indexs:
                if i == j:
                    continue
                Ej = self.getE(j)
                if abs(Ej - Ei) > best_abs:
                    best_abs = abs(Ej - Ei)
                    best_j = j
                    best_Ej = Ej
        else:
            best_j = self.selectJRandom(i) 
            best_Ej = self.getE(best_j)
        return best_j, best_Ej

    # 内层循环
    def innerLoop(self, i):
        Ei = self.getE(i)
        Yi = self.labels[i]
        if (Yi * Ei < -self.tol and self.alpha[i] < self.C) or (Yi * Ei > self.tol and self.alpha[i] > 0):
            best_j, best_Ej = self.selectJ(i, Ei)
            if best_j == -1:
                return 0
            H, L = self.getHAndL(i, best_j)
            old_alphaj = self.alpha[best_j]
            self.updateAlphaJ(i, best_j, H, L, Ei, best_Ej)
            if abs(self.alpha[best_j] - old_alphaj) < self.alpha_tol:
                return 0
            old_alphai = self.alpha[i]
            self.updateAlphaI(i, best_j, old_alphaj)
            self.updateB(i, best_j, old_alphai, old_alphaj, Ei, best_Ej)
            self.updateFAndE(i, best_j)
            return 1
        return 0
    
    # 外层循环
    def outerLoop(self):
        entire_set = True
        alpha_is_changed = 0
        num = 0
        while num < self.iter_num and (alpha_is_changed > 0 or entire_set):
            alpha_is_changed = 0
            if entire_set:
                for i in range(self.N):
                    alpha_is_changed += self.innerLoop(i)
            else:
                set_no_boundry = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0].tolist()
                for i in set_no_boundry:
                    alpha_is_changed += self.innerLoop(i)
            if entire_set:
                entire_set = False
            elif alpha_is_changed == 0:
                entire_set = True
            num += 1
            # print("Train %d : Update %d alpha-pairs"%(num, alpha_is_changed)) # debug
    
    def train(self):
        self.outerLoop()

    # 预测
    def predict(self, x):
        res = -1 if self.f(x) <= 0 else 1
        for key, val in self.counts.items():
            if val == res:
                return key
        return None