import numpy as np
from scipy.spatial.distance import cdist

class K_Means():
    # n_clusters:是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, sensitivity_cluster_data_index=None, filename=None, n_clusters=5, tolerance=0.0000001, max_iter=300):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.sensitivity_cluster_data_index = sensitivity_cluster_data_index

    def Normalization(self, D):
        D_standard = np.empty((D.shape[0], D.shape[1]), dtype=object)
        for attr in range(D.shape[1]):
            if isinstance(D[0, attr], np.ndarray):
                attr_temp = np.array(D[:, attr].tolist())
                range_all = attr_temp.max(axis=0) - attr_temp.min(axis=0)
                scale_factor = range_all.max()
                attr_standard = (attr_temp - attr_temp.min(axis=0)) / scale_factor
                attr_standard = [np.array(x) for x in attr_standard]
                D_standard[:, attr] = attr_standard
            else:
                D_standard[:, attr] = (D[:, attr] - D[:, attr].min()) / (D[:, attr].max() - D[:, attr].min())
        return D_standard
    
    def weight_distance(self, Data, centers, w):
        d_respective = np.empty(Data.shape[1], dtype=object)
        for i in range(Data.shape[1]):
            try:
                if isinstance(Data[:, i][0], np.ndarray):
                    d_respective[i] = w[i] * cdist(np.array(Data[:,i].tolist()), np.array(centers[:,i].tolist())) ** 2
                else:
                    d_respective[i] = w[i] * cdist(Data[:,i].reshape(len(Data[:,i]), 1), centers[:,i].reshape(len(centers[:,i]), 1)) ** 2
            except:
                pass
        d_weighted = np.sqrt(d_respective.sum())
        return d_weighted

    def fit(self, Data, choiceforcenters=[], centerschoicemethod=2, w=None, first_choice=None, runtimes = 1):
        if w is None:
            w = np.ones(Data.shape[1])
        # 归一化
        self.I = np.eye(self.n_clusters)
        self.mingroupd = 1e+10
        for gtimes in range(runtimes):
            if choiceforcenters==[]:
                Dataforchoice = Data.copy()
            else:
                Dataforchoice = Data[choiceforcenters]
            if centerschoicemethod == 1: # kmeans centers
                centers = Data[np.random.choice(len(Data), self.n_clusters, replace=False)]
            if centerschoicemethod == 2: # kmeans++ centers
                if first_choice is None:
                    firstchoice = np.random.choice(len(Dataforchoice), 1)
                else:
                    firstchoice = first_choice
                centers = Dataforchoice[firstchoice]
                Dataforchoice = np.delete(Dataforchoice, firstchoice, axis=0)
                for _ in range(1, self.n_clusters):
                    d2centers = self.weight_distance(Dataforchoice, centers, w)
                    min_d2centers = d2centers.min(axis=1)
                    choice = np.argmax(min_d2centers)
                    centers = np.vstack((centers, Dataforchoice[choice]))
                    Dataforchoice = np.delete(Dataforchoice, choice, axis=0)
            if centerschoicemethod == 3: # simple choice
                centers = Dataforchoice[np.array(range(0,self.n_clusters))]
            for _ in range(self.max_iter):
                prev_centers = centers.copy()
                d_weighted = self.weight_distance(Data, centers, w)
                cluster_index_num = np.argmin(d_weighted, axis=1)
                cluster_index = self.I[cluster_index_num]
                centers = np.sum(Data[:, None, :] * cluster_index[:, :, None], axis=0)/(np.sum(cluster_index, axis=0)[:, None]+1e-10)
                div_centers = centers - prev_centers
                optimized = True
                for i in range(div_centers.shape[0]):
                    if np.sum([np.linalg.norm(j) for j in div_centers[i]]) > self.tolerance:
                        optimized = False
                        break
                if optimized:
                    break
            mean_group_d = []
            for j in range(self.n_clusters):
                mean_group_di = np.mean(d_weighted[np.where(cluster_index_num == j), j])
                mean_group_d.append(mean_group_di)
            groupd = np.mean(np.array(mean_group_d))
            if groupd < self.mingroupd:
                self.mingroupd = groupd
                self.cluster_index = cluster_index
                self.centers = centers
        return self.cluster_index

    def fit2(self, Data, Dataindex, all_Data, centers, labels, w=None):
        # Data: LSPs; Dataindex: Pipe id for LSPs; all_Data: HSPs and LSPs
        # centers: HSPs group centers; labels:group labels for all pipes, containing HSPs already
        if w is None:
            w = np.ones(Data.shape[1])
        for _ in range(self.max_iter):
            prev_centers = centers.copy()
            d_weighted = self.weight_distance(Data, centers, w)
            cluster_index_num = np.argmin(d_weighted, axis=1)
            for i in range(len(cluster_index_num)):
                labels[Dataindex[i]] = cluster_index_num[i]
            cluster_index = self.I[labels]
            centers = np.sum(all_Data[:, None, :] * cluster_index[:, :, None], axis=0) / (
                        np.sum(cluster_index, axis=0)[:, None] + 1e-10)
            div_centers = centers - prev_centers
            optimized = True
            for i in range(div_centers.shape[0]):
                if np.sum([np.linalg.norm(j) for j in div_centers[i]]) > self.tolerance:
                    optimized = False
                    break
            if optimized:
                break
        return labels, cluster_index
    
    

