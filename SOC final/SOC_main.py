import numpy as np
import pandas as pd
import MyWDNFuns
from CwGN_EPACalibrator import CwCalibrator
from myEPANETtoolkit import ENepanet
from KMEANS import K_Means
from plotfigure import plotfigure
from scipy.spatial.distance import cdist
import os

class quickgrouping:
    def __init__(self, filename, savepath, C8, n_clusters, pipe_age, use_material=True,
                 use_age=True, use_diameter=True, use_location=False, pipe_material=None):
        self.filename = filename
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        self.savepath = savepath
        self.plotter = plotfigure(self.filename, savepath=self.savepath)
        self.EN = ENepanet()
        self.EN.ENopen(inpfile=self.filename, rptfile='temp.rpt')

        self.WDNInfo = MyWDNFuns.GetWDNInformation(filename=self.filename)
        self.C8 = C8
        self.pipe_material = pipe_material
        self.pipe_age = pipe_age
        self.pipe_d = self.WDNInfo.d
        self.pipe_coor = self.WDNInfo.pipeCoor
        self.n_clusters = n_clusters
        self.I = np.eye(self.n_clusters)
        self.attr_use = np.array([use_material, use_age, use_diameter, use_location])
        self.delete_Index = np.where(self.attr_use == False)
        self.k = K_Means(n_clusters=self.n_clusters)

    def SetMeasurements(self, kq, numofkq, kH, numofkH, H0, tankLevel, WkH=None, Wkq=None):
        self.kq = kq
        self.numofkq = numofkq
        self.kH = kH
        self.numofkH = numofkH
        self.H0 = H0
        self.tankLevel = tankLevel
        if WkH is None:
            self.WkH = np.ones(len(kH))
        else:
            self.WkH = WkH
        if Wkq is None:
            self.Wkq = np.ones(len(kq)) * 0.001
        else:
            self.Wkq = Wkq

    def weight_distance(self, Data, centers, w=None):
        if w is None:
            w = np.ones(Data.shape[1])
        d_respective = np.empty(Data.shape[1], dtype=object)
        for i in range(Data.shape[1]):
            if isinstance(Data[:, i][0], np.ndarray):
                d_respective[i] = w[i] * cdist(np.array(Data[:, i].tolist()), np.array(centers[:, i].tolist())) ** 2
            else:
                d_respective[i] = w[i] * cdist(Data[:, i].reshape(len(Data[:, i]), 1),
                                               centers[:, i].reshape(len(centers[:, i]), 1)) ** 2
        d_weighted = np.sqrt(d_respective.sum())
        return d_weighted

    def Normalization(self, D):
        self.D_standard = np.empty((D.shape[0], D.shape[1]), dtype=object)
        for attr in range(D.shape[1]):
            if isinstance(D[0, attr], np.ndarray):
                attr_temp = np.array(D[:, attr].tolist())
                range_all = attr_temp.max(axis=0) - attr_temp.min(axis=0)
                scale_factor = range_all.max()
                attr_standard = (attr_temp - attr_temp.min(axis=0)) / scale_factor
                attr_standard = [np.array(x) for x in attr_standard]
                self.D_standard[:, attr] = attr_standard
            else:
                self.D_standard[:, attr] = (D[:, attr] - D[:, attr].min()) / (D[:, attr].max() - D[:, attr].min())
        return self.D_standard

    def get_sen_data(self):
        self.EN.ENsolveH()
        h = np.array(self.EN.ENgetmultiparams('Link', (range(1, len(self.WDNInfo.linkID) + 1)), 10))
        q = np.array(self.EN.ENgetmultiparams('Link', (range(1, len(self.WDNInfo.linkID) + 1)), 8))
        Cw_now = np.array(self.EN.ENgetmultiparams('Link', (range(1, len(self.WDNInfo.linkID) + 1)), 2))
        if h.size > Cw_now.size:
            addCw = -1 * np.ones(h.size - Cw_now.size)
            useCw = np.append(Cw_now, addCw)
        else:
            useCw = Cw_now
        J = MyWDNFuns.GetJacobian(h, q, self.WDNInfo.d, useCw, self.WDNInfo.linkType, self.WDNInfo.A, self.C8)
        JHC = J.HpartialC
        JqC = J.qpartialC
        JmHC = JHC.take((self.numofkH - 1).astype(np.int64), axis=0).T
        JmqC = JqC.take((self.numofkq - 1).astype(np.int64), axis=0).T
        senC = np.hstack((JmHC, JmqC))
        senCH = JmHC * np.repeat(self.WkH.reshape(1, len(self.WkH)), len(self.WDNInfo.linkID), axis=0)
        senCq = JmqC * np.repeat(self.Wkq.reshape(1, len(self.Wkq)), len(self.WDNInfo.linkID), axis=0)
        sumsenCH = abs(senCH).sum(axis=1)
        sumsenCq = abs(senCq).sum(axis=1)
        sumsenC = sumsenCH + sumsenCq
        return senC, sumsenC

    def w_trainer(self, train_mode=1):
        if train_mode == 0:
            self.w = np.ones(3)
        if train_mode == 1:
            wi = np.random.rand(3)
            wi[0] = np.random.uniform(0.6, 0.7)
            wi[1] = np.random.uniform(0.3, 0.4)
            wi[2] = np.random.uniform(0, 0.2)
            self.w = wi / wi.sum() * 3
        if train_mode == 2:
            self.w = np.empty(3)
            self.w[0] = 1.8
            self.w[1] = 1
            self.w[2] = 0.2

    def first_cluster_method(self, sumsenC):
        # self.sensitivity_cluster_data_index = np.where(sumsenC >= 0)[0]
        self.sensitivity_cluster_data_index = np.sort(np.argsort(sumsenC)[-50:])
        self.attr_cluster_data_index = np.array(range(len(self.WDNInfo.linkID)))
        self.attr_cluster_data_index = np.delete(self.attr_cluster_data_index, self.sensitivity_cluster_data_index)
        self.sensitivity_cluster_data_labels = np.array(self.WDNInfo.linkID)[self.sensitivity_cluster_data_index]
        self.senattrData = self.D_standard[self.sensitivity_cluster_data_index, :]
        self.attrData = self.D_standard[self.attr_cluster_data_index, :]
        self.first_group_matrix = self.k.fit(self.senattrData, centerschoicemethod=2, w=self.w, runtimes=100)
        self.sen_grouplabels = np.argmax(self.first_group_matrix, axis=1)
        self.centersforattr = np.sum(self.senattrData[:, None, :] * self.first_group_matrix[:, :, None], axis=0) / (
            np.sum(self.first_group_matrix, axis=0)[:, None])
        self.groupcount = np.sum(self.first_group_matrix, axis=0)

    def refershcenter(self, group, data):
        refreshcenter = self.centers[group, :]
        newcenter = (refreshcenter * self.groupcount[group] + data) / (self.groupcount[group] + 1)
        self.groupcount[group] = self.groupcount[group] + 1
        self.centers[group, :] = newcenter

    def second_cluster_method(self):
        finallabels = np.array(range(self.WDNInfo.linkType.count(1)))
        for i in range(len(self.sen_grouplabels)):
            finallabels[self.sensitivity_cluster_data_index[i]] = self.sen_grouplabels[i]
        final_group, final_group_matrix = self.k.fit2(self.attrData, self.attr_cluster_data_index,
                                                      self.D_standard, self.centersforattr,
                                                      finallabels, w=self.w)
        return final_group, final_group_matrix


if __name__ == '__main__':
    filename = 'Network/Mudu.inp'
    savepath = 'result\\'
    WDNInfo = MyWDNFuns.GetWDNInformation(filename=filename)
    EN = ENepanet()
    EN.ENopen(inpfile=filename, rptfile='temp.rpt')
    EN.ENsolveH()
    C_real = np.array(EN.ENgetmultiparams('Link', (range(1, len(WDNInfo.linkID) + 1)), 2))
    C8 = np.array([])
    tankLevel = np.array([])
    H0 = np.array([])
    presetCk = np.ones(WDNInfo.linkType.count(1)) * 115
    numofkq = np.array(
        [7, 14, 39, 43, 47, 49, 70, 82, 137, 148, 155, 218, 256, 275, 342, 359, 393, 446, 486, 510, 515, 545, 554, 565,
         566, 567])
    numofkH = np.array(
        [1, 20, 22, 48, 51, 55, 77, 96, 129, 151, 158, 173, 185, 206, 211, 216, 224, 236, 253, 272, 298, 308, 313, 321,
         331, 355, 365, 368, 408, 436, 458])

    pipe_data = pd.read_csv('Data/Mudu_pipe.csv', encoding='utf-8')
    pipe_material = pipe_data['material_type'].values
    pipe_age = pipe_data['year'].values
    kq = pd.read_csv('Data/kq.csv', encoding='utf-8')['kq'].values
    kH = pd.read_csv('Data/kH.csv', encoding='utf-8')['kH'].values
    demands = np.array(EN.ENgetmultiparams('Node', (range(1, len(WDNInfo.nodeID) + 1)), 1))
    calibCw = CwCalibrator(ENdll=EN, itermethod=1)
    monitorsq = np.array(WDNInfo.linkID)[numofkq - 1]
    monitorsH = np.array(WDNInfo.nodeID)[numofkH - 1]
    plotter = plotfigure(filename, savepath=savepath, monitor_q=monitorsq, monitor_H=monitorsH)
    Wkq = 1 / (np.abs(kq) * 0.02 / 3.27) * 0.00001  # the default weight is the inverse of its uncertainty
    WkH = 1 / (np.ones(kH.shape[0]) * 0.1) * 1
    W = np.diag(np.append(WkH, Wkq))
    sigma2 = np.diag(1 / (np.array([0.1] * len(numofkH) + list(0.02 / 3.27 * np.abs(kq))) ** 2))
    n_clusters = 10
    quickG = quickgrouping(filename, savepath, C8, n_clusters, pipe_age, pipe_material=pipe_material)
    Data = np.empty((len(pipe_material), 4), dtype=object)
    Data[:, 0] = quickG.pipe_material
    Data[:, 1] = quickG.pipe_age
    Data[:, 2] = quickG.pipe_d
    Data[:, 3] = [np.array(i) for i in quickG.pipe_coor.tolist()]
    Data = np.delete(Data, quickG.delete_Index, axis=1)
    quickG.Normalization(Data)
    quickG.SetMeasurements(kq, numofkq, kH, numofkH, H0, tankLevel, Wkq=Wkq, WkH=WkH)
    senC, sumsenC = quickG.get_sen_data()
    failtimes = 0
    flag = False
    cflag = True
    count = 0
    while flag is False:
        try:
            print('第' + str(count + 1) + '次...')
            if count == 0:
                quickG.w_trainer(train_mode=2)
            else:
                quickG.w_trainer(train_mode=1)
            print('Clustering weights are: ')
            print(quickG.w)
            quickG.first_cluster_method(sumsenC)
            grouplabels, group = quickG.sen_grouplabels, quickG.first_group_matrix
            grouplabels, group = quickG.second_cluster_method()
            demandsuse = demands
            calibCw.SetInpData(demandsuse, kq, numofkq, kH, numofkH, W, tankLevel, WDNInfo.tankID,
                               WDNInfo.nodeID, WDNInfo.linkID, WDNInfo.linkType, WDNInfo.A, WDNInfo.d, C8,
                               presetCk, Gc=group, creal=C_real)
            calibCw.Calibrator(dertaLim=0.1, iterLim=500)
            if calibCw.errcode == 0:
                calibCw.RefreshJacobian()
                flag = True
                break
            else:
                print(calibCw.errcode)
                count += 1
                if count > 1000:
                    cflag = False
                    failtimes += 1
                    break
        except:
            print('error!!!')
            break
    if cflag == True:
        Cwresult = calibCw.ratioGc.T @ calibCw.Cw.T
        groupCRecorder = calibCw.ratioGc.T @ calibCw.CwRecorder.T
        absdertaC = abs(calibCw.deltaCwi)
        print('Grouping number is: %d' % quickG.n_clusters)
        print('Cw calibration results are: ')
        print(Cwresult)
        print('Calibration error is：%.04f' % calibCw.err[-1])
        print('Number of pipes with error less than 10 is: %d' % np.where(abs(calibCw.deltaCwi) < 10)[0].shape[0])
        print('Error of pressure monitors are: ')
        print([round(i, 4) for i in np.array(calibCw.derta)[0: len(numofkH)]])
        print('Mean Error of pressure monitors is: %.02f' % np.mean(np.array(calibCw.derta)[0: len(numofkH)]))
        print('%Error of Flow monitors are: ')
        print([round(i, 4) for i in np.array(calibCw.derta)[len(numofkH):] / kq * 100])
        print('Mean %%Error of Flow monitors is: %.02f' % np.mean(np.array(calibCw.derta)[len(numofkH):] / kq * 100))
        print('Mean Cw Error is: %.02f' % np.mean(absdertaC))
        print('Max Cw Error is: %.02f' % np.max(absdertaC))
        plotter.plot_pipe_group(pipeindex=np.array(WDNInfo.linkID)[np.where(np.array(WDNInfo.linkType) == 1)],
                                pipelabels=grouplabels,
                                # nodeindex= np.array(WDNInfo.nodeID),
                                # nodelabels=WDNInfo.nodeID,
                                savepath=savepath,
                                )
        plotter.plotcalibresult_all(groupCRecorder, savepath=savepath)
        plotter.plotC_err_distribution(calibCw.deltaCwi, savepath=savepath)
    else:
        print('grouping is failed!!')
