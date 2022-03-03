import numpy as np
import pandas as pd
import MyWDNFuns
from myEPANETtoolkit import ENepanet

class CwCalibrator:
    def __init__(self,filename = None, ENdll = None, itermethod=1):
        self.itermethod = itermethod
        if ENdll == None:
            if not filename == None:
                self.EN = ENepanet()
                self.EN.ENopen(filename,'temp.rpt')
            else:
                print('没有输入管网！')
                pass
        else:
            self.EN = ENdll
        
    # set initCw to the model
    def SetInpData(self,demands,kq,numofkq,kH,numofkH,W,tankLevel,tankID,nodeID,linkID,linkType,A,d,C8,initCw,
                   KnownCw_index=None, Gc=None, creal=None):
        if creal is not None:
            self.creal = creal
        self.demands = demands
        self.kq = kq
        self.numofkq = numofkq
        self.kH = kH
        self.numofkH = numofkH
        self.tankLevel = tankLevel
        self.tankID = tankID
        self.nodeID = nodeID
        self.linkID = linkID
        self.linkType = linkType
        self.A = A
        self.d = d
        self.C8 = C8
        self.known_linkC = np.zeros(len(self.linkID))
        if KnownCw_index is not None:
            self.known_linkC[KnownCw_index] = 1
        self.CwRecorder = initCw
        self.Cw = initCw
        self.W = W / np.mean(np.diag(W))
        # calculate the ratio of Gc to allocate pipe roughness after group roughness are calculated
        if Gc is not None:
            self.Gc = Gc.astype(np.float64)
            sumGc = Gc.sum(0)
            ratioGc = self.Gc.copy()
            for i in range(len(sumGc)):
                ratioGc[:,i] = ratioGc[:,i] / sumGc[i]
            self.ratioGc = ratioGc

        #initialize demands
        for i in range(len(nodeID)):
            self.EN.ENsetnodevalue(i+1, 1, demands[i])
                    
        #initialize roughness
        self.SetRoughness(self.Cw)

        #initialize tanklevel
        for i in range(len(tankID)):
            useIndex = self.EN.ENgetnodeindex(tankID[i])
            self.EN.ENsetnodevalue(useIndex,8,tankLevel[i])

    def RefreshJacobian(self, h = [], q = [], Cw = []):
        if len(h) == 0:
            # when input nothing, create Jacobian with current data
            self.SetRoughness(self.Cw)
            self.EN.ENsolveH()
            h = np.array(self.EN.ENgetmultiparams('Link',(range(1,len(self.linkID)+1)),10))
            q = np.array(self.EN.ENgetmultiparams('Link',(range(1,len(self.linkID)+1)),8))
            if h.size > self.Cw.size:
                addCw = -1 * np.ones(h.size-self.Cw.size)
                useCw = np.append(self.Cw,addCw)
            else:
                useCw = self.Cw.copy()
            J = MyWDNFuns.GetJacobian(h,q,self.d,useCw,self.linkType,self.A,self.C8)
        else:
            if h.size > Cw.size:
                addCw = -1 * np.ones(h.size-self.Cw.size)
                useCw = np.append(self.Cw,addCw)
            else:
                useCw = Cw
            # use the inputs to calculate new jacobian
            J = MyWDNFuns.GetJacobian(h,q,self.d,useCw,self.linkType,self.A,self.C8)
        JHC = (J.HpartialC[:,:self.Gc.shape[0]]) @ self.Gc
        JHC = JHC.take(self.numofkH-1,axis = 0)
        JqC = (J.qpartialC[:,:self.Gc.shape[0]]) @ self.Gc
        if len(self.numofkq)==0:
            self.JHCJqC=JHC
        else:
            JqC = JqC.take(self.numofkq-1,axis = 0)
            self.JHCJqC = np.append(JHC,JqC,axis = 0)
        
    
    def SetRoughness(self,Cw):
        for i in range(len(self.linkID)):
            if self.linkType[i] == 1:
                self.EN.ENsetlinkvalue(i+1, 2, Cw[i])
     
    
    def Runiter(self,Cw,kH = [],kq = []):
        #set default kH and kq
        if len(kH) == 0:
            kH = self.kH
        if len(kq) == 0:
            kq = self.kq
        # set initial roughness
        self.SetRoughness(Cw)
        self.EN.ENsolveH()
        calq = np.array(self.EN.ENgetmultiparams('Link',self.numofkq.tolist(),8))
        calH = np.array(self.EN.ENgetmultiparams('Node',self.numofkH.tolist(),11))
        derta = np.append((self.kH-calH), (self.kq-calq)) # error between simulated and monitored value
        err = ((self.W @ derta) ** 2).sum()
        JTJ = self.JHCJqC.T @ self.W @ self.JHCJqC
        if self.itermethod == 0:
            dertaCw = (np.linalg.inv(JTJ)) @ (self.JHCJqC.T @ self.W @ derta)
            dertaCw = self.ratioGc @ dertaCw
            iterCw = Cw + dertaCw
            iterDerta = np.linalg.norm(dertaCw, ord=2)
        else:
            if err < self.err0:
                self.err0 = err
                self.damp = self.damp / self.scaling
            else:
                Cw = Cw - self.dertaCw0
                self.damp = self.damp * self.scaling
            dertaCw = (np.linalg.inv(JTJ + np.diag(np.diag(JTJ)) * self.damp)) @ (self.JHCJqC.T @ self.W @ derta)
            dertaCw = self.ratioGc @ dertaCw
            iterCw = Cw + dertaCw
            iterDerta = np.linalg.norm(dertaCw, ord=2)
            self.dertaCw0 = dertaCw
        return iterCw, iterDerta, err, derta

    def Calibrator(self, dertaLim = 0.3, iterLim = 20, JacobRefreshRate = 1000,\
        CwUpperLim = 150, CwLowerLim = 50, damp=0.01, scaling=5, err0=1000000, dertaCw0 = 10): # define limits to stop iteration
        iterDerta = 10
        iterCounter = 0
        self.errcode = 0
        self.RefreshJacobian()
        self.err = []
        self.deltaCw = []
        self.damp = damp
        self.scaling = scaling
        self.err0 = err0
        self.dertaCw0 = dertaCw0
        while iterDerta > dertaLim and iterCounter < iterLim:
            try:
                self.Cw, iterDerta, err, self.derta = self.Runiter(self.Cw,self.kH,self.kq)
                self.deltaCwi = self.Cw - self.creal
                self.deltaCw.append(np.mean(abs(self.deltaCwi)))
                self.err.append(err)
            except:
                self.errcode = 3 # singular matrix!!
                # print('invalid grouping!!!!')
                break 
            self.CwRecorder = np.vstack([self.CwRecorder, self.Cw])
            iterCounter += 1
            if iterCounter % JacobRefreshRate == 0: #refresh jacobian with stated refresh rate
                self.RefreshJacobian()
            if np.sum(np.isnan(self.Cw)) > 0:
                self.errcode = 1 # errcode1: appear nan results
                break
        if self.errcode == 0:
            print(iterCounter)
            if np.max(self.Cw) > CwUpperLim or np.min(self.Cw) < CwLowerLim:
                self.errcode = 2 # errcode2: appear unreasonable results
