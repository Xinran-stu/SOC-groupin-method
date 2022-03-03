import numpy as np
import wntr
from scipy.stats.distributions import norm
from pyDOE import lhs
class GetJacobian:
    # ref: Liu, N., Du, K., Tu, J., Dong, W., 2017. Analytical solution of Jacobian matrices of WDS models. Procedia Eng. 186, 388-396.
    def __init__(self,h,q,d,C,linkType,Ai,C8):
        A = Ai.copy()
        for i in range(q.shape[0]):
            if q[i] < 0:
                A[:,i] = -A[:,i]
        C = C.astype(np.float64)
        C += 1e-10
        q = np.abs(q)
        h = np.abs(h)
        linkCount = len(q)
        B = np.zeros([linkCount])
        s = np.zeros([linkCount]) + 1e-10
        D = np.zeros([linkCount])
        q += 1e-10
        h += 1e-10
        pumpCount = 0
        for i in range(linkCount):
            if linkType[i] == 1:
                B[i] = q[i] / (1.852*h[i])
                s[i] = q[i] / C[i]
                D[i] = (4.871*q[i]) / (1.852*d[i])
            elif linkType[i] == 2:
                C[i],d[i],s[i],D[i] = -1,-1,0,0
                B[i] = 1 / (2 * C8[pumpCount] * np.abs(q[i]))
                pumpCount += 1
        B,s,D = np.diag(B), np.diag(s), np.diag(D)
        invABAT = np.linalg.inv(A @ B @ A.T)
                
        self.HpartialQ = -invABAT
        self.qpartialQ = B @ A.T @ invABAT
        self.HpartialC = invABAT @ A @ s
        self.qpartialC = s - B @ A.T @ invABAT @ A @ s
        self.Hpartiald = invABAT @ A @ D
        self.qpartiald = D - B @ A.T @ invABAT @ A @ D

        self.drawCwLocate = [81,86,110,114,68,62,33,11,4,1]
        self.drawuQLocate = [24,32,18,9,87,39,49,58,70,72]

class GetWDNInformation:
    def __init__(self,filename = None, wntrFile = None):
        if wntrFile == None:
            if not filename == None:
                wn = wntr.network.WaterNetworkModel(filename)
            else:
                print('没有输入管网！')
                pass
        else:
            wn = wntrFile
        #basic parameters
        self.allnodeCount = wn.num_nodes
        self.linkCount = wn.num_links
        self.tankCount = wn.num_tanks
        self.reservoirCount = wn.num_reservoirs
        self.allNodeIndex = range(1,self.allnodeCount+1)
        self.linkIndex = range(1,self.linkCount+1)
        self.allNodeID = wn.node_name_list
        self.linkID = wn.link_name_list
        self.nodeID = list(wn.nodes.junction_names)
        self.reservoirID = list(wn.nodes.reservoir_names)
        self.tankID = list(wn.nodes.tank_names)
        self.nodeCount = self.allnodeCount-self.tankCount-self.reservoirCount
        self.nodeIndex = []
        for nID in self.nodeID:
            self.nodeIndex.append(self.allNodeID.index(nID)+1)
        self.tankIndex = []
        for tID in self.tankID:
            self.tankIndex.append(self.allNodeID.index(tID)+1)
        self.reservoirIndex = []
        for rID in self.reservoirID:
            self.reservoirIndex.append(self.allNodeID.index(rID)+1)
        self.pumpID = list(wn.links.pump_names)
        self.pumpIndex = []
        for pID in self.pumpID:
            self.pumpIndex.append(self.linkID.index(pID)+1)
        #topology
        self.startNodeID = []
        self.endNodeID = []
        self.startNodeIndex = []
        self.endNodeIndex = []
        self.A = np.zeros([self.nodeCount,self.linkCount])
        self.A10 = np.zeros([self.tankCount + self.reservoirCount,self.linkCount])
        for iLink in self.linkID:
            useLink = wn.get_link(iLink)
            self.startNodeID.append(useLink.start_node_name)
            self.endNodeID.append(useLink.end_node_name)
            self.startNodeIndex.append(self.allNodeID.index(useLink.start_node_name)+1)
            self.endNodeIndex.append(self.allNodeID.index(useLink.end_node_name)+1)
        for iLink in range(self.linkCount):
            for iNode in range(self.nodeCount):
                if self.startNodeIndex[iLink] == self.nodeIndex[iNode]:
                    self.A[iNode,iLink] = -1
                elif self.endNodeIndex[iLink] == self.nodeIndex[iNode]:
                    self.A[iNode,iLink] = 1
            for iNode in range(self.reservoirCount):
                if self.startNodeIndex[iLink] == self.reservoirIndex[iNode]:
                    self.A10[iNode,iLink] = -1
                elif self.endNodeIndex[iLink] == self.reservoirIndex[iNode]:
                    self.A10[iNode,iLink] = 1
            for iNode in range(self.reservoirCount,self.tankCount + self.reservoirCount):
                if self.startNodeIndex[iLink] == self.tankIndex[iNode - self.reservoirCount]:
                    self.A10[iNode,iLink] = -1
                elif self.endNodeIndex[iLink] == self.tankIndex[iNode - self.reservoirCount]:
                    self.A10[iNode,iLink] = 1
        # other parameters
        pipeID = list(wn.links.pipe_names)
        pumpID = list(wn.links.pump_names)
        self.linkType = list(range(self.linkCount))
        for ID in pipeID:
            self.linkType[self.linkID.index(ID)] = 1
        for ID in pumpID:
            self.linkType[self.linkID.index(ID)] = 2
        self.Cw = np.zeros(self.linkCount)
        self.d = np.zeros(self.linkCount)
        self.L = np.zeros(self.linkCount)
        self.Height = np.zeros(self.nodeCount)
        self.tankHeight = np.zeros(self.tankCount)
        Cw = wn.query_link_attribute('roughness')
        d = wn.query_link_attribute('diameter')
        L = wn.query_link_attribute('length')
        Height = wn.query_node_attribute('elevation')
        for iLink in range(self.linkCount):
            if self.linkType[iLink] == 1:
                self.Cw[iLink] = (Cw.loc[self.linkID[iLink]])
                self.d[iLink] = (d.loc[self.linkID[iLink]]) * 1000
                self.L[iLink] = (L.loc[self.linkID[iLink]])
            if self.linkType[iLink] == 2:
                self.Cw[iLink] = -self.linkIndex[iLink]
                self.d[iLink] = 1000
                self.L[iLink] = 1000
        for iNode in range(self.nodeCount):
            self.Height[iNode] = Height.loc[self.nodeID[iNode]]
        for iNode in range(self.tankCount):
            self.tankHeight[iNode] = Height.loc[self.tankID[iNode]]
        self.nodeCoor = wn.query_node_attribute('coordinates')
        self.nodeCoor = self.nodeCoor.apply(lambda x: np.array(x))
        self.startCoor = self.nodeCoor[self.startNodeID].values
        self.endCoor = self.nodeCoor[self.endNodeID].values
        self.pipeCoor = (self.startCoor + self.endCoor) / 2
        self.nodeCoor = self.nodeCoor.values

def LHS_std(x,stdx,numofSample):
    design = lhs(x.shape[0], samples = numofSample)
    for i in range (x.shape[0]):
        # loc: 均值；scale:标准差
        # norm.ppf: 变换为正态分布
        design[:,i] = norm(loc = x[i], scale = stdx[i]).ppf(design[:,i])
    return design.T

def InverseG(G):
    inverseG = G
    inverseG[inverseG != 0] = 1
    return inverseG
