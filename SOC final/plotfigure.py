import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from pylab import mpl
import pandas as pd
import wntr
import numpy as np
import matplotlib
import matplotlib.colors as colors

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.switch_backend('agg')

class plotfigure:
    def __init__(self, filename=None, savepath='', monitor_q=np.array([]), monitor_H=np.array([])):
        self.filename = filename
        self.wn = wntr.network.WaterNetworkModel(filename)
        self.monitor_q = monitor_q
        self.monitor_H = monitor_H
        self.savepath = savepath

    def plot_pipe_group(self, pipeindex=None, G=None, pipelabels=None, savepath=None, Herr=None, qerr=None):
        if pipelabels is not None:
            labels = pipelabels
        elif G is not None:
            labels = np.argmax(G, axis=1)
        else:
            print('No grouping labels!!')
        if savepath is None:
            savepath = self.savepath
        font = {'family': 'Arial', 'size': 9}
        pipe_labels = pd.Series(data=labels, index=pipeindex.tolist())
        wntr.graphics.plot_network(self.wn, link_attribute=pipe_labels,
                                   node_size=2.5, link_width=1.3,
                                   add_colorbar=False)
        for r_name, r in self.wn.reservoirs():
            s3 = plt.scatter(self.wn.get_node(r_name).coordinates[0], self.wn.get_node(r_name).coordinates[1],
                             c='deepskyblue',
                             edgecolor='k', marker='s', s=20, zorder=4, linewidths=0.5)
        for t_name, t in self.wn.tanks():
            plt.scatter(self.wn.get_node(t_name).coordinates[0], self.wn.get_node(t_name).coordinates[1],
                        c='blue', marker='s', s=10, zorder=2)
        j = 0
        for mq in self.monitor_q:
            s1 = plt.scatter(1 / 2 * (self.wn.get_node(self.wn.get_link(mq).start_node_name).coordinates[0]
                                      + self.wn.get_node(self.wn.get_link(mq).end_node_name).coordinates[0]),
                             1 / 2 * (self.wn.get_node(self.wn.get_link(mq).start_node_name).coordinates[1]
                                      + self.wn.get_node(self.wn.get_link(mq).end_node_name).coordinates[1]),
                             c='w',
                             edgecolor='k',
                             marker='o', s=10, zorder=3, linewidth=0.8)
            j = j + 1
        plt.rcParams['font.size'] = 9
        i = 0
        for mH in self.monitor_H:
            s2 = plt.scatter(self.wn.get_node(mH).coordinates[0], self.wn.get_node(mH).coordinates[1],
                             c='w',
                             edgecolor='k',
                             marker='^', s=15, zorder=3, linewidth=0.8)
            i = i + 1
        plt.rcParams['font.size'] = 9
        # cb2 = plt.colorbar(fraction=0.023, pad=0.1, ticks=np.arange(0, 0.22, 0.04))
        # cb2.ax.tick_params(labelsize=9)
        plt.legend((s3, s1, s2), ('reservior', 'pipe flow rate meter', 'node pressure meter'), loc='lower left',
                   prop=font)
        plt.savefig(savepath + 'pipe_groupwithH.png', dpi=500, bbox_inches='tight')
        plt.close()

    def plotcalibresult_all(self, CRecorder, savepath=None):
        plt.figure()
        if savepath is None:
            savepath = self.savepath
        plt.plot(range(0, CRecorder.shape[1]), CRecorder.T)
        plt.legend(
            ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', \
             'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'],
            loc='upper right')
        x_major_locator = MultipleLocator(50)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim([0, CRecorder.shape[1]])
        ylim1 = np.min(CRecorder.T[-1, :]) * 0.6
        ylim2 = np.max(CRecorder.T[-1, :]) * 1.2
        plt.ylim([ylim1, ylim2])
        # plt.ylim([40, 160])
        plt.xlabel('Iteration')
        plt.ylabel('Cw')
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', direction='in')
        plt.savefig(savepath + 'c_result_all.png', dpi=800, bbox_inches='tight')
        plt.close()

    def plotC_err_distribution(self, dertaC, savepath=None):
        if savepath is None:
            savepath = self.savepath
        font = {'family': 'Arial', 'size': 9}
        plt.figure(figsize=(4, 3))
        plt.scatter(range(len(dertaC)), dertaC, c=abs(dertaC),
                    s=20, alpha=0.7, linewidth=0.7)
        plt.axhline(y=-10, color='r', linestyle='--')
        plt.axhline(y=10, color='r', linestyle='--')
        plt.axhline(y=-5, color='r', linestyle='--')
        plt.axhline(y=5, color='r', linestyle='--')
        plt.xlabel('pipe ID', font=font)
        plt.ylabel('CHW error', font=font)
        plt.ylim([-11, 11])
        yticks = list(range(-10,11,5))
        yticks_label = [str(each) for each in yticks]
        plt.yticks(yticks, yticks_label)
        xticks = list(range(0, 567, 100)) + [567]
        xticks_label = [str(each) for each in xticks]
        plt.xticks(xticks, xticks_label)
        ax = plt.gca()
        labelsy = ax.get_yticklabels()
        labelsx = ax.get_xticklabels()
        [label.set_fontname('Arial') for label in labelsy]
        [label.set_fontname('Arial') for label in labelsx]
        ax.tick_params(axis='both', which='both', direction='in', labelsize=9)
        plt.savefig(savepath + 'C_err.png', dpi=500,
                    bbox_inches='tight'
                    )
        plt.close()

    def plotpipedeltaC(self, index, labels, savepath=None):
        if savepath is None:
            savepath = self.savepath
        pipe_labels = pd.Series(data=labels, index=index.tolist())
        wntr.graphics.plot_network(self.wn, link_attribute=pipe_labels, node_size=1, link_width=1.5,
                                   title='pipe_deltaC', add_colorbar=False, link_range=[0, 2 * np.max(labels)])
        for r_name, r in self.wn.reservoirs():
            plt.scatter(self.wn.get_node(r_name).coordinates[0], self.wn.get_node(r_name).coordinates[1],
                        c='green', marker='^', s=60, zorder=2)
        for t_name, t in self.wn.tanks():
            plt.scatter(self.wn.get_node(t_name).coordinates[0], self.wn.get_node(t_name).coordinates[1],
                        c='blue', marker='^', s=60, zorder=2)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
        sm = plt.cm.ScalarMappable(norm=norm)
        sm.set_array([])
        plt.colorbar(sm, shrink=0.5, pad=0.05, ticks=np.linspace(0, 20, 8))
        plt.savefig(savepath + 'pipe_C_err_distribution.png', dpi=500, bbox_inches='tight')
        plt.close()