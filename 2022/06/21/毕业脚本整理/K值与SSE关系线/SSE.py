#----------------------------------------------------------
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.patches as mpatches
import matplotlib
from scipy import interpolate
import pylab as pl
import matplotlib as mpl
import random
import os
import math
from sklearn import datasets
matplotlib.rcParams['font.sans-serif'] = ['FangSong']
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']


# 正规化数据集 X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X,
                         2).sum(axis=1)
    return distances


class Kmeans():
    """Kmeans聚类算法.

	Parameters:
	-----------
	k: int
		聚类的数目.
	max_iterations: int
		最大迭代次数. 
	varepsilon: float
		判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon, 
		则说明算法已经收敛
	"""
    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)

        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids

            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break

        return self.get_cluster_labels(clusters, X)


# define variable
#----------------------------------------------------------
fl = os.popen("wc -l <./data/Trajectory-0.5.txt")
fw = (os.popen("wc -w <./data/Trajectory-0.5.txt"))
data1 = np.loadtxt("./data/Trajectory-0.5.txt")
nn = fl.read()
nn = int(nn)
print("nn=", nn)
nt = fw.read()
nt = int(nt)
nt = nt / (nn * 4)
nt = int(nt)
print("nt=", nt)
first = np.empty([nt, nn], dtype=float)
second = np.empty([nt, nn], dtype=float)
third = np.empty([nt, nn], dtype=float)
d = np.empty([nt, nn], dtype=float)
# load data from file
# you replace this using with open
#----------------------------------------------------------
ntt = 92
first[:, 0:nn] = np.transpose(data1[0:nn, 0:ntt] / 2)
second[:, 0:nn] = np.transpose(data1[0:nn, ntt:2 * ntt] / 2)
third[:, 0:nn] = np.transpose(data1[0:nn, 2 * ntt:3 * ntt] / 2)
d[:, 0:nn] = np.transpose(data1[0:nn, 3 * ntt:4 * ntt])
traj = np.zeros((nn, nt, 3))
traj[:, :, 0] = np.transpose(first)
traj[:, :, 1] = np.transpose(second)
traj[:, :, 2] = np.transpose(third)
X = traj[:, :, :]
XX = X.reshape((nn, nt * 3))

SSE = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(XX)
    SSE.append(km.inertia_)
print("SSE=", SSE)
print("SSE0=", SSE[0])
xx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#plt.plot(xx, SSE)
fig = plt.figure()
ax = fig.gca()
figure0 = ax.plot(xx, np.array(SSE) / 1e9)

ax = plt.gca()
x_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)
#禁用标签偏移量http://www.voidcn.com/article/p-ehmibndo-bsw.html
ax.ticklabel_format(style='sci', useOffset=False)
maxSSE = math.ceil(max(SSE) / 1e9)
print("maxSSE=", math.ceil(max(SSE) / 1e9))
plt.xlim(1, 10, 1)
plt.ylim(0, maxSSE, 1)
ax.set_xlabel("k", fontsize=18, labelpad=10)
ax.set_xlim(1, 10)
ax.set_ylabel("SSE (e9)", fontsize=18, labelpad=10)
ax.set_ylim(0, maxSSE)
plt.tight_layout() # xlabel 坐标轴显示不全

#----------------------------------------------------------
# print hail's diameter
print(len(d))
#----------------------------------------------------------
#-----------------------------------------------------------
plt.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.subplots_adjust(left=0.21, bottom=0.185)
#plt.subplots_adjust(left=0.05, bottom=0.1)
for tick in ax.yaxis.get_major_ticks()[:]:
    tick.set_pad(12)
for line in ax.yaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('black')
    line.set_markersize(7)
    line.set_markeredgewidth(2)
for tick in ax.xaxis.get_major_ticks()[:]:
    tick.set_pad(12)
for line in ax.xaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('black')
    line.set_markersize(7)
    line.set_markeredgewidth(2)
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)

#y_ticks = ax.set_yticks([36])
#y_labels = ax.set_yticklabels(["one"])

plt.savefig('./SSE.png', dpi=500)
#plt.show()
plt.close()
