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
order = np.loadtxt("./order.txt") #聚类结果排序
uvw = np.loadtxt("./data/UVW-0.5.txt")
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
uu = np.empty([nt, nn], dtype=float)
vv = np.empty([nt, nn], dtype=float)
ww = np.empty([nt, nn], dtype=float)
# load data from file
# you replace this using with open
#----------------------------------------------------------
first[:, 0:nn] = np.transpose(data1[0:nn, 0:nt] / 2)
second[:, 0:nn] = np.transpose(data1[0:nn, nt:2*nt] / 2)
third[:, 0:nn] = np.transpose(data1[0:nn, 2*nt:3*nt] / 2)
d[:, 0:nn] = np.transpose(data1[0:nn, 3*nt:4*nt])
uu[:, 0:nn] = np.transpose(uvw[0:nn, 0:nt] / 2)
vv[:, 0:nn] = np.transpose(uvw[0:nn, nt:2*nt] / 2)
ww[:, 0:nn] = np.transpose(uvw[0:nn, 2*nt:3*nt] / 2)
traj = np.zeros((nn, nt, 3))
traj[:, :, 0] = np.transpose(first)
traj[:, :, 1] = np.transpose(second)
traj[:, :, 2] = np.transpose(third)
X = traj[:, :, :]
XX = X.reshape((nn, nt * 3))
#print(X)
#print("间隔")
#print(XX)
#print("test")
#print(X[1, 0, 0])
#print(XX[nt, 0])
#print("test")

SSE = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(XX)
    SSE.append(km.inertia_)
print("SSE=", SSE)
print("SSE0=", SSE[0])
#SSB = map(int, SSE)
### 斜率阈值
SSB = SSE[:]
SSB[0] = 1
for k in range(1, 10):
    #    print(k)
    #    print(SSE[k - 1] / SSE[k])
    SSB[k] = (SSE[k - 1] * 1.0 - SSE[k] * 1.0) / SSE[0] * 1.0
print("SSB=", SSB)
xx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(xx, SSE)
#plt.savefig('./SSE.png', dpi=500)
#plt.show()
#plt.close()

clf = Kmeans(k=5)
y_pred = clf.predict(XX)
print("y_pred", y_pred)
print("y_pred.shape", y_pred.shape)
#np.savetxt('y_pred.txt', (y_pred), delimiter=',', fmt="%d")
y_pred = np.loadtxt('y_pred.txt', delimiter=',')
#----------------------------------------------------------
# print hail's diameter
print(len(d))
#----------------------------------------------------------
#ax = fig.gca()
# draw the figure, the color is r = read
dmax = np.array(d).max()
dmin = np.array(d).min()
print(dmin, dmax)
#----------------------------------------------------------
# delete some aberrant point
i = 0
j = 0
for i in range(nn):
    for j in range(nt):
        if np.any(first[j, i] >= 999):
            #            print(first[j, i])
            first[j, i] = 0
#----------------------------------------------------------
# xy坐标max，mi'
xmax = np.array(first).max()
xmin = np.array(first).min()
print('xmin=', xmin, '  ', 'xmax=', xmax)
ymax = np.array(second).max()
ymin = np.array(second).min()
print('ymin=', ymin, '  ', 'ymax=', ymax)
x_co = (107.0287 - 104.9827) / 180
y_co = (36.88075 - 35.22662) / 180
first = 104.9827 + first * x_co
second = 35.22662 + second * y_co
#print(first)
#print(second)

#----------------------------------------------------------
# choose color for hail's diameter
i = 0
j = 0
tt = 0
flag = 0

## add
#iw = np.where((second[nt - 1, :] >= 36.0000) & (second[nt - 1, :] < 36.5000))
####iw = np.where(y_pred[:] == 1)
####print('iw=', iw[0])
####print('iw.shape=', iw[0].shape)
####print('y_pred.shape=', y_pred.shape)
####print('len=', len(iw[0]))
#print('iw[0]=', iw[0][0])
#print('iw[1]=', iw[0][1])
####maxd = max(max(d[nt - 1, iw]))
####iwd = np.where(d[nt - 1, :] == maxd)
####print("iwd=", iwd)
####print("iwd=", iwd[0])
#print(d[:, iw][:, 0, 0])
#print(d[:, iw][:, 0, 1])

####it = np.empty([len(iw[0])], dtype=int)
####for i in range(len(iw[0])):
####    #    print(i)
####    #    it[i] = min(min(np.where(d[:, iw][:, 0, iw[0][i]] >= 0.3)))
####    it[i] = min(min(np.where(d[:, iw][:, 0, i] >= 0.3)))
####    if (it[i] <= 10):
####        it[i] = 10
####print('it=', it)
## add
mm = -1
#for i in range(nn):
for k in range(5):
    # new a figure and set it into 3d
    fig = plt.figure()
    ax = fig.gca()
    print("K=", k)
    iw = np.where(y_pred[:] == order[k])

    ### 计算大于0.1cm的时刻
    it = np.empty([len(iw[0])], dtype=int)
    for i in range(len(iw[0])):
        #        print("d[:,iw].shape=", d[:, iw].shape)
        #        print(np.where(d[:, iw][:, 0, i] >= 0.1))
        it[i] = min(min(np.where(d[:, iw][:, 0, i] >= 0.0)))
#        if (it[i] <= 10):
#            it[i] = 10
    print('it.shape=', it.shape)
    print('iw.shape=', iw[0].shape)
    #    print('it=', it)

    # 被选中聚类轨迹的最大值
    diwmax = np.amax(d[nt-1, iw], axis=0)
    print(diwmax)
    print("diwmax.shape", diwmax.shape)
    iwiw11 = 999999
    iwiw22 = 999999
    iwiw33 = 999999
    iwiw44 = 999999
    iwiw55 = 999999
    i11 = 999999
    i22 = 999999
    i33 = 999999
    i44 = 999999
    i55 = 999999
    if (np.array(diwmax).max() > 0.5):
        iwiw1 = np.where((diwmax[:] > 0.5) & (diwmax[:] <= 1.0))
        iwiw11 = np.where(diwmax[:] == np.array(diwmax[iwiw1]).max())
    if (np.array(diwmax).max() > 1.0):
        iwiw2 = np.where((diwmax[:] > 1.0) & (diwmax[:] <= 1.5))
        iwiw22 = np.where(diwmax[:] == np.array(diwmax[iwiw2]).max())
    if (np.array(diwmax).max() > 1.5):
        iwiw3 = np.where((diwmax[:] > 1.5) & (diwmax[:] <= 2.0))
        iwiw33 = np.where(diwmax[:] == np.array(diwmax[iwiw3]).max())
    if (np.array(diwmax).max() > 2.0):
        iwiw4 = np.where((diwmax[:] > 2.0) & (diwmax[:] <= 2.5))
        iwiw44 = np.where(diwmax[:] == np.array(diwmax[iwiw4]).max())
    if (np.array(diwmax).max() > 2.5):
        iwiw5 = np.where((diwmax[:] > 2.5))
        iwiw55 = np.where(diwmax[:] == np.array(diwmax[iwiw5]).max())
    if (iwiw11 != 999999):
        print("iwiw11", iwiw11[0][0])
        print("0.5-1.0cm最大直径", diwmax[iwiw11])
        print("0.5-1.0cm轨迹数", iwiw1[0].shape)
        #print("iw", iw)
        #print("iw.shape", iw[0].shape)
        i11 = iw[0][iwiw11[0][0]]
        print("i11", i11)
    if (iwiw22 != 999999):
        print("iwiw22", iwiw22[0][0])
        print("1.0-1.5cm最大直径", diwmax[iwiw22])
        print("1.0-1.5cm轨迹数", iwiw2[0].shape)
        i22 = iw[0][iwiw22[0][0]]
        print("i22", i22)
    if (iwiw33 != 999999):
        print("iwiw33", iwiw33[0])
        print("1.5-2.0cm最大直径", diwmax[iwiw33])
        print("1.5-2.0cm轨迹数", iwiw3[0].shape)
        i33 = iw[0][iwiw33[0][0]]
        print("i33", i33)
    if (iwiw44 != 999999):
        print("iwiw44", iwiw44[0][0])
        print("2.0-2.5cm最大直径", diwmax[iwiw44])
        print("2.0-2.5cm轨迹数", iwiw4[0].shape)
        i44 = iw[0][iwiw44[0][0]]
        print("i44", i44)
    if (iwiw55 != 999999):
        print("iwiw55", iwiw55[0][0])
        print(">2.5cm最大直径", diwmax[iwiw55])
        print(">2.5cm-轨迹数", iwiw5[0].shape)
        i55 = iw[0][iwiw55[0][0]]
        print("i55", i55)
    print('iw=', iw[0])
    print('iw.shape=', iw[0].shape)
    dmaxiw5 = [iwiw11, iwiw22, iwiw33, iwiw44, iwiw55]
    #    print('y_pred.shape=', y_pred.shape)
    #    print('len=', len(iw[0]))
    minlat = 999
    maxlat = 0
    minlon = 999
    maxlon = 0
    maxhei = 0
    diw = [i11, i22, i33, i44, i55]
    kdiw = 0
#    for i in range(nn):
#        if ((i == i11) | (i == i22) | (i == i33) | (i == i44) | (i == i55)):
    for m in range(5):
            i =diw[m]
            if np.array(i==999999):
                continue
            if (maxlon < np.array(
                    first[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())):
                maxlon = np.array(
                    first[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())
            if (minlon > np.array(
                    first[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].min())):
                minlon = np.array(
                    first[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].min())
            if (maxlat < np.array(
                    second[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())):
                maxlat = np.array(
                    second[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())
            if (minlat > np.array(
                    second[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].min())):
                minlat = np.array(
                    second[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].min())
            if (maxhei < np.array(
                    third[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())):
                maxhei = np.array(
                    third[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())
            print("maxlon", maxlon)
            print("minlon", minlon)
            print("maxlat", maxlat)
            print("minlat", minlat)
        
            gyglon = 0
            if np.any((maxlon-minlon)<=0.051):
                gyglon = 1
            if np.any((maxlon-minlon)>0.051):
                gyglat = 0
                maxlon = float(int(math.ceil(maxlon * 100 / 5) *5)/100)
                minlon = float(int(math.floor(minlon * 100 / 5) *5)/100)
            maxlat = float(int(math.ceil(maxlat * 100 / 5) *5)/100)
            minlat = float(int(math.floor(minlat * 100 / 5) *5)/100)
            maxhei = int(math.ceil(maxhei))
            print("maxlon", float(int(math.ceil(maxlon * 100 / 5) *5)/100))
            print("minlon", float(int(math.floor(minlon * 100 / 5) *5)/100))
            print("maxlat", float(math.ceil(maxlat * 100 / 5) *5)/100)
            print("minlat", float(int(math.floor(minlat * 100 / 5) *5)/100))

    for m in range(5):
            i =diw[m]
            if np.array(i==999999):
                break
            fig = plt.figure()
            ax = fig.gca()
#            minlat = 999
#            maxlat = 0
#            minlon = 999
#            maxlon = 0
#            maxhei = 0
#            if (maxlon < np.array(
#                    first[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())):
#                maxlon = np.array(
#                    first[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())
#            if (minlon > np.array(
#                    first[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].min())):
#                minlon = np.array(
#                    first[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].min())
#            if (maxlat < np.array(
#                    second[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())):
#                maxlat = np.array(
#                    second[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())
#            if (minlat > np.array(
#                    second[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].min())):
#                minlat = np.array(
#                    second[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].min())
#            if (maxhei < np.array(
#                    third[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())):
#                maxhei = np.array(
#                    third[it[(np.where(iw[0][:] == i)[0][0])]:nt+1, i].max())
            print(np.array(second[:, i].min()))
            print("Satisfyied I=", i)
            print("Satisfyide J=", it[np.where(iw[0][:] == i)[0][0]])
            for j in range(it[(np.where(iw[0][:] == i)[0][0])], nt):
                if np.any(d[j, i] >= 0) and np.any(d[j, i] <= 0.3):
                    cc = 'lightskyblue'
                if np.any(d[j, i] >= 0.3) and np.any(d[j, i] <= 0.6):
                    cc = 'blue'
                if np.any(d[j, i] > 0.6) and np.any(d[j, i] <= 0.9):
                    cc = 'darkblue'
                if np.any(d[j, i] > 0.9) and np.any(d[j, i] <= 1.2):
                    cc = 'lime'
                if np.any(d[j, i] > 1.2) and np.any(d[j, i] <= 1.5):
                    cc = '#5cac2d' # grass
                if np.any(d[j, i] > 1.5) and np.any(d[j, i] <= 1.8):
                    cc = '#1f6357' # dark green blue
                if np.any(d[j, i] > 1.8) and np.any(d[j, i] <= 2.1):
                    cc = '#fedf08' #dandelion
                if np.any(d[j, i] > 2.1) and np.any(d[j, i] <= 2.4):
                    cc = '#ac7e04' #lemon yellow
                if np.any(d[j, i] > 2.4) and np.any(d[j, i] <= 2.7):
                    cc = '#EE7600' #mustard brown
                if np.any(d[j, i] > 2.7) and np.any(d[j, i] <= 3.0):
                    cc = '#ffb19a' #pale salmon
                if np.any(d[j, i] > 3.0) and np.any(d[j, i] <= 3.3):
                    cc = 'red'
                if np.any(d[j, i] > 3.3) and np.any(d[j, i] <= 3.6):
                    cc = '#CD1076' #deep red
                if (j%5==0 and j<=nt-3):
                    cwind = ax.quiver(first[j+1, i],third[j+1, i],uu[j+1, i],ww[j+1, i],color='#9900ff',width=0.0060,scale=60,alpha=1)
                figure1 = ax.plot(first[j:j + 2, i],
                                  third[j:j + 2, i],
                                  color=cc,
                                  lw=2.0)
                j += 1
            kdiw += 1
            ax.quiverkey(cwind,0.81,0.91,5,"5m·$\mathregular{{s^{-1}}}$",labelpos='E',coordinates='axes',fontproperties={'size':15,'family':'Times New Roman'})
            x_major_locator = MultipleLocator(0.05)
            ax.xaxis.set_major_locator(x_major_locator)
            y_major_locator = MultipleLocator(1)
            ax.yaxis.set_major_locator(y_major_locator)
            #禁用标签偏移量http://www.voidcn.com/article/p-ehmibndo-bsw.html
            ax.ticklabel_format(style='sci', useOffset=False)
        #    plt.xlim(minlon, maxlon+0.002,0.05)
            plt.xlim(minlon, maxlon+0.002,0.05)
            plt.ylim(0, maxhei+1,1)
        #    plt.xticks(rotation=-15+20)
        #    plt.yticks(rotation=40+20)
            ax.set_xlabel("Longtitude", fontsize=18,labelpad =10)
            ax.set_xlim(minlon, maxlon+0.002)
        #    ax.set_xlabel("Latitude", fontsize=18,labelpad =10)
        #    ax.set_xlim(minlat, maxlat+0.002)
            ax.set_ylabel("Height (km)", fontsize=18)
            ax.set_ylim(0, maxhei+1)
        
            if np.any((gyglon)==1):
                maxlon = float(int(math.ceil((maxlon-0.0002) * 100 / 2) *2)/100)
                minlon = float(int(math.floor((minlon+0.0002) * 100 / 2) *2)/100)
                x_major_locator = MultipleLocator(0.02)
                ax.xaxis.set_major_locator(x_major_locator)
                plt.xlim(minlon, maxlon+0.002,0.02)
                ax.set_xlim(minlon, maxlon+0.002)
                print("cnm,要改")
        
            plt.tight_layout() # xlabel 坐标轴显示不全
            #----------------------------------------------------------
            # set figure information
            #ax.set_title("冰雹增长轨迹（水平投影）")
            #ax.set_zlabel("Height", fontsize=15)
            #plt.xlabel('x1', fontsize=7) #xlab
            #plt.ylabel('y1', fontsize=7) #ylabel
            #plt.zlabel('h1', fontsize=7) #ylabel
            #z_major_locator = MultipleLocator(1) #y interval
            ax = plt.gca()
            #ax.yaxis.set_major_locator(z_major_locator)
            #plt.xlim(-3, 40) #xmax=11,xmin=-0.5
            #plt.ylim(25, 60) #xin=-5,xmax=110
            #----------------------------------------------------------
            #    x_major_locator = MultipleLocator(0.5)
            #    y_major_locator = MultipleLocator(0.5)
            #禁用标签偏移量http://www.voidcn.com/article/p-ehmibndo-bsw.html
            ax.ticklabel_format(style='sci', useOffset=False)
            #-----------------------------------------------------------
            plt.tick_params(labelsize=18)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            plt.subplots_adjust(left=0.21, bottom=0.185)
        #    plt.subplots_adjust(left=0.05, bottom=0.1)
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
        
            plt.savefig("./kmeans-lon-K=" + str(k+1)+ "-"+ str(kdiw)+".png", dpi=500)
            plt.close()
