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
from mpl_toolkits.basemap import Basemap
import math
from sklearn import datasets
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.ndimage.filters import gaussian_filter
matplotlib.rcParams['font.sans-serif'] = ['FangSong']
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

#----------------------------------------------------------
fig = plt.figure(figsize=(5, 6))
fig.subplots_adjust(left=0.22, bottom=0.18, right=0.95, top=0.73)
ax = fig.gca()

fl = os.popen("wc -l <./dzx.txt")
fw = os.popen("wc -w <./dzx.txt")
data = np.loadtxt("./dzx.txt")
SC = np.loadtxt("./SC-0.5.txt")
nn = fl.read()
nn = int(nn)
#print("nn=", nn)
fsc = os.popen("wc -w <./SC-0.5.txt")
nsc = fsc.read()
nsc = int(nsc)
#print("nsc=", nsc)
nt = fw.read()
nt = int(nt)
nt = nt / (nn * 1)
nt = int(nt)
#print("nt=", nt)

# get maxium and minium of wind w
# ------------------------------------------------------
WM = np.loadtxt("./W-MIN-MAX-06:00.txt")

w2dmax = np.empty([360, 360], dtype=float)
w2dmin = np.empty([360, 360], dtype=float)

ss = 0
for j in range(360):
    for i in range(360):
        w2dmin[j, i] = WM[ss][2]
        w2dmax[j, i] = WM[ss][3]
        ss += 1
print(w2dmax)
# ------------------------------------------------------

first = np.empty([nn], dtype=float)
second = np.empty([nn], dtype=float)
third = np.empty([nn], dtype=float)
d = np.empty([nn], dtype=float)
qg = np.empty([nn], dtype=float)
qng = np.empty([nn], dtype=float)
w = np.empty([nn], dtype=float)
D = np.empty([360, 360, 35], dtype=float)
WW = np.empty([360, 360, 35], dtype=float)

first = data[0:nn, 1]
second = data[0:nn, 2]
third = data[0:nn, 3]
d = data[0:nn, 4]
qg = data[0:nn, 5]
qng = data[0:nn, 6]
w = data[0:nn, 7]
#----------------------------------------------------------
#print(third.max())
#print(d.max())
#print(qg.max())
#print(qng.max())
#print(w.max())

D[:, :, :] = 0
for i in range(nsc):
    D[int(second[int(SC[i])]),
      int(first[int(SC[i])]),
      int(third[int(SC[i])])] = d[int(SC[i])]

WW[:, :, :] = 0
for i in range(nn):
    WW[int(second[i]), int(first[i]), int(third[i])] = w[i]

q2d = np.amax(D, axis=2)
w2 = np.amax(WW, axis=2)
x = np.arange(0, 360, 1)
y = np.arange(0, 360, 1)
x_co = (107.0287 - 104.9827) / 360
y_co = (36.88075 - 35.22662) / 360
x = 104.9827 + x * x_co
y = 35.22662 + y * y_co
X, Y = np.meshgrid(x, y)
#print("q2dmin=", q2d.min())
#print("q2dmax=", q2d.max())
#sigma = 0.7
#q2d = gaussian_filter(q2d, sigma)

sigma = 0.9
w2 = gaussian_filter(w2, sigma)
w2dmin = gaussian_filter(w2dmin, sigma)
w2dmax = gaussian_filter(w2dmax, sigma)
print(len(w2[0]))
print(len(w2dmax[0]))
print(np.array(w2).max())
print(np.array(w2dmax).max())
print(np.array(w2dmin).min())

#levels_l = [0.1, 0.2, 0.3, 0.4, 0.5]
#levels_l = [1, 5, 10, 15, 20]
levels_m = [-10, -5]
levels_l = [5, 10, 15, 20]

C2 = plt.contour(
    X,
    Y,
    w2dmin,
    4,
    #    linestyles='-',
    #    colors=["#00ff33", "#ffff00", "#ff9933", "red", "#cc00ff"],
    colors=["black", "gray"],
    #                linewidths=np.arange(0.4, 2.4, 0.4))
    linestyles=':',
    linewidths=0.7,
    levels=levels_m)

C = plt.contour(
    X,
    Y,
    w2dmax,
    4,
    #    colors=["#FFC0CB", "#B03060", "red", "#A020F0", "#8A2BE2"],
    colors=["gray", "black", "lightblue", "blue"],
    #                linewidths=np.arange(0.4, 2.4, 0.4))
    linewidths=0.7,
    levels=levels_l)

#labels = [
#    '1m.$\mathregular{{s^{-1}}}$', '5m.$\mathregular{{s^{-1}}}$',
#    '10m.$\mathregular{{s^{-1}}}$', '15m.$\mathregular{{s^{-1}}}$',
#    '20m.$\mathregular{{s^{-1}}}$'
#]
labels = [
    '5m.$\mathregular{{s^{-1}}}$', '10m.$\mathregular{{s^{-1}}}$',
    '15m.$\mathregular{{s^{-1}}}$', '20m.$\mathregular{{s^{-1}}}$'
]

labels2 = ['-10m.$\mathregular{{s^{-1}}}$', '-5m.$\mathregular{{s^{-1}}}$']

for i in range(0, len(labels2)):
    C2.collections[i].set_label(labels2[i])

for i in range(0, len(labels)):
    C.collections[i].set_label(labels[i])

#plt.clabel(C, inline=1, fontsize=2, fmt='%1.0f')
cmap = ListedColormap(
    ["white", "#1aad19", "#99ff33", "#ffff00", "#ff9933", "red"])
#plot_examples([cmap])
#levels = [0, 4, 8, 12, 16, 20, 24]
#levels = [0, 1, 4, 8, 12, 16, 20, 24]
levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
#levels = np.array(levels)
#im = plt.contourf(X, Y, w2, 8, cmap=cmap, levels=levels[:], extend='both')
q2d = np.where(q2d, q2d, np.nan)
im = plt.contourf(
    X,
    Y,
    q2d,
    8,
    colors=["#1aad19", "#00ff33", "#99ff33", "#ffff00", "#ff9933"],
    #    cmap='RdPu',
    levels=levels[:])
#    extend='both')

iw = np.where((q2d[:, :] >= 0.1) & (q2d[:, :] <= 0.2))
iw0 = np.array(iw[0])
iw1 = np.array(iw[1])
#print(iw0)
#print(iw1)

# 坐标边框设置
ax.set_xlabel("Longtitude", fontsize=25)
ax.set_ylabel("Latitude", fontsize=25)
x_major_locator = MultipleLocator(1) #x interval
y_major_locator = MultipleLocator(1) #y interval
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

x_major_locator = MultipleLocator(0.50)
ax.xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(0.50)
ax.yaxis.set_major_locator(y_major_locator)
#禁用标签偏移量http://www.voidcn.com/article/p-ehmibndo-bsw.html
ax.ticklabel_format(style='sci', useOffset=False)

plt.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
#plt.subplots_adjust(left=0.21, bottom=0.185)
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

# 设置图片标签
ax.legend(fontsize=11,
          loc='lower center',
          bbox_to_anchor=(0.1, 1.02, 0.5, 0.5),
          frameon=True,
          fancybox=True,
          framealpha=0.2,
          borderpad=0.3,
          ncol=3,
          markerfirst=True,
          markerscale=1,
          numpoints=1,
          handlelength=0.5)

position1 = fig.add_axes([0.15, 0.95, 0.7, 0.03])
cb = plt.colorbar(im, cax=position1, orientation='horizontal') #方向

# 添加地图
#fig = plt.figure()
ax = fig.add_axes([0.22, 0.179, 0.73, 0.55], frameon=False)
ax.set_alpha(0)
m = Basemap(projection='mill',
            llcrnrlat=35.227,
            llcrnrlon=104.98,
            urcrnrlon=107.03,
            urcrnrlat=36.88)
m.readshapefile('help/gadm36_CHN_shp/gadm36_CHN_1', 'china',
                linewidth=2) #读取省份数据
m.readshapefile('help/gadm36_TWN_shp/gadm36_TWN_1', 'tw', linewidth=2) #读取台湾的数据
m.ax = ax

plt.savefig("./dzx.png", dpi=500)
#plt.show()
