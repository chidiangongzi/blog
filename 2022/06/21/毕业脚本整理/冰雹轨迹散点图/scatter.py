#----------------------------------------------------------
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
import os
from mpl_toolkits.basemap import Basemap
matplotlib.rcParams['font.sans-serif'] = ['FangSong']
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
# define variable
#----------------------------------------------------------
fl = os.popen("wc -l <./Trajectory-0.5.txt")
fw = (os.popen("wc -w <./Trajectory-0.5.txt"))
nn = fl.read()
nn = int(nn)
print("nn=", nn)
nt = fw.read()
nt = int(nt)
nt = nt / (nn * 4)
nt = int(nt)
print("nt=", nt)
first_2000 = np.empty([nt, nn], dtype=float)
second_2000 = np.empty([nt, nn], dtype=float)
third_2000 = np.empty([nt, nn], dtype=float)
d = np.empty([nt, nn], dtype=float)
#----------------------------------------------------------
# load data from file
# you replace this using with open
#----------------------------------------------------------
data1 = np.loadtxt("./Trajectory-0.5.txt")
first_2000[:, 0:nn] = np.transpose(data1[0:nn, 0:nt] / 2)
second_2000[:, 0:nn] = np.transpose(data1[0:nn, nt:2 * nt] / 2)
third_2000[:, 0:nn] = np.transpose(data1[0:nn, 2 * nt:3 * nt] / 2)
d[:, 0:nn] = np.transpose(data1[0:nn, 3 * nt:4 * nt])
x_co = (107.0287 - 104.9827) / 180
y_co = (36.88075 - 35.22662) / 180
first_2000 = 104.9827 + first_2000 * x_co
second_2000 = 35.22662 + second_2000 * y_co
#----------------------------------------------------------
fig = plt.figure(figsize=(5, 6))
fig.subplots_adjust(left=0.22, bottom=0.18, right=0.95, top=0.73)
ax = fig.gca()
# 筛选出满足直径在0.5-1cm 且处于两个区域范围内的轨迹
#iw1 = np.where(((d[nt - 1, :] >= 0.5) & (d[nt - 1, :] < 1)) & ((
#    (first_2000[nt - 1, :] >= 104.98) & (first_2000[nt - 1, :] <= 105.5)
#    & (second_2000[nt - 1, :] >= 35.4) & (second_2000[nt - 1, :] <= 36.2))
#               | ((first_2000[nt - 1, :] >= 105.6)
#                  & (first_2000[nt - 1, :] <= 106.6)
#                  & (second_2000[nt - 1, :] >= 36.00)
#                  & (second_2000[nt - 1, :] <= 36.40))))
iw1 = np.where((d[nt - 1, :] >= 0.5) & (d[nt - 1, :] <= 1.0))
ax.scatter(np.transpose(first_2000[nt - 1, iw1]),
           np.transpose(second_2000[nt - 1, iw1]),
           alpha=0.8,
           edgecolors='k',
           linewidths=0.2,
           s=20,
           label='number of hail (0.5~1.0cm):   ' + str(len(iw1[0])),
           c='blue')
#iw2 = np.where(((d[nt - 1, :] >= 1.0) & (d[nt - 1, :] < 1.5)) & ((
#    (first_2000[nt - 1, :] >= 104.98) & (first_2000[nt - 1, :] <= 105.5)
#    & (second_2000[nt - 1, :] >= 35.4) & (second_2000[nt - 1, :] <= 36.2))
#               | ((first_2000[nt - 1, :] >= 105.6)
#                  & (first_2000[nt - 1, :] <= 106.6)
#                  & (second_2000[nt - 1, :] >= 36.00)
#                  & (second_2000[nt - 1, :] <= 36.40))))
iw2 = np.where((d[nt - 1, :] > 1.0) & (d[nt - 1, :] <= 1.5))
ax.scatter(np.transpose(first_2000[nt - 1, iw2]),
           np.transpose(second_2000[nt - 1, iw2]),
           alpha=0.8,
           edgecolors='k',
           linewidths=0.2,
           s=20,
           label='number of hail (1.0~1.5cm):   ' + str(len(iw2[0])),
           c='green')
#iw3 = np.where(((d[nt - 1, :] >= 1.5) & (d[nt - 1, :] < 2.0)) & ((
#    (first_2000[nt - 1, :] >= 104.98) & (first_2000[nt - 1, :] <= 105.5)
#    & (second_2000[nt - 1, :] >= 35.4) & (second_2000[nt - 1, :] <= 36.2))
#               | ((first_2000[nt - 1, :] >= 105.6)
#                  & (first_2000[nt - 1, :] <= 106.6)
#                  & (second_2000[nt - 1, :] >= 36.00)
#                  & (second_2000[nt - 1, :] <= 36.40))))
iw3 = np.where((d[nt - 1, :] > 1.5) & (d[nt - 1, :] <= 2.0))
ax.scatter(np.transpose(first_2000[nt - 1, iw3]),
           np.transpose(second_2000[nt - 1, iw3]),
           alpha=0.8,
           edgecolors='k',
           linewidths=0.2,
           s=20,
           label='number of hail (1.5~2.0cm):   ' + str(len(iw3[0])),
           c='yellow')

#iw4 = np.where((((d[nt - 1, :] >= 2.0)) & (d[nt - 1, :] < 2.5)) & ((
#    (first_2000[nt - 1, :] >= 104.98) & (first_2000[nt - 1, :] <= 105.5)
#    & (second_2000[nt - 1, :] >= 35.4) & (second_2000[nt - 1, :] <= 36.2))
#               | ((first_2000[nt - 1, :] >= 105.6)
#                  & (first_2000[nt - 1, :] <= 106.6)
#                  & (second_2000[nt - 1, :] >= 36.00)
#                  & (second_2000[nt - 1, :] <= 36.40))))
iw4 = np.where((d[nt - 1, :] > 2.0) & (d[nt - 1, :] <= 2.5))
ax.scatter(np.transpose(first_2000[nt - 1, iw4]),
           np.transpose(second_2000[nt - 1, iw4]),
           alpha=0.8,
           edgecolors='k',
           linewidths=0.2,
           s=20,
           label='number of hail (2.0cm~2.5cm):   ' + str(len(iw4[0])),
           c='red')

#iw5 = np.where(((d[nt - 1, :] >= 2.5)) & ((
#    (first_2000[nt - 1, :] >= 104.98) & (first_2000[nt - 1, :] <= 105.5)
#    & (second_2000[nt - 1, :] >= 35.4) & (second_2000[nt - 1, :] <= 36.2))
#               | ((first_2000[nt - 1, :] >= 105.6)
#                  & (first_2000[nt - 1, :] <= 106.6)
#                  & (second_2000[nt - 1, :] >= 36.00)
#                  & (second_2000[nt - 1, :] <= 36.40))))
iw5 = np.where(((d[nt - 1, :] > 2.5)))
ax.scatter(np.transpose(first_2000[nt - 1, iw5]),
           np.transpose(second_2000[nt - 1, iw5]),
           alpha=0.8,
           edgecolors='k',
           linewidths=0.2,
           s=20,
           label='number of hail (>2.5cm):   ' + str(len(iw5[0])),
           c='purple')

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
ax.legend(fontsize=12,
          loc='lower center',
          bbox_to_anchor=(0.29, 1.02, 0.5, 0.5),
          frameon=True,
          fancybox=True,
          framealpha=0.2,
          borderpad=0.3,
          ncol=1,
          markerfirst=True,
          markerscale=1,
          numpoints=1,
          handlelength=0.5)

#----------------------------------------------------------
x_major_locator = MultipleLocator(0.50)
ax.xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(0.50)
ax.yaxis.set_major_locator(y_major_locator)
#禁用标签偏移量http://www.voidcn.com/article/p-ehmibndo-bsw.html
ax.ticklabel_format(style='sci', useOffset=False)
plt.xlim(104.9827, 107.0287) #确定横轴坐标范围
plt.ylim(35.22662, 36.88075) #确定纵轴坐标范围
#-----------------------------------------------------------

#ax.text(105.2, 36.0, 'This is a txt')
plt.annotate("I", xy=(105.34, 36.55), xytext=(105.34, 36.55), color="green", fontsize=15)
points = [[105.23, 36.78], [105.23, 36.46], [105.59, 36.46], [105.59, 36.78]]
line = plt.Polygon(points,
                   closed=True,
                   fill=None,
                   edgecolor='green',
                   linestyle='--')
plt.gca().add_line(line)

plt.annotate("II",
             xy=(106.58, 35.30),
             xytext=(106.58, 35.30),
             color="orange",
             fontsize=15)
points = [[106.33, 35.38], [106.33, 35.23], [106.80, 35.23], [106.80, 35.38]]
line = plt.Polygon(points,
                   closed=True,
                   fill=None,
                   edgecolor='orange',
                   linestyle='--')
plt.gca().add_line(line)

plt.annotate("III", xy=(105.10, 35.55), xytext=(105.10, 35.55), color="red", fontsize=15)
points = [[105.00, 35.90], [105.00, 35.41], [105.39, 35.41], [105.39, 35.90]]
line = plt.Polygon(points,
                   closed=True,
                   fill=None,
                   edgecolor='red',
                   linestyle='--')
plt.gca().add_line(line)

plt.annotate("IV",
             xy=(106.10, 36.30),
             xytext=(106.10, 36.30),
             color="#ac4f06",
             fontsize=15)
points = [[105.68, 36.78], [105.68, 35.88], [106.75, 35.88], [106.75, 36.78]]
line = plt.Polygon(points,
                   closed=True,
                   fill=None,
                   edgecolor="#ac4f06",
                   linestyle='--')
plt.gca().add_line(line)

plt.annotate("V",
             xy=(105.05, 36.00),
             xytext=(105.05, 36.00),
             color="purple",
             fontsize=15)
points = [[105.00, 36.13], [105.00, 35.83], [105.21, 35.83], [105.21, 36.13]]
line = plt.Polygon(points,
                   closed=True,
                   fill=None,
                   edgecolor='purple',
                   linestyle='--')
plt.gca().add_line(line)

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

#plt.show()
plt.savefig('./scatter.png', dpi=500)
