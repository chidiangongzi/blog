import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.06,
                 1.03 * height,
                 '%s' % round(float(height), 1),
                 fontsize=15)


def autolabel2(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.06,
                 1.01 * height,
                 '%s' % round(float(2 * height), 1),
                 fontsize=15)


plt.figure(figsize=(13, 4))

#解决中文和负号
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['font.serif'] = ['SimHei']
##plt.rcParams['font.weight'] = "bold"
#plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.serif'] = ['Times New Roman']
#plt.rcParams['font.weight'] = "bold"
plt.rcParams['axes.unicode_minus'] = False

# xlabel
labels = ['CCN=50', 'CCN=200', 'CCN=2000', 'CCN=10000']
#d_0_5_1 = np.log([5289, 4283, 4740, 2972])
#d_1_1_5 = np.log([793, 779, 1309, 620])
#d_1_5_2 = np.log([177, 153, 839, 71])
#d_2_2_5 = np.log([38, 34, 429, 8])
#d_2_5 = np.log([9, 13, 94, 2])

d_0_5_1 = np.array([7004 / 8550, 5621 / 7019, 6078 / 8601, 5546 / 6534])
d_1_1_5 = np.array([1147 / 8550, 1129 / 7019, 1404 / 8601, 882 / 6534])
d_1_5_2 = np.array([296 / 8550, 226 / 7019, 766 / 8601, 90 / 6534])
d_2_2_5 = np.array([82 / 8550, 31 / 7019, 322 / 8601, 11 / 6534])
d_2_5 = np.array([21 / 8550, 12 / 7019, 31 / 8601, 5 / 6534])

#d_0_5_1 = d_0_5_1 / 2

ax = plt.gca()
x = np.arange(len(labels))
print(x)
width = 0.2
a = plt.bar(x - 1.5 * width,
            d_0_5_1 * 100,
            width,
            label='0.5~1cm',
            color='blue')
b = plt.bar(x - 0.75 * width,
            d_1_1_5 * 100,
            width,
            label='1~1.5cm',
            color='orange')
c = plt.bar(x, d_1_5_2 * 100, width, label='1.5-2cm', color='green')
d = plt.bar(x + 0.75 * width,
            d_2_2_5 * 100,
            width,
            label='2~2.5cm',
            color='red')
e = plt.bar(x + 1.5 * width,
            d_2_5 * 100,
            width,
            label='>2.5cm',
            color='purple')

####autolabel2(a)
####autolabel(b)
####autolabel(c)
####autolabel(d)
####autolabel(e)

# 调节label大小
tkw = dict(size=5, width=1.5, direction='in', labelsize=20)
#plt.tick_params(axis='both', labelsize=15)
plt.tick_params(axis='both', **tkw)

#plt.yscale("symlog")
plt.xticks(x, labels=labels)

#ax.set_ylabel("percentage (%)", fontsize=25)
#y = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#t = [1, 5, 10, 15, 20, 40, 60, 70, 80, 90, 100]
y = [0, 20, 40, 60, 80, 100]
t = ["0", "20%", "40%", "60%", "80%", "100%"]
plt.yticks(y, t)

#ax.set_ylabel('个数', fontsize=18)
#ax.set_xlabel('统计图', fontsize=18)

ax.legend(loc='upper right',
          fontsize=12,
          frameon=True,
          fancybox=True,
          framealpha=0.8,
          borderpad=0.3,
          labelspacing=1.0,
          ncol=1,
          markerfirst=False,
          markerscale=1,
          numpoints=1,
          bbox_to_anchor=(1.133, 0.80),
          handlelength=2.3)

ax.set_ylabel("Percentage", fontsize=30)
#labels = ax.get_xticklabels() + ax.get_yticklabels()
#[label.set_fontname('Times New Roman') for label in labels]
plt.title("Number of Hailstone", fontsize=30, x=0.5, y=1.03)
plt.tick_params(labelsize=25)
#labels = ax.get_xticklabels() + ax.get_yticklabels()
labelx = ax.get_xticklabels()
labely = ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labelx]
[label.set_fontname('Times New Roman') for label in labely]
# 横坐标下移
for tick in ax.xaxis.get_major_ticks()[:]:
    tick.set_pad(12)
for line in ax.xaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('black')
    line.set_markersize(7)
    line.set_markeredgewidth(2)
# 边框加粗
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
#-----------------------------------------------------
ax1 = ax.inset_axes([0.13, 0.5, 0.145, 0.4])
ax1.set_xlim(0 - 0.25, 0.5)
ax1.set_ylim(0, 4)
#a = ax1.bar(x - 1.5 * width,
#            d_0_5_1 * 100,
#            width,
#            label='0.5-1cm',
#            color='blue')
#b = ax1.bar(x - 0.75 * width,
#            d_1_1_5 * 100,
#            width,
#            label='1-1.5cm',
#            color='orange')
c = ax1.bar(x, d_1_5_2 * 100, width, label='1.5~2cm', color='green')
d = ax1.bar(x + 0.75 * width,
            d_2_2_5 * 100,
            width,
            label='2~2.5cm',
            color='red')
e = ax1.bar(x + 1.5 * width,
            d_2_5 * 100,
            width,
            label='>2.5cm',
            color='purple')
tkw = dict(size=5, width=1.5, direction='in', labelsize=16)
#plt.tick_params(axis='both', labelsize=15)
ax1.tick_params(axis='both', **tkw)
#plt.yscale("symlog")
ax1.set_xticks([0.])
ax1.set_xticklabels(["CCN=50"])
y = [0, 1, 2, 3, 4]
t = ["0", "1%", "2%", "3%", "4%"]
ax1.set_yticks(y)
ax1.set_yticklabels(t)
# 坐标字体
#labels = ax1.get_xticklabels() + ax1.get_yticklabels()
labelx = ax1.get_xticklabels()
labely = ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labelx]
[label.set_fontname('Times New Roman') for label in labely]
# 横坐标下移
for tick in ax1.xaxis.get_major_ticks()[:]:
    tick.set_pad(12)
for line in ax1.xaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('black')
    line.set_markersize(7)
    line.set_markeredgewidth(2)
# 边框加粗
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax1.spines['left'].set_linewidth(bwith)
ax1.spines['top'].set_linewidth(bwith)
ax1.spines['right'].set_linewidth(bwith)
ax1.spines['bottom'].set_linewidth(bwith)
#-----------------------------------------------------
ax2 = ax.inset_axes([0.375, 0.5, 0.1432979, 0.4])
ax2.set_xlim(1 - 0.25, 1.5)
ax2.set_ylim(0, 4)
#a = ax2.bar(x - 1.5 * width,
#            d_0_5_1 * 100,
#            width,
#            label='0.5-1cm',
#            color='blue')
#b = ax2.bar(x - 0.75 * width,
#            d_1_1_5 * 100,
#            width,
#            label='1-1.5cm',
#            color='orange')
c = ax2.bar(x, d_1_5_2 * 100, width, label='1.5-2cm', color='green')
d = ax2.bar(x + 0.75 * width,
            d_2_2_5 * 100,
            width,
            label='2~2.5cm',
            color='red')
e = ax2.bar(x + 1.5 * width,
            d_2_5 * 100,
            width,
            label='>2.5cm',
            color='purple')
tkw = dict(size=5, width=1.5, direction='in', labelsize=16)
#plt.tick_params(axis='both', labelsize=15)
ax2.tick_params(axis='both', **tkw)
#plt.yscale("symlog")
ax2.set_xticks([1.])
ax2.set_xticklabels(["CCN=200"])
y = [0, 1, 2, 3, 4]
t = ["0", "1%", "2%", "3%", "4%"]
ax2.set_yticks(y)
ax2.set_yticklabels(t)
# 坐标字体
#labels = ax1.get_xticklabels() + ax1.get_yticklabels()
labelx = ax2.get_xticklabels()
labely = ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labelx]
[label.set_fontname('Times New Roman') for label in labely]
# 横坐标下移
for tick in ax2.xaxis.get_major_ticks()[:]:
    tick.set_pad(12)
for line in ax2.xaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('black')
    line.set_markersize(7)
    line.set_markeredgewidth(2)
# 边框加粗
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax2.spines['left'].set_linewidth(bwith)
ax2.spines['top'].set_linewidth(bwith)
ax2.spines['right'].set_linewidth(bwith)
ax2.spines['bottom'].set_linewidth(bwith)
#-----------------------------------------------------
ax3 = ax.inset_axes([0.615, 0.5, 0.1432979, 0.4])
ax3.set_xlim(2 - 0.25, 2.5)
ax3.set_ylim(0, 15)
#a = ax3.bar(x - 1.5 * width,
#            d_0_5_1 * 100,
#            width,
#            label='0.5-1cm',
#            color='blue')
#b = ax3.bar(x - 0.75 * width,
#            d_1_1_5 * 100,
#            width,
#            label='1-1.5cm',

c = ax3.bar(x, d_1_5_2 * 100, width, label='1.5-2cm', color='green')
d = ax3.bar(x + 0.75 * width,
            d_2_2_5 * 100,
            width,
            label='2~2.5cm',
            color='red')
e = ax3.bar(x + 1.5 * width,
            d_2_5 * 100,
            width,
            label='>2.5cm',
            color='purple')
tkw = dict(size=5, width=1.5, direction='in', labelsize=16)
#plt.tick_params(axis='both', labelsize=15)
ax3.tick_params(axis='both', **tkw)
#plt.yscale("symlog")
ax3.set_xticks([2.0])
ax3.set_xticklabels(["CCN=2000"])
y = [0, 5, 10, 15]
t = ["0", "5%", "10%", "15%"]
ax3.set_yticks(y)
ax3.set_yticklabels(t)
# 坐标字体
#labels = ax1.get_xticklabels() + ax1.get_yticklabels()
labelx = ax3.get_xticklabels()
labely = ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labelx]
[label.set_fontname('Times New Roman') for label in labely]
# 横坐标下移
for tick in ax3.xaxis.get_major_ticks()[:]:
    tick.set_pad(12)
for line in ax3.xaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('black')
    line.set_markersize(7)
    line.set_markeredgewidth(2)
# 边框加粗
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax3.spines['left'].set_linewidth(bwith)
ax3.spines['top'].set_linewidth(bwith)
ax3.spines['right'].set_linewidth(bwith)
ax3.spines['bottom'].set_linewidth(bwith)
#-----------------------------------------------------
ax4 = ax.inset_axes([0.85, 0.5, 0.1432979, 0.4])
ax4.set_xlim(3 - 0.25, 3.5)
ax4.set_ylim(0, 2)
#a = ax4.bar(x - 1.5 * width,
#            d_0_5_1 * 100,
#            width,
#            label='0.5-1cm',
#            color='blue')
#b = ax4.bar(x - 0.75 * width,
#            d_1_1_5 * 100,
#            width,
#            label='1-1.5cm',
#            color='orange')
c = ax4.bar(x, d_1_5_2 * 100, width, label='1.5-2cm', color='green')
d = ax4.bar(x + 0.75 * width,
            d_2_2_5 * 100,
            width,
            label='2~2.5cm',
            color='red')
e = ax4.bar(x + 1.5 * width,
            d_2_5 * 100,
            width,
            label='>2.5cm',
            color='purple')
tkw = dict(size=5, width=1.5, direction='in', labelsize=16)
#plt.tick_params(axis='both', labelsize=15)
ax4.tick_params(axis='both', **tkw)
#plt.yscale("symlog")
ax4.set_xticks([3.0])
ax4.set_xticklabels(["CCN=10000"])
y = [0, 1, 2]
t = ["0", "1%", "2%"]
ax4.set_yticks(y)
ax4.set_yticklabels(t)
# 坐标字体
#labels = ax1.get_xticklabels() + ax1.get_yticklabels()
labelx = ax4.get_xticklabels()
labely = ax4.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labelx]
[label.set_fontname('Times New Roman') for label in labely]
# 横坐标下移
for tick in ax4.xaxis.get_major_ticks()[:]:
    tick.set_pad(12)
for line in ax4.xaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('black')
    line.set_markersize(7)
    line.set_markeredgewidth(2)
# 边框加粗
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax4.spines['left'].set_linewidth(bwith)
ax4.spines['top'].set_linewidth(bwith)
ax4.spines['right'].set_linewidth(bwith)
ax4.spines['bottom'].set_linewidth(bwith)
#-----------------------------------------------------

#ax.set_ylabel("percentage (%)", fontsize=25)
#y = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#t = [1, 5, 10, 15, 20, 40, 60, 70, 80, 90, 100]
#ax1.yticks(y, t)
#ax.indicate_inset_zoom(ax)
#x2 = [0, 2]
#y4 = [0, 20]
#new_ax = inset_axes(ax1,
#                    width="40%",
#                    height="20%",
#                    loc="lower left",
#                    bbox_to_anchor=(0.3, 0.1, 1, 1),
#                    bbox_transform=ax1.transAxes)
#new_ax.plot(x2, y4)

plt.savefig('bar-ccn%.png')
plt.close
