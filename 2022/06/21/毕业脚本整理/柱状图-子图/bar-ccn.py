import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
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

d_0_5_1 = [7004, 5621, 6078, 5546]
d_1_1_5 = [1147, 1129, 1404, 882]
d_1_5_2 = [296, 226, 766, 90]
d_2_2_5 = [82, 31, 322, 11]
d_2_5 = [21, 12, 31, 5]

ax = plt.gca()
x = np.arange(len(labels))
print(x)
width = 0.2
plt.bar(x - 1.5 * width, d_0_5_1, width, label='0.5~1cm', color='blue')
plt.bar(x - 0.75 * width, d_1_1_5, width, label='1~1.5cm', color='orange')
plt.bar(x, d_1_5_2, width, label='1.5~2cm', color='green')
plt.bar(x + 0.75 * width, d_2_2_5, width, label='2~2.5cm', color='red')
plt.bar(x + 1.5 * width, d_2_5, width, label='>2.5cm', color='purple')

# 调节label大小
tkw = dict(size=5, width=1.5, direction='in', labelsize=25)
#plt.tick_params(axis='both', labelsize=15)
plt.tick_params(axis='both', **tkw)

plt.yscale("log")
#plt.ylim(1, 10000)
#ax.set_ylim(1, 10000)
plt.xticks(x, labels=labels)

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

ax.set_ylabel("Count", fontsize=30)
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

plt.savefig('bar-ccn.png')
plt.close
