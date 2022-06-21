from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib
import numpy as np
import os
import math
matplotlib.rcParams['font.sans-serif'] = ['FangSong']
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

##### ccn=50
### 数据读取
fl = os.popen("wc -l <./ccn-50/data/Trajectory-0.5.txt")
fw = (os.popen("wc -w <./ccn-50/data/Trajectory-0.5.txt"))
data1 = np.loadtxt("./ccn-50/data/Trajectory-0.5.txt")
order = np.loadtxt("./ccn-50/order.txt") #聚类结果排序
nn = fl.read()
nn = int(nn)
#print("nn=", nn)
nt = fw.read()
nt = int(nt)
nt = nt / (nn * 4)
nt = int(nt)
#print("nt=", nt)
first_2000 = np.empty([nt, nn], dtype=float)
second_2000 = np.empty([nt, nn], dtype=float)
third_2000 = np.empty([nt, nn], dtype=float)
d = np.empty([nt, nn], dtype=float)
dlt = np.empty([nt, nn], dtype=float)
qc = np.empty([nt, nn], dtype=float)
qr = np.empty([nt, nn], dtype=float)
xw = np.empty([nt, nn], dtype=float)
xt = np.empty([nt, nn], dtype=float)
xpc2h = np.empty([nt, nn], dtype=float)
xpr2h = np.empty([nt, nn], dtype=float)
xphwet = np.empty([nt, nn], dtype=float)
# load data from file
# you replace this using with open
#----------------------------------------------------------
hail = np.loadtxt("./ccn-50/data/Hail-0.5.txt")
hydro = np.loadtxt("./ccn-50/data/Hydrometer-0.5.txt")
data1 = np.loadtxt("./ccn-50/data/Trajectory-0.5.txt")
first_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 0:nt] / 2)
second_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, nt:2*nt] / 2)
third_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 2*nt:3*nt] / 2)
d[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 3*nt:4*nt])

dlt[0:nt - 2, :] = (d[1:nt - 1, :] - d[0:nt - 2, :]) * 60 / 5
dlt[nt - 1, :] = 0

qc[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, 0:nt])
qr[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, nt:2*nt])
xw[:, 0:nn + 1] = 0.01 * np.transpose(hail[0:nn + 1, 0:nt])
xt[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, nt:2*nt]) - 273.0
xpc2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 2*nt:3*nt])
xpr2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 3*nt:4*nt])
xphwet[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 4*nt:5*nt])
#----------------------------------------------------------
# 获取y_pred
y_pred = np.loadtxt('./ccn-50/y_pred.txt', delimiter=',')
# delete some aberrant point
i = 0
j = 0
for i in range(nn):
    for j in range(nt):
        if np.any(first_2000[j, i] >= 999):
            first_2000[j, i] = 0
#----------------------------------------------------------
# xy坐标max，min'
xmax = np.array(first_2000).max()
xmin = np.array(first_2000).min()
#print('xmin=', xmin, '  ', 'xmax=', xmax)
ymax = np.array(second_2000).max()
ymin = np.array(second_2000).min()
#print('ymin=', ymin, '  ', 'ymax=', ymax)
x_co = (107.0287 - 104.9827) / 180
y_co = (36.88075 - 35.22662) / 180
first_2000 = 104.9827 + first_2000 * x_co
second_2000 = 35.22662 + second_2000 * y_co
## 5类轨迹
wmaxk1d = [0 for x in range(0, 5)]
for k in range(1):
    #print("K=", k)
    #    #print("第", k + 1, "类轨迹数=", len(iw[0]))
    #    #print("fuck", y_pred)
    #    #print("fuck", d[nt - 1, :])
    #    #print("fuck", len(y_pred))
    ##print("fuck", len(d[nt - 1, :]))
    iw1 = np.where((d[nt - 1, :] >= 0.5)
                   & (d[nt - 1, :] <= 1.0))
    iw2 = np.where((d[nt - 1, :] > 1.0)
                   & (d[nt - 1, :] <= 1.5))
    iw3 = np.where((d[nt - 1, :] > 1.5)
                   & (d[nt - 1, :] <= 2.0))
    iw4 = np.where((d[nt - 1, :] > 2.0)
                   & (d[nt - 1, :] <= 2.5))
    iw5 = np.where((d[nt - 1, :] > 2.5))
    # 被选中聚类轨迹的最大值
    diw = [iw1, iw2, iw3, iw4, iw5]
    for nd in range(5):
        wmaxk1d[nd] = np.amax(xw[:, diw[nd][0]], axis=0)
data = wmaxk1d

fig = plt.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.90)
ax = fig.gca()
#ax = plt.subplot(411)
#箱型图名称
labels = ["0.5~1", "1~1.5", "1.5~2", "2~2.5", ">2.5"]
#labels = ["I", "II", "III", "IV", "V"]
#三个箱型图的颜色 RGB （均为0~1的数据）
#colors = [(202 / 255., 96 / 255., 17 / 255.),
#          (255 / 255., 217 / 255., 102 / 255.),
#          (222 / 255., 128 / 255., 68 / 255.),
#          (102 / 255., 128 / 255., 68 / 255.),
#          (58 / 255., 128 / 255., 68 / 255.)]
colors = ["blue", "green", "yellow", "red", "purple"]
#绘制箱型图
#patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
bplot = ax.boxplot(data,
                   patch_artist=True,
                   labels=labels,
                   positions=(1.0, 1.5, 2.0, 2.5, 3.0),
                   showfliers=False,
                   widths=0.3)
#将三个箱分别上色
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)


x_position = [2, 5, 8, 11]
x_position_fmt = ["", "", "", ""]
plt.xticks([i for i in x_position], x_position_fmt, fontsize=18)

#plt.ylabel('CWC  ($\mathregular{10^{-1}}$ g.$\mathregular{kg^{-1}}$)',
#           fontsize=18)
plt.grid(linestyle="--", alpha=0.3) #绘制图中虚线 透明度0.3
plt.xlim(0, 13)
plt.ylim(0, 30)
ax.set_ylim(0,30)
y_major_locator = MultipleLocator(10)
ax.yaxis.set_major_locator(y_major_locator)


ax.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)
ax.spines['bottom'].set_linewidth(bwith)


##### ccn=200
### 数据读取
fl = os.popen("wc -l <./ccn-200/data/Trajectory-0.5.txt")
fw = (os.popen("wc -w <./ccn-200/data/Trajectory-0.5.txt"))
data1 = np.loadtxt("./ccn-200/data/Trajectory-0.5.txt")
order = np.loadtxt("./ccn-200/order.txt") #聚类结果排序
nn = fl.read()
nn = int(nn)
#print("nn=", nn)
nt = fw.read()
nt = int(nt)
nt = nt / (nn * 4)
nt = int(nt)
#print("nt=", nt)
first_2000 = np.empty([nt, nn], dtype=float)
second_2000 = np.empty([nt, nn], dtype=float)
third_2000 = np.empty([nt, nn], dtype=float)
d = np.empty([nt, nn], dtype=float)
dlt = np.empty([nt, nn], dtype=float)
qc = np.empty([nt, nn], dtype=float)
qr = np.empty([nt, nn], dtype=float)
xw = np.empty([nt, nn], dtype=float)
xt = np.empty([nt, nn], dtype=float)
xpc2h = np.empty([nt, nn], dtype=float)
xpr2h = np.empty([nt, nn], dtype=float)
xphwet = np.empty([nt, nn], dtype=float)
# load data from file
# you replace this using with open
#----------------------------------------------------------
hail = np.loadtxt("./ccn-200/data/Hail-0.5.txt")
hydro = np.loadtxt("./ccn-200/data/Hydrometer-0.5.txt")
data1 = np.loadtxt("./ccn-200/data/Trajectory-0.5.txt")
first_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 0:nt] / 2)
second_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, nt:2*nt] / 2)
third_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 2*nt:3*nt] / 2)
d[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 3*nt:4*nt])

dlt[0:nt - 2, :] = (d[1:nt - 1, :] - d[0:nt - 2, :]) * 60 / 5
dlt[nt - 1, :] = 0

qc[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, 0:nt])
qr[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, nt:2*nt])
xw[:, 0:nn + 1] = 0.01 * np.transpose(hail[0:nn + 1, 0:nt])
xt[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, nt:2*nt]) - 273.0
xpc2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 2*nt:3*nt])
xpr2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 3*nt:4*nt])
xphwet[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 4*nt:5*nt])
#----------------------------------------------------------
# 获取y_pred
y_pred = np.loadtxt('./ccn-200/y_pred.txt', delimiter=',')
# delete some aberrant point
i = 0
j = 0
for i in range(nn):
    for j in range(nt):
        if np.any(first_2000[j, i] >= 999):
            first_2000[j, i] = 0
#----------------------------------------------------------
# xy坐标max，min'
xmax = np.array(first_2000).max()
xmin = np.array(first_2000).min()
#print('xmin=', xmin, '  ', 'xmax=', xmax)
ymax = np.array(second_2000).max()
ymin = np.array(second_2000).min()
#print('ymin=', ymin, '  ', 'ymax=', ymax)
x_co = (107.0287 - 104.9827) / 180
y_co = (36.88075 - 35.22662) / 180
first_2000 = 104.9827 + first_2000 * x_co
second_2000 = 35.22662 + second_2000 * y_co
## 5类轨迹
wmaxk1d = [0 for x in range(0, 5)]
for k in range(1):
    #print("K=", k)
    #    #print("第", k + 1, "类轨迹数=", len(iw[0]))
    #    #print("fuck", y_pred)
    #    #print("fuck", d[nt - 1, :])
    #    #print("fuck", len(y_pred))
    ##print("fuck", len(d[nt - 1, :]))
    iw1 = np.where((d[nt - 1, :] >= 0.5)
                   & (d[nt - 1, :] <= 1.0))
    iw2 = np.where((d[nt - 1, :] > 1.0)
                   & (d[nt - 1, :] <= 1.5))
    iw3 = np.where((d[nt - 1, :] > 1.5)
                   & (d[nt - 1, :] <= 2.0))
    iw4 = np.where((d[nt - 1, :] > 2.0)
                   & (d[nt - 1, :] <= 2.5))
    iw5 = np.where((d[nt - 1, :] > 2.5))
    # 被选中聚类轨迹的最大值
    diw = [iw1, iw2, iw3, iw4, iw5]
    for nd in range(5):
        wmaxk1d[nd] = np.amax(xw[:, diw[nd][0]], axis=0)

data = wmaxk1d

#箱型图名称
labels = ["0.5~1", "1~1.5", "1.5~2", "2~2.5", ">2.5"]
#labels = ["I", "II", "III", "IV", "V"]
#三个箱型图的颜色 RGB （均为0~1的数据）
#colors = [(202 / 255., 96 / 255., 17 / 255.),
#          (255 / 255., 217 / 255., 102 / 255.),
#          (222 / 255., 128 / 255., 68 / 255.),
#          (102 / 255., 128 / 255., 68 / 255.),
#          (58 / 255., 128 / 255., 68 / 255.)]
colors = ["blue", "green", "yellow", "red", "purple"]
#绘制箱型图
#patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
bplot = ax.boxplot(data,
                   patch_artist=True,
                   labels=labels,
                   positions=(4.0, 4.5, 5.0, 5.5, 6.0),
                   showfliers=False,
                   widths=0.3)
#将三个箱分别上色
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

x_position = [2, 5, 8, 11]
x_position_fmt = ["", "", "", ""]
plt.xticks([i for i in x_position], x_position_fmt, fontsize=18)

#plt.ylabel('CWC  ($\mathregular{10^{-1}}$ g.$\mathregular{kg^{-1}}$)',
#           fontsize=18)
plt.grid(linestyle="--", alpha=0.3) #绘制图中虚线 透明度0.3
plt.xlim(0, 13)
plt.ylim(0, 30)
ax.set_ylim(0,30)
y_major_locator = MultipleLocator(10)
ax.yaxis.set_major_locator(y_major_locator)

ax.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)
ax.spines['bottom'].set_linewidth(bwith)
#plt.show()


##### ccn=2000
### 数据读取
fl = os.popen("wc -l <./ccn-2000/data/Trajectory-0.5.txt")
fw = (os.popen("wc -w <./ccn-2000/data/Trajectory-0.5.txt"))
data1 = np.loadtxt("./ccn-2000/data/Trajectory-0.5.txt")
order = np.loadtxt("./ccn-2000/order.txt") #聚类结果排序
nn = fl.read()
nn = int(nn)
#print("nn=", nn)
nt = fw.read()
nt = int(nt)
nt = nt / (nn * 4)
nt = int(nt)
#print("nt=", nt)
first_2000 = np.empty([nt, nn], dtype=float)
second_2000 = np.empty([nt, nn], dtype=float)
third_2000 = np.empty([nt, nn], dtype=float)
d = np.empty([nt, nn], dtype=float)
dlt = np.empty([nt, nn], dtype=float)
qc = np.empty([nt, nn], dtype=float)
qr = np.empty([nt, nn], dtype=float)
xw = np.empty([nt, nn], dtype=float)
xt = np.empty([nt, nn], dtype=float)
xpc2h = np.empty([nt, nn], dtype=float)
xpr2h = np.empty([nt, nn], dtype=float)
xphwet = np.empty([nt, nn], dtype=float)
# load data from file
# you replace this using with open
#----------------------------------------------------------
hail = np.loadtxt("./ccn-2000/data/Hail-0.5.txt")
hydro = np.loadtxt("./ccn-2000/data/Hydrometer-0.5.txt")
data1 = np.loadtxt("./ccn-2000/data/Trajectory-0.5.txt")
first_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 0:nt] / 2)
second_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, nt:2*nt] / 2)
third_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 2*nt:3*nt] / 2)
d[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 3*nt:4*nt])

dlt[0:nt - 2, :] = (d[1:nt - 1, :] - d[0:nt - 2, :]) * 60 / 5
dlt[nt - 1, :] = 0

qc[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, 0:nt])
qr[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, nt:2*nt])
xw[:, 0:nn + 1] = 0.01 * np.transpose(hail[0:nn + 1, 0:nt])
xt[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, nt:2*nt]) - 273.0
xpc2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 2*nt:3*nt])
xpr2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 3*nt:4*nt])
xphwet[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 4*nt:5*nt])
#----------------------------------------------------------
# 获取y_pred
y_pred = np.loadtxt('./ccn-2000/y_pred.txt', delimiter=',')
# delete some aberrant point
i = 0
j = 0
for i in range(nn):
    for j in range(nt):
        if np.any(first_2000[j, i] >= 999):
            first_2000[j, i] = 0
#----------------------------------------------------------
# xy坐标max，min'
xmax = np.array(first_2000).max()
xmin = np.array(first_2000).min()
#print('xmin=', xmin, '  ', 'xmax=', xmax)
ymax = np.array(second_2000).max()
ymin = np.array(second_2000).min()
#print('ymin=', ymin, '  ', 'ymax=', ymax)
x_co = (107.0287 - 104.9827) / 180
y_co = (36.88075 - 35.22662) / 180
first_2000 = 104.9827 + first_2000 * x_co
second_2000 = 35.22662 + second_2000 * y_co
## 5类轨迹
wmaxk1d = [0 for x in range(0, 5)]
for k in range(1):
    #print("K=", k)
    #    #print("第", k + 1, "类轨迹数=", len(iw[0]))
    #    #print("fuck", y_pred)
    #    #print("fuck", d[nt - 1, :])
    #    #print("fuck", len(y_pred))
    ##print("fuck", len(d[nt - 1, :]))
    iw1 = np.where((d[nt - 1, :] >= 0.5)
                   & (d[nt - 1, :] <= 1.0))
    iw2 = np.where((d[nt - 1, :] > 1.0)
                   & (d[nt - 1, :] <= 1.5))
    iw3 = np.where((d[nt - 1, :] > 1.5)
                   & (d[nt - 1, :] <= 2.0))
    iw4 = np.where((d[nt - 1, :] > 2.0)
                   & (d[nt - 1, :] <= 2.5))
    iw5 = np.where((d[nt - 1, :] > 2.5))
    # 被选中聚类轨迹的最大值
    diw = [iw1, iw2, iw3, iw4, iw5]
    for nd in range(5):
        wmaxk1d[nd] = np.amax(xw[:, diw[nd][0]], axis=0)

data = wmaxk1d

#箱型图名称
labels = ["0.5~1", "1~1.5", "1.5~2", "2~2.5", ">2.5"]
#labels = ["I", "II", "III", "IV", "V"]
#三个箱型图的颜色 RGB （均为0~1的数据）
#colors = [(202 / 255., 96 / 255., 17 / 255.),
#          (255 / 255., 217 / 255., 102 / 255.),
#          (222 / 255., 128 / 255., 68 / 255.),
#          (102 / 255., 128 / 255., 68 / 255.),
#          (58 / 255., 128 / 255., 68 / 255.)]
colors = ["blue", "green", "yellow", "red", "purple"]
#绘制箱型图
#patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
bplot = ax.boxplot(data,
                   patch_artist=True,
                   labels=labels,
                   positions=(7.0, 7.5, 8.0, 8.5, 9.0),
                   showfliers=False,
                   widths=0.3)
#将三个箱分别上色
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

x_position = [2, 5, 8, 11]
x_position_fmt = ["", "", "", ""]
plt.xticks([i for i in x_position], x_position_fmt, fontsize=18)

#plt.ylabel('CWC  ($\mathregular{10^{-1}}$ g.$\mathregular{kg^{-1}}$)',
#           fontsize=18)
plt.grid(linestyle="--", alpha=0.3) #绘制图中虚线 透明度0.3
plt.xlim(0, 13)
plt.ylim(0, 30)
ax.set_ylim(0,30)
y_major_locator = MultipleLocator(10)
ax.yaxis.set_major_locator(y_major_locator)

ax.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)
ax.spines['bottom'].set_linewidth(bwith)
#plt.show()

##### ccn=10000
### 数据读取
fl = os.popen("wc -l <./ccn-10000/data/Trajectory-0.5.txt")
fw = (os.popen("wc -w <./ccn-10000/data/Trajectory-0.5.txt"))
data1 = np.loadtxt("./ccn-10000/data/Trajectory-0.5.txt")
order = np.loadtxt("./ccn-10000/order.txt") #聚类结果排序
nn = fl.read()
nn = int(nn)
#print("nn=", nn)
nt = fw.read()
nt = int(nt)
nt = nt / (nn * 4)
nt = int(nt)
#print("nt=", nt)
first_2000 = np.empty([nt, nn], dtype=float)
second_2000 = np.empty([nt, nn], dtype=float)
third_2000 = np.empty([nt, nn], dtype=float)
d = np.empty([nt, nn], dtype=float)
dlt = np.empty([nt, nn], dtype=float)
qc = np.empty([nt, nn], dtype=float)
qr = np.empty([nt, nn], dtype=float)
xw = np.empty([nt, nn], dtype=float)
xt = np.empty([nt, nn], dtype=float)
xpc2h = np.empty([nt, nn], dtype=float)
xpr2h = np.empty([nt, nn], dtype=float)
xphwet = np.empty([nt, nn], dtype=float)
# load data from file
# you replace this using with open
#----------------------------------------------------------
hail = np.loadtxt("./ccn-10000/data/Hail-0.5.txt")
hydro = np.loadtxt("./ccn-10000/data/Hydrometer-0.5.txt")
data1 = np.loadtxt("./ccn-10000/data/Trajectory-0.5.txt")
first_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 0:nt] / 2)
second_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, nt:2*nt] / 2)
third_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 2*nt:3*nt] / 2)
d[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 3*nt:4*nt])

dlt[0:nt - 2, :] = (d[1:nt - 1, :] - d[0:nt - 2, :]) * 60 / 5
dlt[nt - 1, :] = 0

qc[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, 0:nt])
qr[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, nt:2*nt])
xw[:, 0:nn + 1] = 0.01 * np.transpose(hail[0:nn + 1, 0:nt])
xt[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, nt:2*nt]) - 273.0
xpc2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 2*nt:3*nt])
xpr2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 3*nt:4*nt])
xphwet[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 4*nt:5*nt])
#----------------------------------------------------------
# 获取y_pred
y_pred = np.loadtxt('./ccn-10000/y_pred.txt', delimiter=',')
# delete some aberrant point
i = 0
j = 0
for i in range(nn):
    for j in range(nt):
        if np.any(first_2000[j, i] >= 999):
            first_2000[j, i] = 0
#----------------------------------------------------------
# xy坐标max，min'
xmax = np.array(first_2000).max()
xmin = np.array(first_2000).min()
#print('xmin=', xmin, '  ', 'xmax=', xmax)
ymax = np.array(second_2000).max()
ymin = np.array(second_2000).min()
#print('ymin=', ymin, '  ', 'ymax=', ymax)
x_co = (107.0287 - 104.9827) / 180
y_co = (36.88075 - 35.22662) / 180
first_2000 = 104.9827 + first_2000 * x_co
second_2000 = 35.22662 + second_2000 * y_co
## 5类轨迹
wmaxk1d = [0 for x in range(0, 5)]
for k in range(1):
    #print("K=", k)
    #    #print("第", k + 1, "类轨迹数=", len(iw[0]))
    #    #print("fuck", y_pred)
    #    #print("fuck", d[nt - 1, :])
    #    #print("fuck", len(y_pred))
    ##print("fuck", len(d[nt - 1, :]))
    iw1 = np.where((d[nt - 1, :] >= 0.5)
                   & (d[nt - 1, :] <= 1.0))
    iw2 = np.where((d[nt - 1, :] > 1.0)
                   & (d[nt - 1, :] <= 1.5))
    iw3 = np.where((d[nt - 1, :] > 1.5)
                   & (d[nt - 1, :] <= 2.0))
    iw4 = np.where((d[nt - 1, :] > 2.0)
                   & (d[nt - 1, :] <= 2.5))
    iw5 = np.where((d[nt - 1, :] > 2.5))
    # 被选中聚类轨迹的最大值
    diw = [iw1, iw2, iw3, iw4, iw5]
    for nd in range(5):
        wmaxk1d[nd] = np.amax(xw[:, diw[nd][0]], axis=0)

data = wmaxk1d

#箱型图名称
labels = ["0.5~1", "1~1.5", "1.5~2", "2~2.5", ">2.5"]
#labels = ["I", "II", "III", "IV", "V"]
#三个箱型图的颜色 RGB （均为0~1的数据）
#colors = [(202 / 255., 96 / 255., 17 / 255.),
#          (255 / 255., 217 / 255., 102 / 255.),
#          (222 / 255., 128 / 255., 68 / 255.),
#          (102 / 255., 128 / 255., 68 / 255.),
#          (58 / 255., 128 / 255., 68 / 255.)]
colors = ["blue", "green", "yellow", "red", "purple"]
#绘制箱型图
#patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
bplot = ax.boxplot(data,
                   patch_artist=True,
                   labels=labels,
                   positions=(10.0, 10.5, 11.0, 11.5, 12.0),
                   showfliers=False,
                   widths=0.3)
#将三个箱分别上色
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)


x_position = [2, 5, 8, 11]
#x_position_fmt = ["I", "II", "III", "VI", "V"]
x_position_fmt = ["CCN=50", "CCN=200", "CCN=2000", "CCN=10000"]
plt.xticks([i for i in x_position], x_position_fmt, fontsize=18)

#plt.ylabel('CWC  ($\mathregular{10^{-1}}$ g.$\mathregular{kg^{-1}}$)',
#           fontsize=18)
plt.grid(linestyle="--", alpha=0.3) #绘制图中虚线 透明度0.3
plt.xlim(0, 13)
plt.ylim(0, 30)
ax.set_ylim(0,30)
y_major_locator = MultipleLocator(10)
ax.yaxis.set_major_locator(y_major_locator)

ax.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)
ax.spines['bottom'].set_linewidth(bwith)
plt.ylabel('W  (m.$\mathregular{{s^{-1}}}$)',

           fontsize=18)

plt.savefig(fname="W-box.png")
