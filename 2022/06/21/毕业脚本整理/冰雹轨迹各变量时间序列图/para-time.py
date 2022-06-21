from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt
import numpy as np
import os
import math


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)



### 数据读取
#----------------------------------------------------------
fl = os.popen("wc -l <./data/Trajectory-0.5.txt")
fw = (os.popen("wc -w <./data/Trajectory-0.5.txt"))
data1 = np.loadtxt("./data/Trajectory-0.5.txt")
order = np.loadtxt("./order.txt") #聚类结果排序
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
hail = np.loadtxt("./data/Hail-0.5.txt")
hydro = np.loadtxt("./data/Hydrometer-0.5.txt")
data1 = np.loadtxt("./data/Trajectory-0.5.txt")
first_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 0:nt] / 2)
second_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, nt:2*nt] / 2)
third_2000[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 2*nt:3*nt] / 2)
d[:, 0:nn + 1] = np.transpose(data1[0:nn + 1, 3*nt:4*nt])

dlt[0:nt - 2, :] = (d[1:nt - 1, :] - d[0:nt - 2, :]) * 60 / 5
dlt[nt - 1, :] = 0

qc[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, 0:nt])
qr[:, 0:nn + 1] = 1000 * np.transpose(hydro[0:nn + 1, nt:2*nt])
xw[:, 0:nn + 1] = 0.01 * np.transpose(hail[0:nn + 1, 0:nt])
xt[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, nt:2*nt])-273.0
xpc2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 2*nt:3*nt])
xpr2h[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 3*nt:4*nt])
xphwet[:, 0:nn + 1] = 1 * np.transpose(hail[0:nn + 1, 4*nt:5*nt])
#----------------------------------------------------------
# 坐标更改，变量修改
#qr = 100 * qr
#xpc2h = 100 * xpc2h
#xpr2h = 100000 * xpr2h
#----------------------------------------------------------
# 获取y_pred
y_pred = np.loadtxt('y_pred.txt', delimiter=',')
#----------------------------------------------------------
# print hail's diameter
print(len(d))
#----------------------------------------------------------
# new a figure and set it into 3d
# draw the figure, the color is r = read
dmax = np.array(d).max()
dmin = np.array(d).min()
#----------------------------------------------------------
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
print('xmin=', xmin, '  ', 'xmax=', xmax)
ymax = np.array(second_2000).max()
ymin = np.array(second_2000).min()
print('ymin=', ymin, '  ', 'ymax=', ymax)
x_co = (107.0287 - 104.9827) / 180
y_co = (36.88075 - 35.22662) / 180
first_2000 = 104.9827 + first_2000 * x_co
second_2000 = 35.22662 + second_2000 * y_co

#----------------------------------------------------------
mm = -1
#for i in range(nn):
for k in range(5):
#for k in range(1):
#for k in range(2,3):
    # new a figure and set it into 3d
    print("K=", k)
    iw = np.where(y_pred[:] == order[k])

    ### 计算大于0.1cm的时刻
    it = np.empty([len(iw[0])], dtype=int)
    for i in range(len(iw[0])):
        #        print("d[:,iw].shape=", d[:, iw].shape)
        #        print(np.where(d[:, iw][:, 0, i] >= 0.1))
        it[i] = min(min(np.where(d[:, iw][:, 0, i] >= 0.1)))
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
    if (np.array(diwmax).max() >= 0.5):
        iwiw1 = np.where((diwmax[:] >= 0.5) & (diwmax[:] <= 1.0))
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
    # choose color for hail's diameter
    ##iw = np.where(y_pred[:] == k)
    ##maxd = max(max(d[nt - 1, iw]))
    ## add

    sn = [i11,i22,i33,i44,i55]
    print("sn=",sn)
    for pp in range(5):
#    for pp in range(1):
        print("pp=",pp)
        ss = sn[pp]
        if (ss == 999999):
            continue
        print('iw=', iw)
        print(third_2000[:, ss])
        nzz = np.array(third_2000[:,ss].min()) ### 最底层
#        ntt = min(min(np.where(third_2000[:, ss] <= nzz)))
        ntt = max(max(max(np.where(dlt[:, ss] > 0.0))),min(min(np.where(third_2000[:, ss] <= nzz))))
        if (np.array(d[:,ss]).min()<=0.1):
            tt0_1 = max(max(np.where(d[:, ss] <= 0.11))) # 0.1cm对应时间
        if (np.array(d[:,ss]).min()>0.1):
            tt0_1 = 1
        #itt = int((ntt + 10) * 50 / 60) # 多10时刻空白的作为预留空间，itt表示分钟
        itt = int((ntt + 2) * 50 / 60) # 多10时刻空白的作为预留空间，itt表示分钟
        ntt = int(itt * 60 / 50) #itt分钟对应的矩阵时刻数
        print("itt=",itt,"min")
        print("ntt=",ntt)
        #----------------------------------------------------------
        fig, ax_hei = plt.subplots(dpi=400, figsize=(13, 5)) #设置画布长12宽4，分辨率为400
        fig.subplots_adjust(left=0.07, bottom=0.18, right=0.48, top=0.73)
        
        ax_dia = ax_hei.twinx()
        ax_dlt = ax_hei.twinx()
        ax_cloud = ax_hei.twinx()
        ax_rain = ax_hei.twinx()
        ax_vcloud = ax_hei.twinx()
        ax_vrain = ax_hei.twinx()
        ax_wwind = ax_hei.twinx()
        ax_tem = ax_hei.twinx()
        
        # Offset the right spine of par2.  The ticks and label have already been
        # placed on the right by twinx above.
        ax_dia.spines["right"].set_position(("axes", 1.0))
        ax_dlt.spines["right"].set_position(("axes", 1.15))
        ax_cloud.spines["right"].set_position(("axes", 1.30))
        ax_rain.spines["right"].set_position(("axes", 1.45))
        ax_vcloud.spines["right"].set_position(("axes", 1.60))
        ax_vrain.spines["right"].set_position(("axes", 1.75))
        ax_wwind.spines["right"].set_position(("axes", 1.90))
        ax_tem.spines["right"].set_position(("axes", 2.08))
        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        make_patch_spines_invisible(ax_dia)
        make_patch_spines_invisible(ax_dlt)
        make_patch_spines_invisible(ax_cloud)
        make_patch_spines_invisible(ax_rain)
        make_patch_spines_invisible(ax_vcloud)
        make_patch_spines_invisible(ax_vrain)
        make_patch_spines_invisible(ax_wwind)
        make_patch_spines_invisible(ax_tem)
        # Second, show the right spine.
        ax_hei.spines["right"].set_visible(False)
        ax_hei.spines["top"].set_visible(False)
        ax_dia.spines["right"].set_visible(True)
        ax_dlt.spines["right"].set_visible(True)
        ax_cloud.spines["right"].set_visible(True)
        ax_rain.spines["right"].set_visible(True)
        ax_vcloud.spines["right"].set_visible(True)
        ax_vrain.spines["right"].set_visible(True)
        ax_wwind.spines["right"].set_visible(True)
        ax_tem.spines["right"].set_visible(True)
        
        #set label for axis
        ax_hei.set_ylabel('Height  (km)', fontsize=18)
        ax_hei.set_xlabel('Times  (minutes)', fontsize=18)
        ax_dia.set_ylabel('Hail Diameter  (cm)', fontsize=18)
        ax_dlt.set_ylabel('R_Hail Diameter  (cm.$\mathregular{min^{-1}}$)', fontsize=18)
        ax_cloud.set_ylabel('CWC  (g.$\mathregular{kg^{-1}}$)', fontsize=18)
        qr_labelorder=int(math.floor(math.log10(np.array(qr[0:ntt, ss]).max())))
        print("qr_labelorder",qr_labelorder)
#        ax_rain.set_ylabel('RWC     ($\mathregular{10^{'+str(qr_labelorder)+'}}$ g/kg)', fontsize=18)
        ax_rain.set_ylabel('RWC  ($\mathregular{10^{-1}}$ g.$\mathregular{kg^{-1}}$)', fontsize=18)
        ## 应对等于0情况
        vc_labelorder = -3
        if (np.array(xpc2h[0:ntt,ss]).max() > 0.0):
            vc_labelorder=int(math.floor(math.log10(np.array(xpc2h[0:ntt, ss]).max())))
        print("vc_labelorder",vc_labelorder)
        ax_vcloud.set_ylabel('R_CWC  ($\mathregular{10^{'+str(vc_labelorder)+'}}$ g.$\mathregular{s^{-1}}$)', fontsize=18)
        ## 应对等于0情况
        vr_labelorder = -6
        if (np.array(xpr2h[0:ntt,ss]).max() > 0.0):
            vr_labelorder=int(math.floor(math.log10(np.array(xpr2h[0:ntt, ss]).max())))
        print("vr_labelorder",vr_labelorder)
        ax_vrain.set_ylabel('R_RWC  ($\mathregular{10^{'+str(vr_labelorder)+'}}$ g.$\mathregular{s^{-1}}$)', fontsize=18)
        ax_wwind.set_ylabel('W  (m.$\mathregular{{s^{-1}}}$)', fontsize=18)
        ax_tem.set_ylabel('Temperature  ($^\circ$C)', fontsize=18)

        curve_hei, = ax_hei.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                 third_2000[0:ntt, ss],
                                 label="Height",
                                 color='black',
                                 lw=2.0)
        curve_dia, = ax_dia.plot(np.linspace(1, ntt, ntt) * 0+tt0_1*50/60,
                                 np.linspace(0,4, ntt),
                                 label="Hail Diameter=0.1",
                                 color='#FF7F24',
                                 linestyle='--',
                                 lw=2.0)
        curve_dia, = ax_dia.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                 d[0:ntt, ss],
                                 label="Hail Diameter",
                                 color='#FF7F24',
                                 lw=2.0)
        curve_dlt, = ax_dlt.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                 dlt[0:ntt, ss],
                                 label="R_Hail Diameter",
                                 color='red',
                                 lw=2.0)
        curve_cloud, = ax_cloud.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                     qc[0:ntt, ss],
                                     label="CWC",
                                     color='palegreen',
                                     lw=2.0)
        curve_rain, = ax_rain.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                   10**(-1*qr_labelorder)*qr[0:ntt, ss],
                                   label="RWC",
                                   color='lightskyblue',
                                   lw=2.0)
        curve_vcloud, = ax_vcloud.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                       10**(-1*vc_labelorder)*xpc2h[0:ntt, ss],
                                       label="R_CWC",
                                       color='green',
                                       lw=2.0)
        curve_vrain, = ax_vrain.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                     10**(-1*vr_labelorder)*xpr2h[0:ntt, ss],
                                     label="R_RWC",
                                     color='darkblue',
                                     lw=2.0)
        curve_wwind, = ax_wwind.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                     np.linspace(1, ntt, ntt) * 0,
                                     label="W=0",
                                     color='#CD1076',
                                     linestyle='--',
                                     lw=2.0)
        curve_wwind, = ax_wwind.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                     xw[0:ntt, ss],
                                     label="W",
                                     color='#CD1076',
                                     lw=2.0)
        curve_tem, = ax_tem.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                     xt[0:ntt, ss]*0,
                                     label="Temperature=0",
                                     color='#8e7618',
                                     linestyle='--',
                                     lw=2.0)
        curve_tem, = ax_tem.plot(np.linspace(1, ntt, ntt) * 50 / 60,
                                     xt[0:ntt, ss],
                                     label="Temperature",
                                     color='#8e7618',
                                     lw=2.0)
    
        n1 = os.system('head -100 ./para-time.py | grep "Trajectory-0.5-1.txt"')
        n2 = os.system('head -100 ./para-time.py | grep "Trajectory-1-1.5.txt"')
        n3 = os.system('head -100 ./para-time.py | grep "Trajectory-1.5-2.txt"')
        n4 = os.system('head -100 ./para-time.py | grep "Trajectory-2-2.5.txt"')
        n5 = os.system('head -100 ./para-time.py | grep "Trajectory-2.5.txt"')
        print("n1=", n1)
        print("n2=", n2)
        print("n3=", n3)
        print("n4=", n4)
        print("n5=", n5)
        # 坐标轴范围
        if (n1 == 0):
            ax_hei.set_xlim(0, itt)
            ax_hei.set_ylim(0, 8)
            ax_dia.set_ylim(0, 2)
            ax_dia.set_yticks([0, 1, 2])
            ax_dlt.set_ylim(0, 1.0, 3)
            ax_cloud.set_ylim(0, 8)
            ax_rain.set_ylim(0, 0.8)
            ax_vcloud.set_ylim(0, 0.01)
            ax_vcloud.set_yticks([0, 0.002, 0.004, 0.006, 0.008, 0.010])
            ax_vrain.set_ylim(0, 0.0003)
            ax_vrain.set_yticks([0, 0.0001, 0.0002, 0.0003])
            ax_wwind.set_ylim(-10, 20)
            ax_tem.set_ylim(-50, 20)
        if (n2 == 0):
            ax_hei.set_xlim(0, itt)
            ax_hei.set_ylim(0, 8)
            ax_dia.set_ylim(0, 2)
            ax_dia.set_yticks([0, 1, 2])
            ax_dlt.set_ylim(0, 2, 3)
            ax_cloud.set_ylim(0, 6)
            ax_rain.set_ylim(0, 0.8)
            ax_vcloud.set_ylim(0, 0.01)
            ax_vcloud.set_yticks([0, 0.002, 0.004, 0.006, 0.008, 0.010])
            ax_vrain.set_ylim(0, 0.00050)
            ax_vrain.set_yticks([0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005])
            ax_wwind.set_ylim(-10, 20)
            ax_tem.set_ylim(-50, 20)
        if (n3 == 0):
            ax_hei.set_xlim(0, itt)
            ax_hei.set_ylim(0, 10)
            ax_dia.set_ylim(0, 3)
            ax_dia.set_yticks([0, 1, 2, 3])
            ax_dlt.set_ylim(0, 2, 3)
            ax_cloud.set_ylim(0, 8)
            ax_rain.set_ylim(0, 0.5)
            ax_vcloud.set_ylim(0, 0.02)
            ax_vcloud.set_yticks([0, 0.005, 0.010, 0.015, 0.020])
            ax_vrain.set_ylim(0, 0.00080)
            ax_vrain.set_yticks([0, 0.0002, 0.0004, 0.0006, 0.0008])
            ax_wwind.set_ylim(-10, 30)
            ax_tem.set_ylim(-50, 20)
        if (n4 == 0):
            ax_hei.set_xlim(0, itt)
            ax_hei.set_ylim(0, 8)
            ax_dia.set_ylim(0, 3)
            ax_dia.set_yticks([0, 1, 2, 3])
            ax_dlt.set_ylim(0, 1.5, 3)
            ax_cloud.set_ylim(0, 6)
            ax_rain.set_ylim(0, 0.8)
            ax_vcloud.set_ylim(0, 0.03)
            ax_vcloud.set_yticks([0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030])
            ax_vrain.set_ylim(0, 0.00080)
            ax_vrain.set_yticks([0, 0.0002, 0.0004, 0.0006, 0.0008])
            ax_wwind.set_ylim(-10, 30)
            ax_tem.set_ylim(-50, 20)
        if (n5 == 0):
            ax_hei.set_xlim(0, itt)
            ax_hei.set_ylim(0, 8)
            ax_dia.set_ylim(0, 4)
            ax_dia.set_yticks([0, 1, 2, 3, 4])
            ax_dlt.set_ylim(0, 2, 3)
            ax_cloud.set_ylim(0, 8)
            ax_rain.set_ylim(0, 0.6)
            ax_vcloud.set_ylim(0, 0.05)
            ax_vcloud.set_yticks([0, 0.010, 0.020, 0.030, 0.040, 0.050])
            ax_vrain.set_ylim(0, 0.0012)
            ax_vrain.set_yticks(
                [0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0012])
            ax_wwind.set_ylim(-10, 30)
            ax_tem.set_ylim(-50, 20)

        # 坐标轴标签范围
        maxhei = np.array(third_2000[0:ntt, ss]).max()
        print("maxHei",maxhei)
        maxhei = int(math.ceil(maxhei/2)*2)
        print("maxHei",maxhei)
        maxdia = np.array(d[0:ntt, ss]).max()
        print("maxDia",maxdia)
        maxdia = int(math.ceil(maxdia/2)*2)
        print("maxDia",maxdia)
        ax_hei.set_xlim(0, itt)
        maxdlt = np.array(dlt[0:ntt, ss]).max()
        print("maxdlt",maxdlt)
        maxdlt = int(math.ceil(maxdlt/2)*2)
        print("maxdlt",maxdlt)
        maxqc = np.array(qc[0:ntt, ss]).max()
        print("maxqc",maxqc)
        maxqc = int(math.ceil(maxqc/2)*2)
        print("maxqc",maxqc)
        maxqr = 10**(-1*qr_labelorder)*np.array(qr[0:ntt, ss]).max()
        print("maxqr",maxqr)
        maxqr = int(math.ceil(maxqr/2)*2)
        print("maxqr",maxqr)
        maxxpc2h = 10**(-1*vc_labelorder)*np.array(xpc2h[0:ntt, ss]).max()
        print("maxxpc2h",maxxpc2h)
        maxxpc2h = int(math.ceil(maxxpc2h/2)*2)
        print("maxxpc2h",maxxpc2h)
        maxxpr2h = 10**(-1*vr_labelorder)*np.array(xpr2h[0:ntt, ss]).max()
        print("maxxpr2h",maxxpr2h)
        maxxpr2h = int(math.ceil(maxxpr2h/2)*2)
        print("maxxpr2h",maxxpr2h)
        maxxw = np.array(xw[0:ntt, ss]).max()
        print("maxxw",maxxw)
        maxxw = int(math.ceil(maxxw/10)*10)
        print("maxxw",maxxw)
        ax_hei.set_xlim(0, itt)
#        ax_hei.set_ylim(0, maxhei)
#        ax_hei.set_yticks(np.arange(0,maxhei+0.1,2))
        ax_hei.set_ylim(0, 10)
        ax_hei.set_yticks(np.arange(0,10+0.1,2))
#        ax_dia.set_ylim(0, maxdia)
#        ax_dia.set_yticks(np.arange(0,maxdia+0.1,1))
        ax_dia.set_ylim(0, 4)
        ax_dia.set_yticks(np.arange(0,4+0.1,1))
#        ax_dlt.set_ylim(0, maxdlt)
#        ax_dlt.set_yticks(np.arange(0,maxdlt+0.1,1))
        ax_dlt.set_ylim(0, 2)
        ax_dlt.set_yticks(np.arange(0,2+0.1,1))
#        ax_cloud.set_ylim(0, maxqc)
#        arr = 1
#        if (maxqc>=4):
#            arr = 2
#        ax_cloud.set_yticks(np.arange(0,maxqc+0.1,arr))
        ax_cloud.set_ylim(0, 8)
        ax_cloud.set_yticks(np.arange(0,8+0.1,2))
#        arr = 1
#        if (maxqr>=4):
#            arr = 2
#        ax_rain.set_ylim(0, maxqr)
#        ax_rain.set_yticks(np.arange(0,maxqr+0.1,arr))
        ax_rain.set_ylim(0, 12)
        ax_rain.set_yticks(np.arange(0,12+0.1,4))
        arr = 1
        if (maxxpc2h>=4):
            arr = 2
        ax_vcloud.set_ylim(0, maxxpc2h)
        ax_vcloud.set_yticks(np.arange(0,maxxpc2h+0.1,arr))
        if (maxxpc2h==0):
            ax_vcloud.set_yticks(np.arange(0,2.1,1))
        arr = 1
        if (maxxpr2h>=4):
            arr = 2
        ax_vrain.set_ylim(0, maxxpr2h)
        ax_vrain.set_yticks(np.arange(0,maxxpr2h+0.1,arr))
        if (maxxpr2h==0):
            ax_vrain.set_yticks(np.arange(0,2.1,1))
#        ax_wwind.set_ylim(-10, maxxw)
#        ax_wwind.set_yticks(np.arange(-10,maxxw+0.1,10))
        ax_wwind.set_ylim(-10, 30)
        ax_wwind.set_yticks(np.arange(-10,30+0.1,10))
        ax_tem.set_ylim(-50, 20)
        ax_tem.set_yticks(np.arange(-50,20+0.1,10))


        # 坐标轴标签
        ax_hei.ticklabel_format(style='sci', useOffset=False)
        ax_hei.yaxis.label.set_color(curve_hei.get_color())
        ax_dia.yaxis.label.set_color(curve_dia.get_color())
        ax_dlt.yaxis.label.set_color(curve_dlt.get_color())
        ax_cloud.yaxis.label.set_color(curve_cloud.get_color())
        ax_rain.yaxis.label.set_color(curve_rain.get_color())
        ax_vcloud.yaxis.label.set_color(curve_vcloud.get_color())
        ax_vrain.yaxis.label.set_color(curve_vrain.get_color())
        ax_wwind.yaxis.label.set_color(curve_wwind.get_color())
        ax_tem.yaxis.label.set_color(curve_tem.get_color())
    
        # 边框
        bwith = 2 #边框宽度设置为1x.spines['bottom'].set_linewidth(bwith)
        ax_hei.spines['left'].set_linewidth(bwith)
        #ax_hei.spines['top'].set_linewidth(bwith)
        #ax_hei.spines['right'].set_linewidth(bwith)
        ax_hei.spines['bottom'].set_linewidth(bwith)
        ax_dia.spines['left'].set_linewidth(bwith)
        ax_dia.spines['right'].set_linewidth(bwith)
        ax_dlt.spines['right'].set_linewidth(bwith)
        ax_cloud.spines['right'].set_linewidth(bwith)
        ax_rain.spines['right'].set_linewidth(bwith)
        ax_vcloud.spines['right'].set_linewidth(bwith)
        ax_vrain.spines['right'].set_linewidth(bwith)
        ax_wwind.spines['right'].set_linewidth(bwith)
        ax_tem.spines['right'].set_linewidth(bwith)
    
        ax_dia.spines['left'].set_color("#FF7F24")
        ax_dia.spines['right'].set_color('#FF7F24')
        ax_dlt.spines['right'].set_color('red')
        ax_cloud.spines['right'].set_color('palegreen')
        ax_rain.spines['right'].set_color('lightskyblue')
        ax_vcloud.spines['right'].set_color('green')
        ax_vrain.spines['right'].set_color('darkblue')
        ax_wwind.spines['right'].set_color('#CD1076')
        ax_tem.spines['right'].set_color('#8e7618')

        # 刻度长度和颜色 刻度标签字体大小
        tkw = dict(size=5, width=1.5, direction='in', labelsize=15)
        ax_hei.tick_params(axis='x', **tkw)
        ax_hei.tick_params(axis='y', colors=curve_hei.get_color(), **tkw)
        ax_dia.tick_params(axis='y', colors=curve_dia.get_color(), **tkw)
        ax_dlt.tick_params(axis='y', colors=curve_dlt.get_color(), **tkw)
        ax_cloud.tick_params(axis='y', colors=curve_cloud.get_color(), **tkw)
        ax_rain.tick_params(axis='y', colors=curve_rain.get_color(), **tkw)
        ax_vcloud.tick_params(axis='y', colors=curve_vcloud.get_color(), **tkw)
        ax_vrain.tick_params(axis='y', colors=curve_vrain.get_color(), **tkw)
        ax_wwind.tick_params(axis='y', colors=curve_wwind.get_color(), **tkw)
        ax_tem.tick_params(axis='y', colors=curve_tem.get_color(), **tkw)

        # 设置刻度标签和坐标轴之间距离
        for tick in ax_hei.xaxis.get_major_ticks()[:]:
            tick.set_pad(12)
        for tick in ax_hei.yaxis.get_major_ticks()[:]:
            tick.set_pad(12)

        lines = [
            curve_hei, curve_wwind,curve_dia, curve_dlt, curve_cloud, curve_rain, curve_vcloud,
            curve_vrain, curve_tem
        ]

        # 设置图片标签
        ax_hei.legend(lines, [l.get_label() for l in lines],
                      loc='upper center',
                      fontsize=15,
                      frameon=True,
                      fancybox=True,
                      framealpha=0.8,
                      borderpad=0.3,
                      labelspacing=1.0,
                      ncol=5,
                      markerfirst=False,
                      markerscale=1,
                      numpoints=1,
                      bbox_to_anchor=(1.10, 1.47),
                      handlelength=3.5)

        plt.savefig('./para-time-K='+str(k)+'_'+str(pp)+'.png')
        plt.close()
