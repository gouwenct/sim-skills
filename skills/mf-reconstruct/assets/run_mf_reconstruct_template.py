

import Run_Recon_3DSIM as _RUN1
import argparse
import os
import numpy as np
import Utils as _U
from pathlib import Path




#光路
def opt_opical_parameter1():
    #光学参数-488-525 油镜
    opt_op = argparse.Namespace()
    
    # "基本信息"
    # opt_op.n = 1.27
    # opt_op.theta =  (np.pi / 3)   

    # opt_op.NAzm = opt_op.n * np.sin(opt_op.theta)  # 照明光路IR
    # opt_op.NAjs = 1.15#1.33#1.27 #1.33  # 接收光路OTF 物镜
    # opt_op.na = opt_op.NAjs
    # opt_op.namin = min(opt_op.NAjs, opt_op.n)  # 接收光路OTF，取小

    # opt_op.Wzm = 488  # Ir
    # opt_op.Wjs = 525  # OTF

    # opt_op.na_mask = 1.15#1.33#1.27  # 1.27    #让mask尽量大一点
    # opt_op.n_mask = 1.27
    
    
    # "基本信息"
    # opt_op.n = 1.33
    # opt_op.theta = 60 * (np.pi / 180)   

    # opt_op.NAzm = opt_op.n * np.sin(opt_op.theta)  # 照明光路IR
    # opt_op.NAjs = 1.27  # 接收光路OTF 物镜
    # opt_op.na = opt_op.NAjs
    # opt_op.namin = min(opt_op.NAjs, opt_op.n)  # 接收光路OTF，取小

    # opt_op.Wzm = 488  # Ir
    # opt_op.Wjs = 525  # OTF

    # opt_op.na_mask = 1.27  # 1.27    #让mask尽量大一点
    # opt_op.n_mask = 1.33
    #######################################################
    
    
    "基本信息"  # mirror 0-run5-488-20260724-3
    opt_op.n = 1.33
    opt_op.theta = 71.33 * (np.pi / 180)   

    opt_op.NAzm = opt_op.n * np.sin(opt_op.theta)  # 照明光路IR
    opt_op.NAjs = 1.33#1.27 #1.33  # 接收光路OTF 物镜
    opt_op.na = opt_op.NAjs
    opt_op.namin = min(opt_op.NAjs, opt_op.n)  # 接收光路OTF，取小

    opt_op.Wzm = 488  # Ir
    opt_op.Wjs = 525  # OTF

    opt_op.na_mask = 1.33#1.27  # mirror 3D recon
    opt_op.n_mask = 1.33
    
    #################################203 HIS-SIM############
    # opt_op.n = 1.5
    # opt_op.theta = 44 * (np.pi / 180)   

    # opt_op.NAzm = opt_op.n * np.sin(opt_op.theta)  # 照明光路IR
    # opt_op.NAjs = 1.5#1.27 #1.33  # 接收光路OTF 物镜
    # opt_op.na = opt_op.NAjs
    # opt_op.namin = min(opt_op.NAjs, opt_op.n)  # 接收光路OTF，取小

    # opt_op.Wzm = 488  # Ir
    # opt_op.Wjs = 525  # OTF

    # opt_op.na_mask = 1.5#1.27  # 1.27    #让mask尽量大一点
    # opt_op.n_mask = 1.5
    # opt_op.fc= 145
    
    # n*sinθ   
    opt_op.waterN = 1.33#1.27#1.33 #
    opt_op.lamdex = 488  # 488#561#488        #lamdex 入射光波长  560
    opt_op.lamdem = 525  # 525#607#525       #lamdem 出射光波长 与OTF相关  600
    
    ######################################################
    
    return opt_op

def opt_RUN1_parameter1(opt_1):
    #第一套运行参数 MP-55-4B模式
    # MP (7,7,3)  slice(7,5,3)  tr(5,5,3) 2d(3,3,3)
    
    opt_1.flag_N = 5  # imaging setting parameter
    opt_1.Nshifts = 5  # phase shifts for each stripe
    opt_1.Nangles = 3  # number of orientations of stripes
    opt_1.nc = opt_1.Nangles * opt_1.Nshifts
    # opt.pz_manual = 0  # 轴向位移量

    #pixelsize — mirror 0-run5-488-20260724-3
    opt_1.dx = 60
    opt_1.dz = 193.8#96.9*2

    opt_1.Zpadding = 0  # 3D-SIM 中不需要轴向padding？
    
    #Apoz（mear=2 时 Run_Recon 会再改 Apoz/fcxs）
    opt_1.Apoz = 900   
    opt_1.fcxs = 1.1
    opt_1.plongxs = 1.1
    # opt_1.Apo = imread(r'F:\\4TB图片-续\\2024\\一月\\1.28-仿真\\17、D3-v=0-轴向双层移动\\正向\\【make_GT2】Apo-GT-200.tif') #仿真图片用~~~~~

    # opt_1.Apo = None
    ##############################################################
    # opt_1.W1 = [0.9]  #第一次   
    # opt_1.W = [5e-4]  #第二次 #4B
    opt_1.W = [1e-4]  # mirror 3D recon
    return opt_1


def define_run_parameter():
    run_parameter = dict(
        re_mode=0,                             # ~~~~~~~~~~~【mode】【0:Wiener   1:HIFI   3:opensim 】~~~~~~~~~~~
        # OTF相关
        createOTF=0,                           # ~~~~~~~~~~~~~~【OTF】读取(0)还是产生OTF(1)、插值得到OTF】~~~~~~~~~~~~~~~~~
        # 参数相关!!!
        loadflag=1,                             # ~~~~~~~~~~~~~【参数】是否读取参数： 0--各自算参数，1--读取同一个参数p cfai，
                                                                                    #2--同P 自己算cfai+保存 ， 3--自己算P，同一个cfai+不保存

        just_cfai=0,                            # 正常运行为0 ，只参数估计为1~~~~~~~~~~~~~~~~只~~~~~~~~~~~~~~~~~~~~~
        
        PCA=0,  
        fanwei=30,
        cfaii=0,                                # 指定c值 越大，高频被抑制； C越小 高频被放大
        pzzz=0,                                 #3D-SIM这里为0
        pzzz_range=0,                           #3D-SIM这里为0
        
        # notch相关参数
        notch=[0],                            # 0 不使用12阶notch，1 使用12阶notch
        
        # 遍历notch （嵌套的循环）#################################################################
        # 0阶notch参数                                                                          ##
        notch0ths=[0.95],     # 0.998 notch0th， 0阶notch抑制度（0-1）
        # notch0ths=[0.5],
        attFWHM0s=[1.5],       # attFWHM0， 0阶notch范围
        # 2阶notch参数/FZ                                                                       ##
        attStrFZs=[0.8],    # attStrFZ，2阶notch抑制度
        # attStrFZs=[0],    # attStrFZ，2阶notch抑制度
        attFWHMFZs=[1.5],      # attFWHMFZ,2阶notch范围
        # 1阶notch 参数                                                                         ##
        attStrs=[0.8],       # 1阶notch抑制度
        # attStrs=[0],       # 1阶notch抑制度
        attFWHMs=[1.5],      # 1阶notch范围
        # 1阶旁边notch参数                                                                      ##
        notch1th_pangs=[1],  # 1阶旁notch抑制度  notch1th_pang,对sk1阶旁边亮点的抑制
        #########################################################################################
        
        # 额外参数
        bleach = 0 ,            #光漂白校正
        directions=[[0, 1, 2]], #重建方向
        orderForRe=[6],         #重建阶数     #0 0th  ；1 ±1th ； 2 ±2th 
                                             #3 0±1   ；4 0±2  ；5 ±1±2   6 all
        Method_c=[5],           # 常规方法1----5（默认）    方法2-----4
        Msum_maskMove=1,                   #Msum mask是生成式1（默认） 还是移动式2
        decon_meth="4B",       #XYD 先移动的频域重构法    XC 先除的空域重构法   （两者差不多）~~~~~~~~~
        
        B4_double = 0,         #分方向滤波否0   1分方向
        Dr_sub = '',
        
        
        #下面参数不太重要
        ##cfai 体估计还是面估计
        piancha=0,                              # 偏差，无偏差为0
        flagV=1,                                # 默认1为体估计
        # Apo_Z=4,                                # 控制轴向分辨率，值越小，轴向衰减越平缓
        
        pg=0,                                   # 估计时的参数大小，为0时 后面用opt.pxy为之赋值   0则后面自己计算
        PSF=False,
        
        #   #HIFI模式下参数
        W222=0.1,                              # HIFI的
        RL=0,                                  # 是否RL  只在HIFI mode下进行
        otfHIFI=[1],                           # 0:自己产生  1：读取   2 插值  只在HIFI mode下进行
    )
        
        
    return run_parameter


def Get_path(run_parameter):
 #OTF文件夹
    OTFone = os.environ["MF_SIM_OTF_ROOT"]
    
    savepath0 = os.environ["MF_SIM_DATA_ROOT"]

    data_name = str(Path(str(savepath0)).name)  #提取当日数据名称
    run_parameter['data_name'] = data_name   
    run_parameter['OTFname'] = OTFone 
    return savepath0,OTFone,run_parameter



if __name__ == '__main__':
    
    opt = opt_opical_parameter1() #488油镜参数
    opt = opt_RUN1_parameter1(opt)  #部分运行参数
    run_parameter = define_run_parameter() #部分呢运行参数
    savepath0,OTFone,run_parameter = Get_path(run_parameter) #path文件-- 修改
    
    FANWEI = [19]   #todo  # 19、mp_5610ms (iterdir order)
    
    run_parameter['movex']=0
    run_parameter['movey']=0
    run_parameter['movex_long'] = 462
    run_parameter['movey_long'] = 462
    run_parameter['mear'] = 2
    
    "一、no padding求参数"
    run_parameter['PSF'] = False
    run_parameter['Dr_sub ']= ""
    
    run_parameter['loadflag'] = 0
    opt.Zpadding = 0
    run_parameter['pzzzs'] = [3]
    run_parameter['just_cfai'] = 1

    
    # _RUN1.main(opt,run_parameter,savepath0,OTFone,FANWEI) #求出参数即退出
    ###################################################################################
    
    opt = opt_RUN1_parameter1(opt)  #部分运行参数
    
    "二、MF slice 仅估参（对齐 3D recon 20260724-3）"
    run_parameter['PSF'] = False

    run_parameter['Dr_sub'] = "01-488_padding_slice"

    run_parameter['notch'] = [0]
    run_parameter['notch0ths']=[0.6]
    
    #2D OR MF 切换
    opt.flag_N = 5  # imaging setting parameter
    opt.Nshifts = 5  # phase shifts for each stripe
    opt.Nangles = 3  # number of orientations of stripes
    opt.nc = opt.Nangles * opt.Nshifts
    
    opt.dx = 54  # mirror 3D recon
    opt.W = [1e-4]
    run_parameter['loadflag'] = 0
    opt.Zpadding = 0
    run_parameter['just_cfai'] = 1  # 1 只估参（试跑）
    run_parameter['decon_meth'] = '4B'
    run_parameter['fanwei'] = 40  # mirror 3D fanwei=40
    
    _RUN1.main(opt,run_parameter,savepath0,OTFone,FANWEI)    #仅估参
    
