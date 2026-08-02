"""
GPU parameter-estimation entry (2026-07-25).

HARD BOUNDARY: does not edit Main_recon_MP.py / Run_recon_MP.py / Para_Estimate.py.
Injects Para_Estimate_GPU into Main_recon_MP._PE before reconstruction runs.

Defaults:
  run_parameter['parameter_device'] = 'auto'   # auto|cuda|cpu
Validation isolation:
  Run from an isolated working directory and set MP_SIM_RECON_DATA_ROOT to an
  isolated input root. Utils writes dataOut below the current working directory.
Edit FANWEI / paths for next datasets as usual.
"""
# --- inject GPU parameter-estimation module (must happen before Run_recon_MP use) ---
import Main_recon_MP as _MR
import Para_Estimate_GPU as _PEG
_MR._PE = _PEG

import Run_recon_MP as _RUN1
_RUN1._MMP._PE = _PEG  # ensure Run path sees the same module

import argparse
import os
import numpy as np
import Utils as _U
from pathlib import Path
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def _skip_irtest_for_parameter_validation():
    """Skip reconstruction-only shift matrices after parameters are saved."""
    if os.environ.get('MP_SIM_SKIP_IRTEST', '0').strip() != '1':
        return

    def parameter_only_irtest(*args, **kwargs):
        print('[0-run5-488-20260725-GPU] skipping IrTEST for parameter-only validation')
        return None, None

    _MR._WR.IrTEST = parameter_only_irtest



def save_parameter(opt,run_parameter):
    #保存运行参数（无需for循环）
    from datetime import datetime
    now = datetime.now().strftime("%H-%M-%S")
    
    opt_dic = vars(opt)
    merged_dic = opt_dic|run_parameter
    # print(merged_dic)
    txt_path = _U.save_path(f'opt_run_parameter-{now}.txt',run_parameter)
    _U.save_txt(txt_path,merged_dic)
    
    return 0

#光路
def opt_opical_parameter1():
    #488-525 oil 
    opt_op = argparse.Namespace()
    
    opt_op.n = 1.33                                   
    opt_op.theta = 71.33 * (np.pi / 180)   

    opt_op.NAzm = opt_op.n * np.sin(opt_op.theta)     # NA of Illumination optical path 
    opt_op.NAjs = 1.33                                # NA of Receiving optical path
    opt_op.na = opt_op.NAjs                           
    opt_op.namin = min(opt_op.NAjs, opt_op.n)         # min NA

    opt_op.Wzm = 488  # Ir
    opt_op.Wjs = 525  # OTF

    opt_op.na_mask = 1.33 # 1.27                      # OTF mask parameter
    opt_op.n_mask = 1.33

    # n*sinθ   
    opt_op.waterN = 1.33 #
    opt_op.lamdex = 488  # 488#561#488        #lamdex 入射光波长  560
    opt_op.lamdem = 525  # 525#607#525       #lamdem 出射光波长 与OTF相关  600

    return opt_op


def opt_RUN1_parameter1(opt_1):
    #第一套运行参数 MP-55-4B模式
    #base MP 5+5
    opt_1.flag_N  = "MP"    # imaging setting parameter MP=7  3D=5  2D=3
    opt_1.Nshifts = 5       # phase shifts for each stripe (raw data) 
    opt_1.Norders = 7       # after separate total orders  0，±2，(±1,±1) 3D-MP=7  3D=5  2D=3
    opt_1.Nangles = 3       # number of orientations of stripes
    opt_1.nc = opt_1.Nangles*opt_1.Norders  #
    opt_1.nz = 8#16
    opt_1.nx = 462                        
    opt_1.ny = 462
    opt_1.sequence = 55                   # 15+15 mode
    opt_1.faiXY = 2/5                     #XY  Lateral phase spacing
    opt_1.faiZ  = 0.52                    #Z   Axial phase spacing 
    opt_1.order7 = [0,1,2,3,4, 0,2]       #52(3x7) [2/5 π] mode order
    opt_1.order5 = [0,1,2,3,4]            #c15+15 [2/5 π] mode order 
    
    #pixelsize
    opt_1.dx = 60#45#60#60#72#60# 40            # Lateral piexlsize 
    opt_1.dz = 193.8                      # Axial piexlsize
    
    #Apoz 参数
    opt_1.Apoz = 900                      # Axial range of the apodization function 
    opt_1.fcxs = 1.1#1.4                  # Axial range of the apodization function 1
    opt_1.plongxs = 1.1#1.5               # Axial range of the apodization function 2
    
    opt_1.W = [1e-3]
    
    return opt_1


def define_run_parameter():
    run_parameter = dict(
        # re_mode=0,                              # ~~~~~~~~~~~【mode】【0:Wiener   1:HIFI   3:opensim 】~~~~~~~~~~~
        # OTF相关
        createOTF=0,      #<--！！！-           # 0:Load OTF  1: generate OTF
        # 参数相关!!!
        loadflag=0,      #<--！！！--           # 0:Recalculate parameters and save file; 
                                                # 1:load all parameters; 
                                                # 2:only load wave vector P, calculate modulation depth and phase, and save file; 
                                                # 3:only load wave vector P, calculate modulation depth and phase without saving the file
                                                # 4: [0 or 3]
                                                # 5: [0 or 1]
        just_cfai=0,                            # 1: only estimate parameters    0:  Reconstructed image 
                                                                                       
        # PCA=0,                                   
        fanwei=40,                              # Range for calculating the wave vector (20 define)
        
        pzzzs=[0],                              # Theoretical pz length
        # pzzzs=[4.0],
        pzzz_range=0,                           #3D-SIM这里为0
        
        # notch   #################################################################
        notch=[0],             # 0: Not using first and second-order notch; 1: Not using first and second-order notch
                               # Note, this parameter does not control the 0th-order notch
        # 0阶notch参数                                                                          ##
        notch0ths=[0.95],      # 0th-order, Notcch suppression level（0-1）
        attFWHM0s=[1.5],       # 0th-order, Notch range
        # 2阶notch参数/FZ                                                                       ##
        attStrFZs=[0.8],       # 2th-order, Notcch suppression level（0-1）
        attFWHMFZs=[1.5],      # 2th-order, Notch range
        # 1阶notch 参数                                                                         ##
        attStrs=[0.8],         # 1st-order, Notcch suppression level（0-1）
        attFWHMs=[1.5],        # 1st-order,Notch range
        # 1阶旁边notch参数                                                                      ##
        notch1st_pangs=[1],    # First-order side lobes (1 define)
        #########################################################################################
        # 额外参数c
        
        directions=[[0,1,2]],  #Direction of reconstruction
        orderForRe=[6],        #Direction of reconstruction     #0 0th  ；1 ±1st ； 2 ±2th  #3 0±1   ；4 0±2  ；5 ±1±2   6 all (define)
        Method_c=[5],          #Method for eliminating modulation depth (5: define) 
        Msum_maskMove=1,       #Msum mask is 1 for generating (default) or 2 for shifting
        # decon_meth="4B",       #XYD 先移动的频域重构法    XC 先除的空域重构法   （两者差不多）   #~~~~~~~~~~
        #                         #4B  4B那样XYD然后整体相除，用了0阶notch
        
        # B4_double = 0,          #分方向滤波  0:否  1:是c
        joint_deconvolution = 1,    #Joint deconvolution ,0 no,  1 yes  define
        joints = 0,#1               #0 （0,1,2）  or 1 （1,2）   
        remove2 = 1,                 #remove 2th phase
        Dr_sub = "01",           # cReconstructed sub-filec

        pad_xishu = 1,            #padding Coefficient
        #file
        save= 0  ,
        fast = 0  ,              # Reconstructed sub-filec
        chazhi = 0    ,          # Whether to perform automatic axial interpolation


        #下面参数不太重要

        ##cfai 体估计还是面估计
        piancha=0,                              # Amount of movement during surface parameter estimation
        flagV=1,                                # Whether to perform automatic axial interpolation: 1 (definecc)
        
        pg=0,                                   # 估计时的参数大小，为0时 后面用opt.pxy为之赋值   0则后面自己计算
        # PSF=False,                              # 加载的PSF还是OTF（仅多层有效果）

        # # #   #HIFI模式下参数
        # W222=0.1,                               # HIFI的
        # RL=0,                                   # 是否RL  只在HIFI mode下进行
        # otfHIFI=[1],                            # 0:自己产生  1：读取   2 插值  只在HIFI mode下进行

        # GPU parameter-estimation device policy (only consumed by Para_Estimate_GPU)
        # auto: CUDA if free VRAM >= 2.5 * estimated peak; else CPU
        # cuda: require CUDA, fail clearly (no silent fallback)
        # cpu: force CPU path
        parameter_device='auto',
        # essential: final parameters/logs plus compact lihe summary
        # cfai_png: also save modulation-depth/phase PNGs
        # debug: retain all legacy diagnostic images
        parameter_output_level=os.environ.get(
            'MP_SIM_PARAMETER_OUTPUT', 'essential'
        ).strip().lower(),
           
    )
    if run_parameter['parameter_output_level'] not in {
        'essential', 'cfai_png', 'debug'
    }:
        raise ValueError(
            "MP_SIM_PARAMETER_OUTPUT must be 'essential', 'cfai_png', or 'debug', got "
            f"{run_parameter['parameter_output_level']!r}"
        )
    return run_parameter
    

def Get_path(run_parameter):
    #todo
    
    ' Water OTF'
    'Oil OTF'
    OTFone = os.environ["MP_SIM_OTF_ROOT"]

    'sample path'
    # savepath0 = r'D:\GW\code\MP-SIM\MP-real\MP-realV2.7\dataOut\F2-20240509beads'
    # savepath0 = r'..\..\..\process预处理-MP2DMF\\F2-20240509-beads'
    savepath0 = os.environ["MP_SIM_RECON_DATA_ROOT"]
    
    data_name = str(Path(str(savepath0)).name)  #提取当日数据名称
    run_parameter['data_name'] = data_name   
    run_parameter['OTFname'] = OTFone 
    return savepath0,OTFone,run_parameter

if __name__ == '__main__':

    # Initialize optical path parameters, run parameters
    opt = opt_opical_parameter1()   #488 oil 
    opt = opt_RUN1_parameter1(opt)  
    run_parameter = define_run_parameter() # running parameter
    savepath0,OTFone,run_parameter = Get_path(run_parameter) #set path

    fanwei_override = os.environ.get('MP_SIM_FANWEI', '').strip()
    FANWEI = (
        [int(value.strip()) for value in fanwei_override.split(',') if value.strip()]
        if fanwei_override else [18]
    )    # file range; validation root with one sample uses MP_SIM_FANWEI=1
    run_parameter['movex']=0
    run_parameter['movey']=0
    run_parameter['movex_long'] = 462
    run_parameter['movey_long'] = 462
    run_parameter['use_mask']=1
    run_parameter['mear'] = 2#1
     
    
    #先运行一遍求参数
    
    opt.nz = 8
    opt.nx = 462
    opt.ny = 462
    opt.W = [1e-4]
    pz8 = 3
    if run_parameter['parameter_output_level'] == 'debug':
        run_parameter['DEBUG'] = dict(save=1, loa=1,sep=0,est=0,mov=0,wie=1)
    else:
        run_parameter['DEBUG'] = dict(save=0, loa=0,sep=0,est=0,mov=0,wie=0)
    
    
    '一、求参数 Data requiring padding, calculate parameters without padding first'
    run_parameter['Dr_sub'] = "01-488-AVG_padding" #"RU"MM" LU
    run_parameter['W_order'] = os.environ.get(
        'MP_SIM_W_ORDER', '488-6.5-P-GPU'
    )

    run_parameter['loadflag'] = 0
    opt.Zpadding = 0#4 
    run_parameter['pzzz'] = pz8/8*(opt.nz+2*opt.Zpadding) #求参数时 不padding
    run_parameter['just_cfai'] = 0
    # _RUN1.main(opt,run_parameter,savepath0,OTFone,FANWEI)    #padding重建

    
    "二、仅估参（单数据 01-488_padding，just_cfai=1）"
    opt.Zpadding = 0#4
    run_parameter['Dr_sub'] = "01-488_padding" #"RU"MM" LU
    
    run_parameter['notch'] = [0]           
    run_parameter['notch0ths']=[0.6]
    run_parameter['loadflag'] = 0
    
    run_parameter['pzzz'] = pz8/8*(opt.nz+2*opt.Zpadding) #8层-1.33，1.27 油镜的 pz=3
    run_parameter['just_cfai'] = 1  # 1 只估参
    
    run_parameter['save'] = 0  #tag
    run_parameter['fast'] = 0  #tag
    run_parameter['dr_timepoints'] = [3,4,39,40]  #只估第一组(0003,0004)+最后一组(0039,0040)
    run_parameter['chazhi'] = 0 #tag   
    
    run_parameter['XZ_save'] = 0 #tag 
    run_parameter['YZ_save'] = 0 #tag   

    # Default GPU auto; set MP_SIM_PARAMETER_DEVICE=cpu|cuda for explicit runs.
    run_parameter['parameter_device'] = os.environ.get(
        'MP_SIM_PARAMETER_DEVICE', 'auto'
    ).strip().lower()
    
    # OTFone = r'D:\GW\code\OTF4.1\MP-16'
    print('[0-run5-488-20260725-GPU] using Para_Estimate_GPU via Main_recon_MP._PE')
    print(f"[0-run5-488-20260725-GPU] parameter_device={run_parameter['parameter_device']}")
    print(f"[0-run5-488-20260725-GPU] FANWEI={FANWEI}, W_order={run_parameter['W_order']}")
    print(f"[0-run5-488-20260725-GPU] data_name={run_parameter['data_name']}")
    print(
        '[0-run5-488-20260725-GPU] parameter_output_level='
        f"{run_parameter['parameter_output_level']}"
    )
    
    opt.parameter_output_level = run_parameter['parameter_output_level']
    _skip_irtest_for_parameter_validation()
    _RUN1.main(opt,run_parameter,savepath0,OTFone,FANWEI)    #仅估参
