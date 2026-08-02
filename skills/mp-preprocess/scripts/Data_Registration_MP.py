import importlib

# %matplotlib inline
import os
from pathlib import Path
import numpy as np
import torch
from tifffile import imread

import CalRegistration as _C
import Cam_process as ca
import GetRawdata as _G
import main as _m
import Utils as _U
import BF as _BF
import AVG_Data as _AD


def get_sift_projection_method(run_parameter):
    method = run_parameter.get('sift_projection_method', 'tenengrad')
    if method not in ('tenengrad', 'mean'):
        raise ValueError(
            "sift_projection_method must be 'tenengrad' or 'mean', "
            f"got {method!r}"
        )
    return method


def peizhun_dir_match(dir_name: str, peizhun_file: str) -> bool:
    """
    Match peizhun folder by name without 'peizhun' swallowing 'peizhun2'/'peizhun3'.

    Uses: peizhun_file appears in dir_name, and is not followed by a digit.
    Examples: peizhun_file='peizhun' → 99、peizhun yes; 99、peizhun2 no.
    """
    import re

    if not peizhun_file:
        return False
    return re.search(re.escape(str(peizhun_file)) + r'(?!\d)', str(dir_name)) is not None


def get_sift_cache_priority(path, peizhun_file, method):
    path = Path(path)
    if path.suffix.lower() != '.npy' or str(peizhun_file) not in path.name:
        return None
    if path.stem.endswith(f'_{method}'):
        return 0
    if method == 'mean' and not path.stem.endswith('_tenengrad'):
        return 1
    return None

def mask_file(para):
    FileName = _U.save_path("", para)

    Make_Dir(FileName, "Dr")
    # Make_Dir(FileName, "Dr-s")
    Make_Dir(FileName, "OTF")
    # Make_Dir(FileName, "OTF-s")
    parameter_next = Make_Dir(FileName, "parameter")
    Make_Dir(parameter_next, "para")
    Make_Dir(parameter_next, "488")
    Make_Dir(parameter_next, "561")
    Make_Dir(FileName, "results")
    
    return 0

def Make_Dir(FileName, file_dir):
    File1 = FileName / file_dir
    if not File1.exists():
        File1.mkdir()
        print(f" maker new folder :{File1}")
    return File1

def mask2(Dr, kuodaxy=1, sigma=0.25, padsize=0):
    """边缘矩形mask"""
    nz, nx, ny = Dr.shape
    ny = nx * kuodaxy
    nx = ny * kuodaxy
    nz = nz

    x = torch.arange(1, nx + 1).reshape(1, -1)
    y = torch.arange(1, ny + 1).reshape(-1, 1)

    sigma = sigma
    mask2d = (torch.sigmoid(sigma * (x)) - torch.sigmoid(sigma * (x - ny - 1))).repeat(
        nx, 1
    ) * (torch.sigmoid(sigma * (y)) - torch.sigmoid(sigma * (y - nx - 1))).repeat(1, ny)

    return mask2d.numpy()



def mask2_self(nx,ny,kuodaxy=1,sigma = 0.25,padsize=0):
    """边缘矩形mask"""
    # ny = nx
    # nx = ny
    # nz = nz

    x = torch.arange(1,nx+1).reshape(-1,1)
    y = torch.arange(1,ny+1).reshape(1,-1)

    sigma=sigma;
    mask2d = ((torch.sigmoid(sigma*(x))-torch.sigmoid(sigma*(x-nx-1)))
                * (torch.sigmoid(sigma*(y))-torch.sigmoid(sigma*(y-ny-1))))

    return mask2d.numpy();

def trans_PAZ_TO_PZA(data, Nshift=7, nc=21, nz=8, nx=430, ny=430):
    # 将一种拍图模式PAZ变为另一种拍图模式PZA
    # transfer (phase,direction,nz)-> (phase,nz,direction)
    # （7,3，nz）  ->   (7,nz,3)    取出 1-7、
    # nc = 21
    # nz = 8
    # nx = 430
    # ny = 430
    data_PZA = []
    data = data.reshape(-1, Nshift, nx, ny)  # 每7的分为一组，一共24组
    group, _, _, _ = data.shape
    for i in range(3):
        data_PZA.append(data[i:group:3, :, :])  # 同时取出了nz那么多帧
    data_PZA = np.array(data_PZA)
    data_PZA = _U.image_reshape(data_PZA, dim=4).reshape(nc, nz, nx, ny)
    return data_PZA

def Transfer2NPD(Dr_read):
    #transfer (phase,direction,nz)-> (nz,phase,direction)
    nc,nz,nx,ny = Dr_read.shape
    Dr_read = Dr_read.reshape(nc*nz,nx,ny)
    Dr_zpd = np.zeros_like(Dr_read).reshape(nc,nz,nx,ny) #4D!!!
    for i in range(nc):
        Dr_zpd[i] = Dr_read[i:(nc*nz):nc,:,:]  #最终Dr 变量为我们所要的raw data
    Dr_zpd = _U.image_reshape(Dr_zpd,dim=4).reshape(nc,nz,nx,ny)
    return Dr_zpd


def getWF(raw_data,para):
    """
    Dr：将3D-->4维array （nc,nz,nx,ny）
    思路：将raw data 每nc帧累加为一帧
    """
    
    nc =para['zhenshu']
    nth = para['nth_total']
    #3D变4D
    if len(raw_data.shape)==3:
        ncnz, nx, ny = raw_data.shape
        raw_data = raw_data.reshape(nc,ncnz//nc,nx,ny)
        # print(ncnz, nx, ny)
    nc,nz, nx, ny = raw_data.shape
    raw_data = Transfer2NPD(raw_data)   #重建raw data 的格式
    
    #4D变WF
    WF = np.mean(raw_data,axis=0)   #将第一维度累加
    
    '简单相加 与 边缘衰减'
    # mask = mask2(WF, sigma=0.25)
    mask = mask2_self(nx,ny,sigma=0.25)
    WF = WF * mask
    # WF = _U.normalize(WF)
    
    if (nth%10==0 or nth==1) == False : # 
        return WF,WF,WF
    
    
    '横向、轴向扩大(频域补零)'
    WF_padding = _U.pad_or_cut(WF, x=nx,y =ny,z=nz+2 )  #空域先padding 0  避免首尾重复
    
    nx_kuoda = 2*nx
    ny_kuoda = 2*ny
    nz_kuoda = 2*(nz+2)
    
    WF2_k = _U.pad_or_cut(_U.fftn(WF_padding), x=nx_kuoda, y=ny_kuoda, z=nz_kuoda)  #直接频域padding 
    WF2_r = _U.ifftn(WF2_k).real
    WF2_r[WF2_r < 0] = 0

    #20->16，归一化
    WF2_rcai = WF2_r[2:-2]
    WF2_rcai = _U.normalize(WF2_rcai)

    '空域补零'
    ###################轴向不变的情况-SLICE
    # WF3_k = _U.pad_or_cut(_U.fftn(WF), x=nx_kuoda, y=ny_kuoda, z=nz)
    # WF3_r = _U.ifftn(WF3_k).real
    # WF3_r[WF3_r < 0] = 0

    # #归一化
    # WF3_rcai = WF3_r
    # WF3_rcai = _U.normalize(WF3_rcai)
    
    WF3 = Interpolate(WF,nx=nx_kuoda,ny=ny_kuoda,nz=2*nz)
    WF3_n = _U.normalize(WF3)
    
    # '横向、轴向均变为30nm层间距'   #先不弄，
    # #8层 层间距为195 -> 空域补零 padding 10层对应195 ------->  频域补零  10->65层 对应30nm
    # dz = 195   # MP 采样层间距
    # dz_T=30    # 目标层间距
    # dz_ratio = dz/dz_T
    # WF4_k = _U.pad_or_cut(_U.fftn(WF_padding), x=nx_kuoda, y=ny_kuoda, z = int(dz_ratio* (nz+2))-1 )
    # WF4_r = _U.ifftn(WF4_k).real
    # WF4_r[WF4_r < 0] = 0
    
    # #归一化
    # WF4_rcai = WF4_r[6:-7]
    # WF4_rcai = _U.normalize(WF4_rcai)
    
    return WF,WF2_rcai,WF3_n


def Interpolate(image,nx,ny,nz,strt = 'trilinear'):
    """
    image为4D,3D都可以，但必须是numpy
     mode (str): algorithm used for upsampling:
        ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
    """
    
    before_dim = len(image.shape)

    image = _U.image_reshape(image,dim=5)   #1,cc,nz,nx,ny)

    #转变为torch
    image = _U.ToTensor(image) 

    if torch.is_complex(image)!=True : #非复数
        image_inter = torch.nn.functional.interpolate((image),size=[nz,nx,ny],mode=strt,align_corners=False)
    else: #复数拆开插值，再组合
        image_real = image.real
        image_imag = image.imag
        image_real_inter = torch.nn.functional.interpolate((image_real),size=[nz,nx,ny],mode=strt,align_corners=False)
        image_imag_inter = torch.nn.functional.interpolate((image_imag),size=[nz,nx,ny],mode=strt,align_corners=False)
        image_inter = image_real_inter+1j*image_imag_inter
        
    return _U.image_reshape(_U.ToNumpy(image_inter),dim=before_dim)

def Get_WF_PAZ(Dr_stack_cai,para):
    # 【WF】最后一个时间点得到WF
    WF,WF2_rcai,WF2_k,WF3 = getWF(Dr_stack_cai,para)


    _U.save_tif_v21(f'WF-R-{WF.shape}.tif',WF,para)
    _U.save_tif_v21(f'WF-R-{WF2_rcai.shape}.tif',WF2_rcai,para)
    _U.save_tif_v21(f'WF-K-{WF2_rcai.shape}.tif',abs(WF2_k),para)
    _U.save_tif_v21(f'WF3-R-{WF3.shape}.tif',WF3,para)
    
    #[PZA] 最后一个时间点PAZ -> PZA
    dir_name = para['dir_name']
    ncnz,nx,ny = Dr_stack_cai.shape
    # data_PZA = trans_PAZ_TO_PZA(Dr_stack_cai, Nshift=7, nc=21,nz=ncnz//21,nx=nx, ny=ny)
    data_PZA = trans_PAZ_TO_PZA(Dr_stack_cai, Nshift=para['zhenshu']//3, nc=para['zhenshu'],nz=ncnz//para['zhenshu'],nx=nx, ny=ny)
    
    data_PZA = _U.image_reshape(data_PZA,dim=3)
    _U.save_tif_v21(fr'PZA-{dir_name}.tif',data_PZA,para)  


def DeleteError(imagS_M,FANWEI):
    #仅适用于488，561 双通道 30，30 情况
    assert len(imagS_M.shape)==4  #(8,15*point,512,512)  400*15
    nc,nz,nx,ny = imagS_M.shape
    imagS_M_5D = imagS_M.reshape(8,-1,60,nx,ny)  #30,30 至少15+15（通道1），15+15（通道2）模式的一个时间点

    for i in FANWEI:
        imagS_M_5D = np.delete(imagS_M_5D,i-1,axis=1)

    return imagS_M_5D.reshape(8,-1,nx,ny)



def bleach(data4D):
    nc,nz,nx,ny = data4D.shape
    first = np.mean(data4D[:,0,50:-50,50:-50])

    for i in range(0,nz): 
        temp = np.mean(data4D[:,i,50:-50,0:-50])
        rate = first/temp
        data4D[:,i,:,:] = data4D[:,i,:,:]*rate
    return data4D


def Rough_Affine(input_path,para,Registration = 1):
    '''
    计算配准数据的仿射矩阵
    用以指导配准数据
    '''
    loop_dir = Path(str(input_path)).iterdir()
    # pp_name = "regis_"+str(path_peizhun.parent.name[0:5])
    peizhun_file = para['peizhun_file']
    sift_projection_method = (
        get_sift_projection_method(para) if Registration == 3 else None
    )
    # dir_name = str(path_peizhun.name)
    
    ##############################优先找含有npy的文件
    if Registration == 3 and sift_projection_method == 'mean':
        cache_files = []
        for ldr_j in loop_dir:
            dir_name = ldr_j.name 
            suffix = ldr_j.suffix
            print(dir_name)
            print(suffix)
            if ldr_j.is_dir() == 1:                           # 跳过文件夹
                print("【Loop-FILE】跳过0：", dir_name)
                continue
            priority = get_sift_cache_priority(
                ldr_j, peizhun_file, sift_projection_method
            )
            if priority is not None:
                cache_files.append((priority, ldr_j.name, ldr_j))
        if cache_files:
            _, _, cache_path = sorted(cache_files)[0]
            Hs = np.load(cache_path)
            print(f'加载{cache_path.name} ({sift_projection_method})')
            return Hs
    elif Registration == 3:
        print('Tenengrad模式重新计算焦面并执行层间隔检查，不直接复用旧H缓存')
    ###############################【loop 配准各种样本】
    loop_dir = Path(str(input_path)).iterdir()
    i = 0
    flag = 0
    for ldr_i in loop_dir:
        i += 1
        dir_name = ldr_i.name                               # 获取文件夹名
        
        if not peizhun_dir_match(dir_name, peizhun_file):  # 找到配准文件夹（peizhun 不匹配 peizhun2）
            print("【Loop-FILE】跳过1：para['peizhun_file']没有", dir_name)
            continue
        
        elif ldr_i.is_dir() != 1:                           # 跳过文件
            print("【Loop-FILE】跳过2：", dir_name)
            continue
            
        elif peizhun_dir_match(dir_name, peizhun_file):  # 找到配准目录；多个时取遍历到的第一个
            '''
            mode=2 为计算rough仿射矩阵
            Registration = 3 为SIFT算法计算配准矩阵,1为自己的配准
            file_suffix     为文件后缀
            stand = 4       在0-7中选定一个参考区域，默认为4（第五区域）
            ratio_thresh    SIFI中阈值，值越大，匹配特征点越大，但可能错误的越多
            '''
            flag=1
            print("【Loop-FILE】处理：", dir_name)
            if Registration == 1:
                coordinate_set = []
                theta_set = []
                scale_set = []
                coordinate_beads, theta, scale, imag_5 = _m.main(
                    ldr_i, mode=2, Registration=1, file_suffix=dir_name, stand=para["stand"],para=para
                )  
                coordinate_set.append(coordinate_beads)
                theta_set.append(theta)
                scale_set.append(scale)
                return coordinate_beads, theta, scale   #平移、旋转、缩放
            elif Registration == 3:
                print(f'SIFT配准图像方法: {sift_projection_method}')
                Hs = _m.main(
                    ldr_i,pro_meth=sift_projection_method,mode=2,Registration=3,
                    file_suffix=dir_name,
                    stand=para["stand"],
                    ratio_thresh=0.7,  #越大匹配的越多
                    para=para
                )  #
                
                HS_path=_U.save_path(
                    f'Hs_base_{peizhun_file}_{sift_projection_method}.npy', para
                ) #保存HS文件
                np.save(HS_path,Hs)   
                #保存Hs
                np.save(
                    Path(input_path) / f'{dir_name}_{sift_projection_method}.npy',
                    Hs
                )
                return Hs
    if flag == 0:
        raise('没找到MP_peizhun文件夹，请修改-配准数据文件夹-命名方法为MP_peizhun')
   
def Fine_Affine_1_1(input_path,coordinate_beads, theta, scale,para,FANWEI=[1],Registration = 1,func_f=2121):
    '加入文件夹控制程序'
    loop_dir = Path(str(input_path)).iterdir()  # 用于遍历总文件夹
    data_name = Path(str(input_path)).name      # 提取当日数据名称
    time_name = data_name[0:4]  # 提取时间
    para['data_name'] = data_name
    print(f'{data_name=:},{time_name=:}')
    ###############################【loop 配准各种样本】
    i = 0
    for ldr_i in loop_dir:
        i += 1
        dir_name = ldr_i.name                               # 获取文件夹名
        if 'C3DSIM' in dir_name :                       # 跳过3DSIM文件夹
            print("【Loop-FILE】1跳过：", dir_name)
            continue
        elif 'peizhun' in dir_name :                       # 跳过配准文件夹
            print("【Loop-FILE】2跳过：", dir_name)
            continue
        elif 'cam1' in dir_name:
            print("【Loop-FILE】3跳过：", dir_name)
            continue
        elif 'cam2' in dir_name:
            print("【Loop-FILE】4跳过：", dir_name)
            continue
        elif ldr_i.is_dir() != 1:                           # 跳过文件
            print("【Loop-FILE】5跳过：", dir_name)
            continue
        
        if i not in FANWEI:
            print("【Loop-FILE】跳过：", dir_name)
            continue
        
        print("【Loop-FILE】处理：", dir_name)
        para['dir_name'] = dir_name
        file_suffix = str(time_name) + "_" + str(dir_name)
        print(f'{file_suffix=:}')

        ###################### 【样本配准】########################
        print("仿射变换")
        imagS_M = _m.main(
            ldr_i,mode=4,                      #mode=4 就是要计算一下45；6直接移动
            pgxy=np.copy(coordinate_beads),
            theta=np.copy(theta),
            scale=np.copy(scale),
            file_suffix=file_suffix,
            para=para,Registration=Registration
        )  # 配准的那个量也需要

        #####################创建基本文件夹#######################
        
        mask_file(para)                          #建立文件夹
        # HS_path=_U.save_path('Hs_base.npy',para) #保存HS文件
        # np.save(HS_path,Hs_base)
        # asd=asdd
        ########################BF处理##################################
        if para['BF_kongzhi']==1:
            imagS_M2 = _BF.BFmain(imagS_M)
        else:
            imagS_M2 = np.copy(imagS_M)
        del imagS_M
        ##################### 【提取样本-平均】##########################
        Dr_stack_cai = _G.get_rawdata(
            imagS_M2, para, file_suffix="", func_f=func_f
        )  # 2121 2115 219 123

        ###################WF+PZA#############
        # print('#############计算WF和PAZ###########')
        # Get_WF_PAZ(Dr_stack_cai,para)   
        
    return 



def _avg_dr_folder(para, dr_suffix=''):
    """Odd/even AVG under Dr/01-{W}{suffix} (e.g. '' or '_regOnly')."""
    FileName = _U.save_path("", para)
    w0 = para['W_order'][0]
    w1 = para['W_order'][1]
    if w0 != w1:
        _AD.main(FileName / 'Dr' / f'01-{w0}{dr_suffix}')
        _AD.main(FileName / 'Dr' / f'01-{w1}{dr_suffix}')
    else:
        _AD.main(FileName / 'Dr' / f'01-{w0}{dr_suffix}')


def _extract_and_avg_dr(
    imagS_M,
    para,
    func_f,
    *,
    dr_suffix='',
    brightness_before_records=None,
    track_label='',
):
    """BF → get_rawdata → channel AVG. dr_suffix selects Dr subfolder tag."""
    if track_label:
        print(f'【Dr写出】{track_label}  dr_name_suffix={dr_suffix!r}')
    if para['BF_kongzhi'] == 1:
        imagS_M2 = _BF.BFmain(imagS_M)
    else:
        imagS_M2 = np.copy(imagS_M)
    para['dr_name_suffix'] = dr_suffix
    try:
        dr_stack = _G.get_rawdata(
            imagS_M2,
            para,
            file_suffix="",
            func_f=func_f,
            brightness_before_records=brightness_before_records,
        )
        _avg_dr_folder(para, dr_suffix=dr_suffix)
    finally:
        para['dr_name_suffix'] = ''
    del imagS_M2
    return dr_stack


def Fine_Affine_3(input_path,Hs_base,para,Registration = 3,func_f=2121,Hs_sets=""):
    loop_dir = Path(str(input_path)).iterdir()  # 用于遍历总文件夹
    data_name = Path(str(input_path)).name      # 提取当日数据名称
    time_name = data_name[0:4]  # 提取时间
    para['data_name'] = data_name
    print(f'{data_name=:},{time_name=:}')
    save_reg_only = int(para.get('save_reg_only', 0) or 0)
    ###############################【loop 配准各种样本】
    i = 0
    for ldr_i in loop_dir:
        i += 1
        dir_name = ldr_i.name                               # 获取文件夹名
        if 'C3DSIM' in dir_name :                       # 跳过3DSIM文件夹
            print("【Loop-FILE】1跳过：", dir_name)
            continue
        elif 'peizhun' in dir_name :                       # 跳过配准文件夹
            print("【Loop-FILE】2跳过：", dir_name)
            continue
        elif 'cam1' in dir_name:
            print("【Loop-FILE】3跳过：", dir_name)
            continue
        elif 'cam2' in dir_name:
            print("【Loop-FILE】4跳过：", dir_name)
            continue
        elif ldr_i.is_dir() != 1:                           # 跳过文件
            print("【Loop-FILE】5跳过：", dir_name)
            continue
        
        if i not in para['FANWEI']:
            print("【Loop-FILE】跳过：", dir_name)
            continue
        
        print("【Loop-FILE】处理：", dir_name)
        para['dir_name'] = dir_name
        file_suffix = str(time_name) + "_" + str(dir_name)
        print(f'{file_suffix=:}')

        ###############循环读入数据###############
        if Hs_sets!='':
            # print(np.array(Hs_sets).shape)
            indexx = para['FANWEI'].index(i)
            Hs_base = Hs_sets[indexx]
            para['peizhun_file'] = para['FANWEI_peizhun'][indexx]
        ###################### 【样本配准】########################
        print("透视变换，SIFT method")
        imagS_M = _m.main(
            ldr_i/para['subname'], mode=6, Registration=3, Hs=Hs_base, file_suffix=file_suffix,para=para
        )  # 配准的那个量也需要

        brightness_before_records = _G.collect_brightness_records(
            imagS_M, para, func_f, stage='before-correction'
        )

        ###################剔除异常时间点#####################
        # imagS_M = DeleteError(imagS_M,FANWEI=para['errorpoint'])  #这里会在里面-1

        #####################创建基本文件夹#######################
        mask_file(para)                          #建立文件夹
        HS_path=_U.save_path('Hs_base.npy',para) #保存HS文件
        # HS_path_txt=_U.save_path('Hs_base.txt',para) #保存HS文件
        np.save(HS_path,Hs_base)

        from datetime import datetime #保存参数
        now = datetime.now().strftime("%H-%M-%S")
        txt_path = _U.save_path(f'para-{now}.txt',para)
        _U.save_txt(txt_path,para)
        # np.savetxt(HS_path_txt, Hs_base)

        # ---------- B 轨（可选）：背景已扣 + 配准；不做区域/方向亮度矫正 ----------
        # save_reg_only=1 时额外写出 Dr/01-488_regOnly（及 AVG）；SIMcheck 可用；recon 仍只用 A
        if save_reg_only:
            print(
                '【save_reg_only=1】写出 B 轨 Dr：仅配准+bleach，'
                '跳过 prism/ori 亮度矫正 → 01-*-regOnly'
            )
            imagS_M_B = bleach(np.copy(imagS_M))
            _extract_and_avg_dr(
                imagS_M_B,
                para,
                func_f,
                dr_suffix='_regOnly',
                brightness_before_records=brightness_before_records,
                track_label='B_regOnly',
            )
            del imagS_M_B

        ###################【样本配准完之后，在做分光校正】A 轨（默认 / recon）########
        if para['W_order'][0]!=para['W_order'][1]:  #dual color
            _U.save_tif('imagS_M_nojust_488.tif',imagS_M.reshape(8,-1,30,512,512)[:,0::2,:,:])
            _U.save_tif('imagS_M_nojust_561.tif',imagS_M.reshape(8,-1,30,512,512)[:,1::2,:,:])
        
            imagS_M = _m.prism_correct_2(imagS_M,margin=100) #不同通道区域分光矫正
            _U.save_tif('imagS_M_just1_488.tif',imagS_M.reshape(8,-1,30,512,512)[:,0::2,:,:])
            _U.save_tif('imagS_M_just1_561.tif',imagS_M.reshape(8,-1,30,512,512)[:,1::2,:,:])
            imagS_M = _m.ori_correct_2(imagS_M) #不同方向分光矫正
            _U.save_tif('imagS_M_just2_488.tif',imagS_M.reshape(8,-1,30,512,512)[:,0::2,:,:])
            _U.save_tif('imagS_M_just2_561.tif',imagS_M.reshape(8,-1,30,512,512)[:,1::2,:,:])

            # imagS_M = _m.phase_correct_2(imagS_M) #不同相位分光矫正
            
            imagS_M = bleach(imagS_M)
            _U.save_tif('imagS_M_just3_488.tif',imagS_M.reshape(8,-1,30,512,512)[:,0::2,:,:])
            _U.save_tif('imagS_M_just3_561.tif',imagS_M.reshape(8,-1,30,512,512)[:,1::2,:,:])

        else:
            imagS_M = _m.prism_correct(imagS_M)  #不同区域分光矫正
            imagS_M = bleach(imagS_M)
         
        ##################### 【提取样本-平均】A：亮度矫正+配准 → 01-488 ##########################
        Dr_stack_cai = _extract_and_avg_dr(
            imagS_M,
            para,
            func_f,
            dr_suffix='',
            brightness_before_records=brightness_before_records,
            track_label='A_bright+reg',
        )
        del imagS_M
        ###################WF+PZA#############
        # print('#############计算WF和PAZ###########')
        # if para["nxny"]!=para["nyny"]: #不相等就不做WF
        #     Get_WF_PAZ(Dr_stack_cai,para)
        #     pass
        # elif para["nxny"]==para["nyny"]: #相等情况才做WF
        #     Get_WF_PAZ(Dr_stack_cai,para)
    _U.print_quality_warning_summary(para)
    return 
    



