import argparse
import re

import os
from pathlib import Path

import DataProcess as _DP
import numpy as np
import Para_Estimate as _PE
import pyotf
import Seperate_CK as _SC
import torch
import Utils as _U
import Wiener_Reconstruction as _WR
import WindowF as _WF
from numpy.fft import fftshift, ifftshift
from skimage import restoration  # RL-反卷积
from tifffile import imread, imwrite  # ###
from torch.fft import fftshift as tfftshift
from torch.fft import ifftshift as tifftshift
# from loguru import logger
# %matplotlib inline
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_DR_TP_RE = re.compile(r"_(\d{3,5})\.(?:tif|tiff)$", re.I)


def _dr_timepoint_index(path: Path):
    """Parse trailing _0001 / _0003 from Dr stack filename; None if missing."""
    m = _DR_TP_RE.search(path.name)
    return int(m.group(1)) if m else None


def _list_dr_files_for_recon(file_name1, run_parameter=None):
    """
    List Dr TIFF stacks for recon, sorted by name.

    Default: skip first warm-up timepoints (_0001/_0002), start from _0003.
    If no _0003 (or later) exists, fall back to reading from _0001.

    Override with run_parameter['dr_start_timepoint'] (int, default 3).
    Set to 1 to keep legacy behaviour (read from first file).
    """
    run_parameter = run_parameter or {}
    root = Path(str(file_name1))
    files = sorted(
        (
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda p: p.name,
    )
    tp_whitelist = run_parameter.get("dr_timepoints")
    if tp_whitelist:
        try:
            allowed = {int(t) for t in tp_whitelist}
        except (TypeError, ValueError):
            allowed = set()
        if allowed:
            selected = [
                p
                for p in files
                if (ti := _dr_timepoint_index(p)) is not None and ti in allowed
            ]
            if selected:
                print(
                    f"【Dr timepoint】whitelist {sorted(allowed)} -> "
                    f"{len(selected)} files: {[p.name for p in selected]}"
                )
                return selected
    try:
        start_tp = int(run_parameter.get("dr_start_timepoint", 3))
    except (TypeError, ValueError):
        start_tp = 3
    if start_tp < 1:
        start_tp = 1

    if start_tp <= 1 or not files:
        return files

    preferred = [
        p
        for p in files
        if (ti := _dr_timepoint_index(p)) is not None and ti >= start_tp
    ]
    if preferred:
        print(
            f"【Dr timepoint】skip tp < {start_tp:04d} "
            f"(warm-up); start from {preferred[0].name} (n={len(preferred)})"
        )
        return preferred

    print(
        f"【Dr timepoint】no _{start_tp:04d}+ found; "
        f"fallback from first file (n={len(files)})"
    )
    return files

def FM_fix_error(temp,w3,size = 10):
    #对FM 倒过来之后，有异常亮点的去除map
    import cv2
    temps=[]
    nz,_,_ = temp.shape
    for i in range(nz):
        sobelx = cv2.Sobel(temp[i],cv2.CV_64F,1,0,ksize=3)#默认ksize=3
        sobely = cv2.Sobel(temp[i],cv2.CV_64F,0,1)
        gm = cv2.sqrt(sobelx ** 2 + sobely ** 2)
        
        gm_1 = gm    #gm 1阶倒
        #接一个关操作
        gm = _U.Cir_dilate(gm,size=size)
        gm = _U.Cir_Erode(gm,size=size)

        #取log，截断第99位点之后的异常值
        gm= np.log(gm+1)
        gm_ = np.percentile(gm, 99)
        gm[gm> (gm_*2)] = w3
        print('截断阈值',gm_)
        temps.append(gm)
    temps= np.array(temps)
    return temps,gm_1      #根据gm_1设置关操作的size大小，最终temps 是一群数值差不多的值

def T_conv3D(image,psf):
    '要求传入3D的空域图像，psf，返回3D卷积结果'
    from fft_conv_pytorch import fft_conv,FFTConv3d
    
    assert len(image.shape)==3
    assert len(psf.shape)==3
    
    # inz,inx,iny = image.shape
    pnz,pnx,pny = psf.shape
    
    '1.to torch'
    timage = _U.ToTensor(image).to(torch.float64).view(1,1,*image.shape)
    tpsf   = _U.ToTensor(psf).to(torch.float64).view(1,1,*psf.shape)
    
    '2.fft-conv'
    with torch.no_grad():
        fft_conv = FFTConv3d(1, 1,(0),padding=(pnz//2,pnx//2,pny//2), padding_mode='constant',bias=False) #  卷积之后的尺寸为 2p+1
        fft_conv.weight = torch.nn.Parameter(tpsf) 
        out = fft_conv(timage)
    
    '3.del'
    out = out[0,0,:-1,:-1,:-1]             #删除多余的
    # out = ifftn_RFN(out)

    return _U.image_reshape(out,dim=3).numpy()     #维度修正/numpy

def main(opt,file_name1,file_name2,file_name3,run_parameter,isdebug=False):
    #file operation
    # Default: skip warm-up _0001/_0002, start from _0003 (see _list_dr_files_for_recon).
    files = _list_dr_files_for_recon(file_name1, run_parameter)
    p2 = Path(str(file_name2)).iterdir()
    
    if Path(str(file_name2)).is_dir() and not any(p2): #OTF check
        # run_parameter['createOTF']==1
        # print('OTF文件夹无OTF，后面根据参数产生OTF中....')c
        raise('OTF not found based on the OTF path')
    else:
        p2 = Path(str(file_name2)).iterdir() #
    

    same_region_fai = []  #存放同region的 不同时间点
    same_region_pxy = []
    p1_i_ = -1
    for p1_i in files:
        if not p1_i.is_file():           
            continue
        
        p1_i_+=1  # 012345 开始
        if opt.sequence==55 and p1_i_%2==1 : 
            print("Skip:",p1_i)
            continue
        
        print("p1_i:",p1_i)
        filename = p1_i.name[-12:-4] #获得名字            #[01]_01
        run_parameter['filename'] = filename
        print('**Suffix**:',filename)
        
        #######################################1[ load Dr ]#########################################
        print("1 load raw data------------")
        #'raw data'
        Dr4D_ZPA,opt = _DP.loadANDprocess_Dr(opt,p1_i,run_parameter) #load; Ajust; ZPA; 加窗

        #Zpadding
        Dr = _DP.Image_zpadding_WF(opt,Dr4D_ZPA,xishu=run_parameter['pad_xishu'])
        # Dr = _DP.Image_zpadding_replease(opt,Dr4D_ZPA,xishu=run_parameter['pad_xishu'])               
               
        del Dr4D_ZPA
        
        #save WF
        if isdebug['loa']==1:
            movex = run_parameter['movex']
            movey = run_parameter['movey']
            _U.save_tif_v21(rf'WF8-rawdataSUM-{movex,movey}.tif',np.mean(Dr,axis=0),run_parameter)#每个Z，将三方向不同相位相加，形成伪WF
            _U.save_tif_v21('Dr.tif',Dr[[0,1,2]],run_parameter)
        
        #
        if opt.sequence==55 or opt.sequence==5343 or opt.sequence==5352:
            #再读入一组时间点进来
            # print('\n 读入第二组Dr_2进来')
            idx = p1_i_+1
            if idx >= len(files):
                raise IndexError(
                    f"sequence={opt.sequence} needs pair for {p1_i.name}, "
                    f"but only {len(files)} Dr files after timepoint filter"
                )
            p2_i = files[idx]
            Dr4D_ZPA2,opt = _DP.loadANDprocess_Dr(opt,p2_i,run_parameter)
            Dr_2 = _DP.Image_zpadding_WF(opt,Dr4D_ZPA2,xishu=run_parameter['pad_xishu'])
            # Dr_2 = _DP.Image_zpadding_replease(opt,Dr4D_ZPA2,xishu=run_parameter['pad_xishu'])               

            del Dr4D_ZPA2
            
            if isdebug['loa']==1:
                _U.save_tif_v21('Dr_2.tif',Dr_2[[0,1,2]],run_parameter)
            
        print("#########Dr,opt finish#######")
        print(f'######## rawdata nc = {opt.Nangles*opt.Nshifts},nz_re = {opt.nz_re},nz = {opt.nz},nx = {opt.nx},ny = {opt.ny}')
        print('######## opt.fc={},\n opt.fc_z={},\n opt.pxy={},\n opt.pz={}'.format(np.round(opt.fc,2),np.round(opt.fc_z,2),np.round(opt.pxy,2),np.round(opt.pz,2)))
        print(f'################################# ')
        #####################################2[ OTF] #####################################################
        #load_OTF
        if run_parameter['createOTF']==0:       #load otf
            for p2_k in p2:
                if not p2_k.is_file():           #
                    continue                
                #MP/2D       (nz，nx，ny)    (nx,ny)
                OTF3D,OTF2D,psf = _DP.load_OTF(opt,p2_k,Debug=isdebug['loa'])
                OTF3D[abs(OTF3D)<1e-10]=0
                OTF2D[abs(OTF2D)<1e-10]=0
                
                if isdebug['loa']==1:
                    _U.save_tif_v21('OTF3D_load.tif',abs(OTF3D),run_parameter)
                    _U.save_tif_v21('OTF2D_load.tif',abs(OTF2D),run_parameter)
        else:
            raise('can\'t generate 3DSIM OTF,please input OTF ')
        
        # Spatial_interpolation
        if opt.nz !=1 and opt.Zpadding!=0 and OTF3D.shape[0]!=(opt.nz+2*opt.Zpadding):  #如果padding了， and OTF与padding之后不相等
            nz,nx,ny = OTF3D.shape
            PSF3D =  np.fft.fftshift(_U.ifftn(OTF3D)).real
            PSF3D = _U.pad_or_cut(PSF3D,x=nx,y=ny,z=opt.nz+2*opt.Zpadding)
            OTF3D = abs(_U.fftn(np.fft.ifftshift(PSF3D)))

            if isdebug['loa']==1:
                _U.save_tif_v21('OTF3D_Spatial_interpolation.tif',abs(OTF3D),run_parameter)
            
        
        #####################################3 [OTFmask]#####################################################
        print("3 OTFmask generate---------")
        # # OTFmask3D/2D
        OTFmask3D,OTFmask2D = _DP.generatemask(opt,isdebug)
      
        assert len(OTFmask3D.shape)==3                 # (nz_re，2nx，2ny)
        assert len(OTFmask2D.shape)==2                 # (2nx,2ny)
        
        if isdebug['loa']==1:
            _U.save_tif_v21( 'OTFmask3D.tif', OTFmask3D, run_parameter )
            
        #亚像素移动
        # ZV,XV,YV = _U.meshgrid(opt,kuodaxy=1,kuodaz=1)
        # shiftmat = np.exp(2j*np.pi*(0.5/462*XV+0.5/462*YV + 0*ZV),dtype=opt.dtype_com)
        # OTFmask3D_move = _U.fftshift(_U.fftn( _U.ifftshift(_U.ifftn(OTFmask3D)) * shiftmat)).real
        
        # _U.save_tif(r'OTFmask3D_move.tif',OTFmask3D_move)
        ##################################[Lateralx2  Axialx2] OTF3D OTF2D、OTFmask3D OTFmask2D  #######################
        
        opt.OTF = _U.pad_or_cut((OTF3D),x=2*opt.nx,y=2*opt.ny,z=opt.nz_re)
        opt.OTF2D = _U.pad_or_cut((OTF2D),x=2*opt.nx,y=2*opt.ny,z=1)
        # opt.psf = psf
        opt.OTFmask3D = _U.pad_or_cut(OTFmask3D,x=2*opt.nx,y=2*opt.ny,z=opt.nz_re)
        opt.OTFmask2D = _U.pad_or_cut(OTFmask2D,x=2*opt.nx,y=2*opt.ny,z=1)

        if isdebug['loa']==1:
            _U.save_tif_v21( 'opt.OTF.tif', abs(opt.OTF), run_parameter )
            _U.save_tif_v21( 'opt.OTF2D.tif', abs(opt.OTF2D), run_parameter )
                    
            _U.save_tif_v21( 'opt.OTFmask3D.tif', opt.OTFmask3D, run_parameter )
            _U.save_tif_v21( 'opt.OTFmask2D.tif', opt.OTFmask2D, run_parameter )
        
    
        #####################################4 [分离分量]##############################
        print("2 分离分量------------")    
        for alpha_i in [0]:
            if opt.sequence==55:
                #Dr,Dr_2 都是
                assert Dr.shape[0] == 15
                assert Dr_2.shape[0] == 15
                print("first data sep.....")
                ck1 = _SC.separate_step_one(opt,Dr,run_parameter,Nshifts=5,Debug=isdebug['sep'])  
                print("second data sep.....")
                ck2 = _SC.separate_step_one(opt,Dr_2,run_parameter,Nshifts=5,Debug=isdebug['sep'])  

                
                print(" 参数估计-1阶段-----主要为2阶PGxy-----------")
                flag_temp = 0 #是否加载cfai参数  0 not load  1 load
                cfai_file = _U.save_path(f'cfai',run_parameter)   #make 'cfai' folder
                'get PGxy(3,5,3)'
                if not cfai_file.exists():
                    cfai_file.mkdir()
                
                if run_parameter['loadflag'] == 0:                #cal all
                    '(3,5,2)'
                    PGxy = _PE.parameter_estimationP2(opt,ck1,Debug=isdebug['est'],filename=filename+'5x5',run_parameter=run_parameter)
                elif run_parameter['loadflag'] == 1:
                    '(3,7,2)/(3,7,3)'
                    cfai,PGxy,_,_ = _PE.loadparameter(opt,file_name3,run_parameter)
                    if run_parameter['remove2']==1 or run_parameter['joint_deconvolution']==1: 
                        cfai02 = _PE.loadparameter4(opt,file_name3,run_parameter)
                    flag_temp = 1
                elif run_parameter['loadflag'] == 3:                #load PGXY, cal pz/cfai + not save
                    '(3,7,2)'
                    PGxy,_,_ = _PE.loadparameter2(opt,file_name3,run_parameter) 
                    
                elif run_parameter['loadflag'] == 4:                #try load PGxy，load cfai  [0+3]
                    try:  #加载P
                        PGxy,_,_ = _PE.loadparameter2(opt,file_name3,run_parameter) 
                    except:
                        pass
                    if flag_temp==0: # 从0计算
                        PGxy = _PE.parameter_estimationP2(opt,ck1,Debug=isdebug['est'],filename=filename+'5x5',run_parameter=run_parameter)
                            
                elif run_parameter['loadflag'] == 5:                #try load PGXY，CFAI, else cal PGXY，CFAI 
                    try:   #flag_temp=1
                        cfai,PGxy,_,_ = _PE.loadparameter(opt,file_name3,run_parameter)
                        if run_parameter['remove2']==1 or run_parameter['joint_deconvolution']==1: 
                            cfai02 = _PE.loadparameter4(opt,file_name3,run_parameter)
                        flag_temp = 1
                    except:  #
                        pass
                    if flag_temp==0: #从头计算
                        print('从头开始计算')
                        PGxy = _PE.parameter_estimationP2(opt,ck1,Debug=isdebug['est'],filename=filename+'5x5',run_parameter=run_parameter)
                
                PGxy_2 = PGxy[:,[0,1,2],:]   #PGxy_2（3,3）
               ################以上加载的PGxy 为2阶长度，cfai为第一阶段需要消除的初相，不能用在第二阶段/ 当remove2 == 0 时 又需要用在第二阶段###################
               
                '计算2阶初相 #得到（3,3,3）的2阶初相'
                print(" 参数估计-1阶段-----主要为2阶cfai-----------")
                if flag_temp == 0 : #计算得到2阶初相
                    #①计算2阶初相 (每方向计算两阶的cfai) (3,3,3)
                    cfai_02_ck1 = _PE.parameter_estimation_02(opt,ck1[[0,1,2,5,6,7,10,11,12]],PGxy_2,Norders=3,Debug=isdebug['est'],filename=filename+"-ck1",run_parameter=run_parameter)
                    cfai_02_ck2 = _PE.parameter_estimation_02(opt,ck2[[0,1,2,5,6,7,10,11,12]],PGxy_2,Norders=3,Debug=isdebug['est'],filename=filename+"-ck2",run_parameter=run_parameter)
                    
                    # #两个时间点的0阶相加
                    # ck3 = np.copy(ck1)
                    # ck3[0] = ck1[0] + ck2[0]
                    # ck3[5] = ck1[5] + ck2[5]
                    # ck3[10] = ck1[10] + ck2[10]
                    # cfai_02_ck3_0 = _PE.parameter_estimation_02(opt,ck3[[0,1,2,5,6,7,10,11,12]],PGxy_2,Norders=3,Debug=isdebug['est'],filename=filename+"-ck2",run_parameter=run_parameter)
                    
                    # ck4 = np.copy(ck2)
                    # ck4[0] = ck1[0] + ck2[0]
                    # ck4[5] = ck1[5] + ck2[5]
                    # ck4[10] = ck1[10] + ck2[10]
                    # cfai_02_ck4_0 = _PE.parameter_estimation_02(opt,ck4[[0,1,2,5,6,7,10,11,12]],PGxy_2,Norders=3,Debug=isdebug['est'],filename=filename+"-ck2",run_parameter=run_parameter)
                
                elif flag_temp == 1 :   #load get 2th order phase
                    cfai_02_ck1 = cfai[:,[0,1,2],:]   #load from file
                    cfai_02_ck2 = cfai02              #load from file
                    

                
                
                "得到ck 7分量  (是否消去2阶初相再计算) " #
                if run_parameter['remove2']==1:
                    print(r'remove 2th-order phase')
        
                    #get （3,5,3）的2th/1st phase
                    cfai_021_ck1 = np.zeros((3,5,3))
                    cfai_021_ck2 = np.zeros((3,5,3))
                    for cfai_021_ck1_i in range(3): 
                        cfai_021_ck1[cfai_021_ck1_i,0:3] = cfai_02_ck1[cfai_021_ck1_i,0:3]
                        cfai_021_ck1[cfai_021_ck1_i,3:5] = cfai_02_ck1[cfai_021_ck1_i,1:3]/2
                            
                        cfai_021_ck2[cfai_021_ck1_i,0:3] = cfai_02_ck2[cfai_021_ck1_i,0:3]
                        cfai_021_ck2[cfai_021_ck1_i,3:5] = cfai_02_ck2[cfai_021_ck1_i,1:3]/2
                            
                    #消去,后面的cfai 令之为0
                    ck1_wofai = _WR.removefai(opt,ck1,cfai_021_ck1)
                    ck2_wofai = _WR.removefai(opt,ck2,cfai_021_ck2) #
                    # ck2_wofai = _WR.removefai(opt,ck2,cfai_021_ck1) 
                    
                    "Obtaining a 7th-order spectrum by solving two sets of 1st-order spectra simultaneously. 联立两次的ck1 ck2 得到1阶"
                    ck = _SC.GETck(opt,ck1_wofai,ck2_wofai,run_parameter) #ck为7阶  取ck1
                    
                else:
                    ck = _SC.GETck(opt,ck1,ck2,run_parameter) #ck为7阶  取ck1
                
                
                
                
                #【Joint deconv】
                if opt.sequence==55 and run_parameter['joint_deconvolution']==1: #
                    ck_02 = np.zeros_like(ck[0:9])
                    for iii in range(3):
                        for jjj in [0,1,2]:
                            if run_parameter['remove2']==1:  
                                ck_02[3*iii+jjj] = np.copy(ck2_wofai[iii*5+jjj])
                            else:
                                ck_02[3*iii+jjj] = np.copy(ck2[iii*5+jjj])  
                    
            elif  opt.sequence==52:
                assert Dr.shape[0] == 21
                print("7X7 sep.....")
                ck = _SC.separate(opt,Dr,run_parameter,Nshifts=7,Debug=isdebug[2],alpha=alpha_i)  

            else:
                raise('sequcen error')
            
            #todo
            if run_parameter['use_mask']==1:
                if run_parameter['mear'] ==1:
                    jieshu = [0]
                    jieshu2 = [0]
                else:
                    jieshu = [0,1,2,3,4,5,6]
                    jieshu2 = [0,1,2]
                    
                for iiii in range(3):
                    for jjjj in jieshu:
                        ck[iiii*7+jjjj] = ck[iiii*7+jjjj] * _U.pad_or_cut(opt.OTFmask3D,opt.nx,opt.ny,opt.nz+2*opt.Zpadding)
                    if run_parameter['joint_deconvolution']==1:
                        for kkkk in jieshu2:
                            ck_02[iiii*3+kkkk] = ck_02[iiii*3+kkkk] * _U.pad_or_cut(opt.OTFmask3D,opt.nx,opt.ny,opt.nz+2*opt.Zpadding)
            
            if isdebug['sep']==1:
                _U.save_tif_v21(f'【】ck7X7-[0123456]-{filename}.tif',abs( ck[[0,1,2,3,4,5,6]] ),run_parameter)
                
                
            filename = p1_i.name[-8:-4] #获得4位名字 
            assert len(ck.shape)==4               # (21,nz,nx,ny)

            temppp=5
            if ("001" in filename or run_parameter['save']==1) and isdebug['sep']==1:
                _U.save_tif_v21(f'【】ck-[0123456]-{filename}-r.tif',abs( ck[[0,1,2,3,4,5,6]] ),run_parameter)
                _U.save_tif_v21(f'【】ck2-[0123456]-{filename}-r.tif',abs( ck[[0+temppp,1+temppp,2+temppp,3+temppp,4+temppp,5+temppp,6+temppp]] ),run_parameter)
                _U.save_tif_v21(f'【】ck3-[0123456]-{filename}-r.tif',abs( ck[[0+temppp*2,1+temppp*2,2+temppp*2,3+temppp*2,4+temppp*2,5+temppp*2,6+temppp*2]] ),run_parameter)
        
            del ck1,ck2
            if run_parameter['remove2']==1:
                del ck1_wofai,ck2_wofai
            ######################################5 [1 order parameter estimate]##############################
            ##################################到这一步其实已经有PGxy了，不需要再算一遍PGxy###
            print()
            print("4 参数估计 2 level-----主要为1阶-----------")

            '将PGxy_2(3,3,2)->(3,7,2)'
            # PGxy_2
            PGxy = np.zeros((3,7,2))
            PGxy[:,0:3,:] = PGxy_2
            PGxy[:,3,:] = PGxy[:,1,:] / 2
            PGxy[:,4,:] = PGxy[:,1,:] / 2
            PGxy[:,5,:] = PGxy[:,2,:] / 2
            PGxy[:,6,:] = PGxy[:,2,:] / 2

            'get 1 order parameter'
            if flag_temp==0: #计算得到1阶初相
                print('cal first-order')
                cfai_1jie7,_,_,_,_ = _PE.parameter_estimation3(opt,ck,PGxy,Debug=isdebug['est'],filename=filename+'2 level',run_parameter=run_parameter,jieshu = [3,4,5,6]) 
                cfai_1jie = cfai_1jie7[:,[3,4,5,6],:]
            elif flag_temp==1: #加载得到1阶初相
                print('load first-order parameter')
                cfai_1jie = cfai[:,[3,4,5,6],:]
            
            'save,princt PGxy(3,7,2)、Cfai(3,7,3) '
            #整合，统计
            def parameter_Finalsetting(opt,PGxy,cfai_02_ck1,cfai_1jie,cfai_02_ck2,run_parameter):
                cfai = np.zeros((3,7,3))
                cfai[:,[0,1,2],:] = cfai_02_ck1
                cfai[:,[3,4,5,6],:] = cfai_1jie#[:,[3,4,5,6],:]
            
                cfai_02  = np.copy(cfai_02_ck2)
                cfai[:,0,2] = 1
                
                PGz = np.zeros((3,7))
                if run_parameter['pzzz'] != 0 and opt.flag_N=='MP':
                    # print('手动设置PZ为',run_parameter['pzzz'])
                    PGz[:,[3,5]] = -run_parameter['pzzz']
                    PGz[:,[4,6]] = run_parameter['pzzz']
                
                P_long = (np.sqrt(PGxy[:,:,0]**2 + PGxy[:,:,1]**2)).round(2)
                Plong = np.mean(P_long[:,1])
                
                #cfai_save(2阶为需要消除的，1阶为后续除掉)
                cfai_save = np.copy(cfai)
                cfai_02_save = np.copy(cfai_02_ck2)
                if run_parameter['remove2']==1:
                    cfai[:,[0,1,2],1] = 0            #前面已经消除，所以这里置为0 
                    cfai_02[:,[0,1,2],1] = 0
                return PGxy,cfai,cfai_save,cfai_02,cfai_02_save,PGz,P_long,Plong
            
            PGxy,cfai,cfai_save,cfai_02,cfai_02_save,PGz,P_long,plong = parameter_Finalsetting(opt,PGxy,cfai_02_ck1,cfai_1jie,cfai_02_ck2,run_parameter)
            #保存文件
            if flag_temp==0 and run_parameter['loadflag'] != 3:
                _PE.saveparameter(file_name3,PGxy,PGz,cfai_save,run_parameter)  
                if opt.sequence==55 and run_parameter['joint_deconvolution']==1:  
                    _PE.saveparameter02(file_name3,cfai_02_save,run_parameter)         
            elif run_parameter['loadflag'] == 3:
                print('loadflag=3: calculated cfai/phase are not saved as parameter NPY')
                            
            same_region_fai.append(cfai)  #存放同region的 不同时间点
            same_region_pxy.append(PGxy)
            
            opt.plong = plong             
            
            #利用计算的PGzz 代替PGz
            PGzz,PGxy1 = _DP.get_pzp1(opt,PGxy)  #只是看看
            # PGz = PGzz 
            
            print('(只做参考)根据1阶计算的PGzz',PGzz)
            
            run_parameter['PGz_avg'] = np.mean(abs(PGz[:,-4:None])).round(3)
            ########################################[频谱重建]############################
            print()
            print(" 准备重建------------")
            ###########构建Ir:移动矩阵
            print(" make Ir---------")
            # Irtest,Irtestxy,Irtestz = _WR.IrTEST(opt,PGxy,PGz,kuodaz=opt.kuodapz,Zchushu=opt.kuodapz*(opt.nz+2*opt.Zpadding))
            Irtest,Irtest_1 = _WR.IrTEST(opt,PGxy,PGz,kuodaz=opt.kuodapz,Zchushu=opt.kuodapz*(opt.nz+2*opt.Zpadding))

            #####参数估计结束
            if run_parameter['just_cfai']:
                # same_region_fai.append(cfai)  #存放同region的 不同时间点
                # same_region_pxy.append(PGxy)
                break
            
            ###########重建开始########
            directions = run_parameter['directions'] #   #方向 
            BattStrFZs = run_parameter['attStrFZs']
            BattFWHMFZs = run_parameter['attFWHMFZs']
            BattStrs = run_parameter['attStrs']
            BattFWHMts = run_parameter['attFWHMs']
            
            for direction in directions:                           #【重建哪些方向】
                print(f'direction mode：{direction}..........')
                for order_flag in run_parameter['orderForRe']:      #【重建哪几阶】
                    # if order_flag==0:
                    #     plong2 = 0
                    # else:
                    #     plong2 = plong
                        
                    for Method_c in run_parameter['Method_c']:        #【C值方法】  4为方法2,5为常规方法1
                        for notch in run_parameter['notch']:      #是否使用notch，0 不加 ，1加
                            if notch==0:   #notch12 不用，不用循环那么多遍
                                run_parameter['attStrFZs'] = [0]
                                run_parameter['attFWHMFZs'] = [1]
                                run_parameter['attStrs'] = [0]
                                run_parameter['attFWHMs'] = [1]
                            else:         #notch12 用， 重新取回循环
                                run_parameter['attStrFZs'] = BattStrFZs
                                run_parameter['attFWHMFZs'] = BattFWHMFZs
                                run_parameter['attStrs'] = BattStrs
                                run_parameter['attFWHMs'] = BattFWHMts
                            
                            ######### ⑤构建分母
                            # run_parameter['decon_meth']=='4B':
                            Msum_mask = _WR.GetMsum_mask(opt,Irtest,run_parameter,flagg=order_flag,Debug=isdebug['mov'],direction=direction )
                            #生成FZotf
                            FZotf,FZotfmask = _WR.GetFZotf(opt,Irtest,order_flag=order_flag,run_parameter=run_parameter,PGxy=PGxy,PGz=PGz,direction=direction) #仅仅是生成了各个位置上的OTF，没有notch 没有mask 没有平方
                            FZotf = abs(FZotf)
                            
                            if isdebug['mov']==1:
                                _U.save_tif_v21(r'FZotf-abs.tif',abs(FZotf[[0,1,2,3,4]]),run_parameter)
                                _U.save_tif_v21(r'Msum_mask.tif',abs(Msum_mask),run_parameter)
                            
                            #移动分子 # 仅仅移动+paddingFZ
                            if run_parameter['joint_deconvolution']==0:
                                FZall = _WR.GetFZ(opt,ck,Irtest,order_flag=order_flag,run_parameter=run_parameter,direction=direction)
                            elif opt.sequence==55 and run_parameter['joint_deconvolution']==1:
                                FZall,FZall_02 = _WR.GetFZ_Joint(opt,ck,ck_02,Irtest,order_flag=order_flag,run_parameter=run_parameter,direction=direction)
                            
                            # FZall = FZall*FZotfmask
                            if isdebug['mov']==1:
                                _U.save_tif_v21(r'FZall.tif',abs(FZall[[0,1,2,3,4]]),run_parameter)
                            
                            
                            # cfai[:,3:7,0] = 0.3
                            
                            for notch0th in run_parameter['notch0ths']:  #0阶notch参数
                                run_parameter['notch0th'] = notch0th   
                                for attFWHM0 in run_parameter['attFWHM0s']:
                                    run_parameter['attFWHM0'] = attFWHM0
                                    for notch1st_pang in run_parameter['notch1st_pangs']: #1阶旁边
                                        run_parameter['notch1st_pang'] = notch1st_pang
                                        for attStrFZ in run_parameter['attStrFZs']:#2阶
                                            run_parameter['attStrFZ'] = attStrFZ
                                            for attFWHMFZ in run_parameter['attFWHMFZs']:
                                                run_parameter['attFWHMFZ'] = attFWHMFZ
                                                for attStr in run_parameter['attStrs']:#1阶
                                                    run_parameter['attStr'] = attStr
                                                    for attFWHM in run_parameter['attFWHMs']:
                                                        run_parameter['attFWHM'] = attFWHM
                                                        
                                                        w1 = 1
                                                        if run_parameter['joint_deconvolution']==0:
                                                            Wiener1 = _WR.GetWiener1(opt,_U.ToNumpy(FZall),_U.ToNumpy(FZotf),PGxy,PGz,cfai,w1,order_flag,run_parameter,notch,direction=[0,1,2])
                                                        elif opt.sequence==55 and run_parameter['joint_deconvolution']==1:
                                                            Wiener1 = _WR.GetWiener1_Joint(opt,_U.ToNumpy(FZall),_U.ToNumpy(FZall_02),_U.ToNumpy(FZotf),PGxy,PGz,cfai,cfai_02,w1,order_flag,run_parameter,notch,direction=[0,1,2])
                                                        
                                                        if isdebug['wie']==1:
                                                            _U.save_tif_v21('Wiener1-k-.tif',abs(Wiener1),run_parameter)
                                                        
                                                        for w2 in opt.W:
                                                            if  run_parameter['joint_deconvolution']==0:
                                                                filename2 = filename
                                                                Msum_all = _WR.GetWiener2(opt,_U.ToNumpy(Wiener1),_U.ToNumpy(FZotf),PGxy,PGz,cfai,w2,order_flag,run_parameter,notch,direction=[0,1,2])
                                                            elif opt.sequence==55 and run_parameter['joint_deconvolution']==1: #构建分母
                                                                filename2 = filename+'J'
                                                                Msum_all = _WR.GetWiener2_Joint(opt,_U.ToNumpy(Wiener1),_U.ToNumpy(FZotf),PGxy,PGz,cfai,cfai_02,w2,order_flag,run_parameter,notch,direction=[0,1,2])
                                                            
                                                            'Wiener'
                                                            Apo = _WF.bhs_3D_Z(opt,opt.plongxs*opt.plong,opt.fcxs*opt.fc,kuodaz=opt.kuodapz, Zjust=opt.Apoz )
                                                            
                                                            if run_parameter['mear'] != 1:
                                                                Apo = Apo**2
                                                            
                                                            Wiener2 =Wiener1/Msum_all
                                                            # Wiener2 = Wiener2*Msum_mask
                                                           
                                                            
                                                            if isdebug['wie']==1:
                                                                _U.save_tif_v21(f'Apo-GetWiener2-{opt.Apoz}.tif',abs(Apo),run_parameter)
                                                                _U.save_tif_v21(f'Wiener1-{opt.Apoz}.tif',abs(Wiener1),run_parameter)
                                                                _U.save_tif_v21(f'Wiener2noAPO-{opt.Apoz}-{w2}-{notch0th}.tif',abs(Wiener2),run_parameter)
                                                            
                                                            Wiener2 = Wiener2* Apo
                                                            
                                                            finalr,finalk = _WR.Final_ifft(opt,_U.ToNumpy(Wiener2),Method_c,w2,notch,order_flag,direction,run_parameter=run_parameter,filename=filename2,p1_i_=p1_i_)
        
        if  run_parameter['fast']==1:
            #退出文件遍历
            break
    if run_parameter['just_cfai']: #用于二次配准
        return cfai,PGxy,PGz,plong,same_region_fai,same_region_pxy,Irtest
    
    #保存
    from datetime import datetime
    now = datetime.now().strftime("%H-%M-%S")
    
    #保存估计参数（在第一层for 之后）
    PARA = {}
    PARA['PGxy'] = PGxy
    PARA['P_long'] = P_long
    PARA['PGz'] = PGz
    PARA['cfai_save'] = cfai_save
    PARA['cfai_02_save'] =  cfai_02_save
    Dr_sub = run_parameter['Dr_sub']
    txt_path = _U.save_path(f'PARA-【{Dr_sub}】-{now}.txt',run_parameter)
    _U.save_txt(txt_path,PARA)
    
    #保存运行参数（无需for循环）
    # from datetime import datetime
    # logger.info(f"**run_parameter**：\n{run_parameter}")
    # now = datetime.now().strftime("%H-%M-%S")
    # txt_path = _U.save_path(f'run_parameter-{filename}-{now}.txt',run_parameter)
    # _U.save_txt(txt_path,run_parameter)
        
    return opt,cfai,PGxy,PGz,plong,finalr,finalk,same_region_fai,same_region_pxy,

