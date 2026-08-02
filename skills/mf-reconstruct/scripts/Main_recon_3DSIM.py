import argparse
# import imp
from pathlib import Path
import numpy as np
import torch
import pyotf

import Utils as _U
import DataProcess as _DP
import Seperate_CK as _SC
import Para_Estimate as _PE
import Wiener_Reconstruction as _WR
import WindowF as _WF

from numpy.fft import fftshift, ifftshift
# from skimage import restoration  # RL-反卷积
from tifffile import imread, imwrite  # #####not ned
from torch.fft import fftshift as tfftshift
from torch.fft import ifftshift as tifftshift

# %matplotlib inline

def cutoff2D(OTF2D, fc):
    # OTF2D (3,opt.nx,opt.ny)

    _, nx, ny = OTF2D.shape
    _, XV, YV = _U.meshgridxy_self(nx=nx, ny=ny)
    kr = np.sqrt(XV**2 + YV**2)
    kr = _U.image_reshape(kr, dim=2)
    OTF2D[:, kr > fc] = 0
    return OTF2D

def cutoff4D(data, fc):
    # data (nx,nz,opt.nx,opt.ny)

    _, _, nx, ny = data.shape
    _, XV, YV = _U.meshgridxy_self(nx=nx, ny=ny)
    kr = np.sqrt(XV**2 + YV**2)
    kr = _U.image_reshape(kr, dim=2)
    data[:, :, kr > fc] = 0
    return data

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
        gm[gm> (gm_*1.5)] = w3
        print('截断阈值',gm_)
        temps.append(gm)
    temps= np.array(temps)
    return temps,gm_1      #根据gm_1设置关操作的size大小，最终temps 是一群数值差不多的值


def main(opt,file_name1,file_name2,file_name3,run_parameter,isdebug=False):
    '''
    1：Dr 文件夹
    2: OTF 文件夹
    3：parameter 参数估计文件夹
    loadflag：0 自己估计；1 加载之前的值
    '''
    #file operation
    p1 = Path(str(file_name1)).iterdir() #仅遍历当前目录下内容
    p2 = Path(str(file_name2)).iterdir() #仅遍历当前目录下内容
    
    if Path(str(file_name2)).is_dir() and not any(p2): #OTF文件夹为空，则自己产生OTF
        run_parameter['createOTF']==1
        print('OTF文件夹无OTF，后面根据参数产生OTF中....')
    else:
        p2 = Path(str(file_name2)).iterdir() #
    
    same_region_fai = []  #存放同region的 不同时间点
    same_region_pxy = []
#     print(file_name1)
    p1_i_ = -1
    for p1_i in p1:
        if not p1_i.is_file():           #忽略文件夹
            continue
        
        p1_i_+=1  #
        filename = p1_i.name[-7:-4]     #获取文件后缀名字
        run_parameter['filename'] = filename
        print(p1_i)
        print('后缀名:',filename)
        dir_name = run_parameter["dir_name"]
        #######################################1 [加载Dr OTF or PSF ,产生OTF]#####################################################
        print("1 加载数据------------")      
        #加载DR
        Dr,opt = _DP.load_RAW_DATA(opt,p1_i,run_parameter)    #return 4D(nc,nz,nx,ny)Dr, opt
        
        #边缘加窗
        Dr = _DP.windows(opt,Dr)
        
        #漂白校正
        if run_parameter['bleach']==1:
            print('进行光漂白校正')
            Dr = _DP.bleach_adjust(Dr)
            _U.save_tif_v21('Dr-mask-bleach.tif',Dr,run_parameter)
        
        #保存WF图像
        WF = np.sum(Dr,axis=0)
        if WF.shape[0]!=opt.nz:
            temp = WF[opt.Zpadding:-opt.Zpadding]
            temp_k = _U.pad_or_cut(_U.fftn(temp),x=2*opt.nx,y=2*opt.ny,z=opt.nz)
            temp_r = _U.ifftn(temp_k).real
            temp_r[temp_r<0] =0 
            del WF
            del temp_k
            WF = temp_r
        _U.save_tif_v21(f'WF-{dir_name}-{filename}.tif',WF,run_parameter)
        
        #加载OTF
        if run_parameter['createOTF']==0:       #读入OTF or PSF
            for p2_k in p2:
                if not p2_k.is_file():           #忽略文件夹
                    continue
                if opt.flag_N==5: #3D-SIM,      OTF为 (3，nz，2nx，2ny) (3,2nx,2ny)
                    OTF4D,OTF2D,OTFsm = _DP.load_OTF_3DSIM(opt,p2_k,Debug=isdebug[1]) 
                    
                else:             #MP/2D       (3,nz，2nx，2ny)    (3,2nx,2ny)
                    OTF4D,OTF2D,OTFsm,opt.psf = _DP.load_OTF(opt,p2_k,Debug=isdebug[1],PSF=run_parameter['PSF']) ##！！！把 OTF扩大放到外面来 扩大2*xy z视kuodapz情况
               
                # OTF4D[OTF4D<1e-10]=0
                # OTF2D[OTF2D<1e-10]=0
                _U.save_tif_v21('【1】OTF4D_实际.tif',OTF4D,run_parameter)
                _U.save_tif_v21('【1】OTF2D_实际.tif',OTF2D,run_parameter)
                
        elif run_parameter['createOTF']==1 and opt.flag_N!=5:    #产生OTF
            pass
            
        elif run_parameter['createOTF']==1 and opt.flag_N==5:   # 3D-SIM OTF重采样
            #重采样、强度归一化   #疑问，使用NMISOTF pz不是固定的吗？这样能行？
            pass
            
        else:
            raise('can\'t generate 3DSIM OTF ')
        
        # # opt.OTF2D fc截断
        # opt.OTF2D = cutoff2D(opt.OTF2D,opt.fc)
        
        assert len(OTF4D.shape)==4 #
        
        assert len(OTF2D.shape)==3 # ()
        _,nz_OTF4d,nx_OTF,ny_OTF = OTF4D.shape
         
        #对OTF 插值
        if (opt.nz !=1 and opt.Zpadding!=0) or (nz_OTF4d != opt.nz) :
            # nz,nx,ny = OTF3D.shape
            # OTF4D_chahzi = np.zeros( (3,opt.nz+2*opt.Zpadding,2*opt.nx,2*opt.ny) )
            # for i_otf in range(3):
                # OTF4D_chahzi[i_otf] = _U.Interpolate(OTF4D[i_otf],nx=nx_OTF,ny=ny_OTF,nz=opt.nz+2*opt.Zpadding)
            OTF4D= _U.Interpolate(OTF4D,nx=nx_OTF,ny=ny_OTF,nz=opt.nz_re)

            
        opt.OTF   = np.copy(OTF4D)
        opt.OTF2D = np.copy(OTF2D)
        _U.save_tif_v21('【1】opt.OTF.tif',abs(opt.OTF),run_parameter)
        # _U.save_tif('opt.OTF2D.tif',opt.OTF2D)
        #对于3D-SIM来说  opt.OTF[0] == H_zhongxin , opt.OTF[1] == H_yiweiP , opt.OTF[2] == H_yiweiPhalf  
        ###这里验证一下，  0 1经过归一化后是不是相同的数值
        
        
        
        if 1: #输出参数
            print()
            print('NA = {},refmed = {},lamdex = {},lamdem = {}'.format(opt.na, opt.n,opt.lamdex,opt.lamdem) )
            print('opt.fc={},opt.pxy= {},opt.pz={}'.format(np.round(opt.fc,3), np.round(opt.pxy,3), np.round(opt.pz,3) ) ) 
            print()
        
        #####################################3 OTFmask处理[其他数据处理]#####################################################
        print()
        PGxy = np.zeros((3,5,2))
        #得到OTFmask3D
        opt.OTFmask3D = _DP.otf_mask3D_SIM_PE(opt,opt.nx,opt.ny,opt.nz_re,PGxy,paddingxy=2)
        opt.OTFmask2D = _DP.otf_mask2D_SIM_PE(opt.OTFmask3D,piancha=0) 
        
        opt.OTFmask3D = np.copy(opt.OTF)
        opt.OTFmask3D[opt.OTFmask3D!=0] =1
        # opt.OTFmask3D[opt.OTFmask3D!=0] =1

        opt.OTFmask2D = opt.OTFmask3D[:,0,:,:]        
        # for i in range(3):
        #     temp = _PE.Cir_dilate_3D(opt.OTFmask3D[i].astype(np.uint8),size=2)
        #     temp = _PE.Cir_Erode_3D(temp,size=1)
        #     opt.OTFmask3D[i] = temp.astype(np.bool_)
            
        
        assert len(opt.OTFmask3D.shape)==4                 #(3，nz，2nx，2ny) 
        assert len(opt.OTFmask2D.shape)==3                 #(3,2nx,2ny)
        assert opt.OTFmask3D.shape[1] == opt.kuodapz*(opt.nz+2*opt.Zpadding) #
        assert opt.OTFmask3D.shape[2] == opt.kuodaxy*(opt.nx) #
        
        # opt.OTF = opt.OTF * opt.OTFmask3D   #【考虑删去】
        _U.save_tif_v21('opt.OTFmask3D.tif',opt.OTFmask3D,run_parameter)
        #####################################2 [分离分量]##############################
        print()
        print("2 分离分量------------")                  #分离之前没有做频谱截断
        
        ck = _SC.separate(opt,Dr,mask_flag=1,run_parameter=run_parameter,RL=run_parameter['RL'],re_mode=run_parameter['re_mode'],Debug=isdebug[2])      #只是分离出来了而已
        assert len(ck.shape)==4                      # (15,nz,nx,ny)
        # ck = cutoff4D(ck,opt.fc)           
        
        
        if run_parameter['mear']==1: #todo
            jieshu = 1
        else:
            jieshu = opt.Nshifts
        
        for ck_i in range(3):
            for ck_j in range(jieshu):
                ck[ck_i*opt.Nshifts+ck_j] =ck[ck_i*opt.Nshifts+ck_j] * _U.pad_or_cut(opt.OTFmask3D[0],opt.nx,opt.ny,opt.nz+2*opt.Zpadding)

        # _U.save_tif_v21(f'②Ck-[0,1,2].tif', abs(ck[[0,1,2]]) ,run_parameter) 
        _U.save_tif_v21(f'②Ck-[0-9].tif', abs(ck) * _U.pad_or_cut(opt.OTFmask2D[0],x=opt.nx,y=opt.ny) ,run_parameter)
        # ck_pe,ck_angle = drain_ck(ck)            # 这个函数取出每个频谱分量 最中间一层+相位层，用于参数估计
                                                 # （9，nx，ny）
        #因此下面内容需要更新 只传入ck_pe 进行
        ######################################4 [参数估计]##############################
        print()
        print("4 参数估计------------")
        cfai_file = _U.save_path(f'cfai',run_parameter)   #构建cfai文件夹
        if not cfai_file.exists():
            cfai_file.mkdir()
        #自己估计
        if run_parameter['loadflag'] == -1:   #自己估计  
            cfai,PGxy,PGz,plong,pgz = _PE.parameter_estimation_1(opt,ck,Debug=isdebug[4],filename=filename,run_parameter=run_parameter)
            _PE.saveparameter(file_name3,PGxy,PGz,cfai,run_parameter) 
            
        if run_parameter['loadflag'] == 0:   #自己估计  
            cfai,PGxy,PGz,plong,pgz = _PE.parameter_estimation(opt,ck,Debug=isdebug[4],filename=filename,run_parameter=run_parameter)
            _PE.saveparameter(file_name3,PGxy,PGz,cfai,run_parameter) 
            
        elif run_parameter['loadflag'] == 1:                #加载文件
            cfai,PGxy,PGz,plong = _PE.loadparameter(opt,file_name3)
        
        elif run_parameter['loadflag'] == 2:                #读取PGxy，自己求cfai  并保存为文件
            PGxy,plong = _PE.loadparameter2(opt,file_name3)
            cfai,PGxy,PGz,plong,pgz = _PE.parameter_estimation2(opt,ck,PGxy,Debug=isdebug[4],filename=filename,run_parameter=run_parameter)
            _PE.saveparameter(file_name3,PGxy,PGz,cfai,run_parameter) 
                
        elif run_parameter['loadflag'] == 3:                #读取PGxy，自己求cfai,
            PGxy,plong = _PE.loadparameter2(opt,file_name3)
            # cfai,PGxy,PGz,plong = _PE.loadparameter(opt,file_name3)
            cfai,PGxy,PGz,plong,pgz = _PE.parameter_estimation2(opt,ck,PGxy,Debug=isdebug[4],filename=filename,run_parameter=run_parameter)
            # cfai[:,:,1] =  cfai2[:,:,1]              #改为读取C，自己求fai，的操作  (3,X,3)
            
        elif run_parameter['loadflag'] == 4:                #尝试读取PGXY，，否者自己计算 #自己求pz，cfai +不存
            flag_temp=0
            print('尝试读取文件')
            try:
                PGxy,plong = _PE.loadparameter2(opt,file_name3)
                cfai,PGxy,PGz,plong,pgz = _PE.parameter_estimation2(opt,ck,PGxy,Debug=isdebug[4],filename=filename,run_parameter=run_parameter)
                flag_temp=1
            except:
                pass
            if flag_temp==0: #计算
                print('读取失败')
                cfai,PGxy,PGz,plong,pgz = _PE.parameter_estimation(opt,ck,Debug=isdebug[4],filename=filename,run_parameter=run_parameter)
                _PE.saveparameter(file_name3,PGxy,PGz,cfai,run_parameter) 

        elif run_parameter['loadflag'] == 5:
            flag_temp=0
            try:
                cfai,PGxy,PGz,plong = _PE.loadparameter(opt,file_name3)
                flag_temp=1
            except:
                pass
            if flag_temp==0: #计算
                cfai,PGxy,PGz,plong,pgz = _PE.parameter_estimation(opt,ck,Debug=isdebug[4],filename=filename,run_parameter=run_parameter)
                _PE.saveparameter(file_name3,PGxy,PGz,cfai,run_parameter) 
                
        
        same_region_fai.append(cfai)  #存放同region的 不同时间点
        same_region_pxy.append(PGxy)
        
        #设置c值
        if run_parameter['cfaii'] != 0:
            cfai[:,1:,0] = run_parameter['cfaii']
        #设置φ值    
        
        P_long = (np.sqrt(PGxy[:,:,0]**2 + PGxy[:,:,1]**2)).round(2)
        opt.plong=plong
        print('后缀名:',filename)
        print(PGxy)
        print(P_long)
        print(plong)
        print('cfai=',cfai.round(2))
        
        #利用计算的PGzz 代替PGz
        PGzz,PGxy1 = _DP.get_pzp1(opt,PGxy)
        PGz = PGzz 
        print('3D的PGz不会移动的',PGz)
        
        ########################################[频谱重建]############################
        print()
        print(" 准备重建------------")
        # opt.OTF[opt.OTF>0.4]=opt.OTF*1.5
        # opt.OTF = np.where(opt.OTF>0.2, 1.5*opt.OTF,opt.OTF )
        # _U.save_tif_v21('opt.OTF.tif',opt.OTF,run_parameter)
        ###########构建Ir:移动矩阵
        # Irtest,Irtestxy,Irtestz = _WR.IrTEST(opt,PGxy,PGz,kuodaz=opt.kuodapz,Zchushu=opt.kuodapz*(opt.nz+2*opt.Zpadding),pz_true=run_parameter['pzzz'])
        Irtest,Irtest_1 = _WR.IrTEST3D_SIM(opt,PGxy,PGz,kuodaz=opt.kuodapz,Zchushu=opt.kuodapz*(opt.nz+2*opt.Zpadding),pz_true=run_parameter['pzzz'])
        Irtestxy=0
        Irtestz=0
        if run_parameter['just_cfai']:
            """只做参数估计，skip后面steps"""
            continue
        
        directions=run_parameter['directions'] # [[0,1,2]]   #方向 #!!!!!!!!!!!!!!!!!!!!!!!!
        BattStrFZs = (run_parameter['attStrFZs'])
        BattFWHMFZs = (run_parameter['attFWHMFZs'])
        BattStrs = (run_parameter['attStrs'])
        BattFWHMts = (run_parameter['attFWHMs'])
        print('下面开始重建')
        for direction in directions:                          #遍历方向模式
            print(f'direction mode：{direction}..........')
            for order_flag in run_parameter['orderForRe']:    #0 0th  ；1 ±1th ；2 ±2th #
                print(f'  order mode:{order_flag}..')
                for Method_c in run_parameter['Method_c']:      #4为方法2,5为常规方法1
                    
                    ########## ⑤构建分母---------------
                    # OTFmask3D_all = _DP.otf_mask3D_SIM(opt,opt.nx,opt.ny,opt.nz_re,PGxy,paddingxy=2)
                    #对上面数据进行膨胀操作，消除1个像素的值
                    # for i in range(15):
                    #     temp = _PE.Cir_dilate_3D(OTFmask3D_all[i].astype(np.uint8),size=3)
                    #     temp = _PE.Cir_Erode_3D(temp,size=3)
                    #     OTFmask3D_all[i] = temp.astype(np.bool_)
                    ###################---------------
    
                    for notch in run_parameter['notch']:      #是否使用notch，0 不加 ，1加
                        #先保存好这些记录
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

                        # Msum,FZotf,Msum2,Msum_mask,notchall = _WR.GetMsum(opt,Irtest,Irtestxy,Irtestz,cfai,PGxy,PGz,OTFmask3D_all,chushi=Method_c,flagg=order_flag,Debug=isdebug[5],direction=direction,notch=notch,run_parameter=run_parameter)
                        # _U.save_tif_v21(f'⑤Msum-{order_flag}-{notch}.tif',abs(Msum),run_parameter)
                        # _U.save_tif_v21('⑤Msum_mask.tif',abs(Msum_mask),run_parameter)
                        ######### 得到sk
                        sk = _WR.getsk(opt,ck,cfai,methods=Method_c,Debug=isdebug[6]) 
                        # _U.save_tif_v21('sk.tif',abs(sk),run_parameter)
                        # _U.save_tif_v21('sk-real.tif',abs(sk),run_parameter)

            
                        #两种模式
                        # if run_parameter['decon_meth']=='XYD':        #先移动 再相除
                        #     # bhs,qiezhiname = _WR.geneate_qiezhi(opt,opt.kuodapz,opt.EEdge,qiezhi=1)
                        #     OTFmask3D_all = _DP.otf_mask3D_SIM(opt,opt.nx,opt.ny,opt.nz_re,PGxy,paddingxy=2)

                        #     #整合分子

                        #     #遍历notch参数
                        #     for notch0th in run_parameter['notch0ths']:  #0阶notch参数
                        #         run_parameter['notch0th'] = notch0th   
                        #         for attFWHM0 in run_parameter['attFWHM0s']:
                        #             run_parameter['attFWHM0'] = attFWHM0
                        #             for notch1th_pang in run_parameter['notch1th_pangs']: #1阶旁边
                        #                 run_parameter['notch1th_pang'] = notch1th_pang
                        #                 for attStrFZ in run_parameter['attStrFZs']:#2阶
                        #                     run_parameter['attStrFZ'] = attStrFZ
                        #                     for attFWHMFZ in run_parameter['attFWHMFZs']:
                        #                         run_parameter['attFWHMFZ'] = attFWHMFZ
                        #                         for attStr in run_parameter['attStrs']:#1阶
                        #                             run_parameter['attStr'] = attStr
                        #                             for attFWHM in run_parameter['attFWHMs']:
                        #                                 run_parameter['attFWHM'] = attFWHM

                        #                                 Msum,FZotf,Msum2,Msum_mask,notchall,_ = _WR.GetMsum(opt,Irtest,Irtestxy,Irtestz,cfai,PGxy,PGz,OTFmask3D_all,chushi=Method_c,flagg=order_flag,Debug=isdebug[5],direction=direction,notch=notch,run_parameter=run_parameter)
                        #                                 _U.save_tif_v21(f'⑦Msum-{order_flag}-{notch}.tif',abs(Msum),run_parameter)

                                                        
                        #                                 #开始遍历FZ
                        #                                 FZall = _WR.GetFZall(opt,sk,FZotf,Irtest,Irtestxy,Irtestz,cfai,Msum,PGxy,PGz,OTFmask3D_all,chushi=Method_c,flagg=order_flag,Debug=1,direction=direction,notch=notch,run_parameter=run_parameter)
                        #                                 # _U.save_tif(f'⑦FZall-log{order_flag}-{notch}.tif',_U.visual(FZall))
                        #                                 _U.save_tif_v21(f'⑦FZall-{order_flag}-{notch}.tif',abs(FZall),run_parameter)


                        #                                 for w1 in opt.W:  # 遍历维纳参数
                        #                                     for qiezhii in [4]: #切趾的方式  #切趾的方式 0:无切趾  1:   2:   3:bhs   4 bhs_Z
                                                                
                        #                                         # Apo,qiezhiname = _WR.geneate_qiezhi(opt,opt.kuodapz,opt.EEdge,qiezhi=qiezhii,lenz=run_parameter['Apo_Z'])#lenz 越小越宽
                        #                                         Apo = _WF.bhs_3D_Z(opt,plong,1.5*opt.fc,kuodaz=opt.kuodapz, Zjust=opt.Apoz )
                        #                                         qiezhiname = 'bhs_Z'
                        #                                         run_parameter['qiezhiname'] = qiezhiname 

                        #                                         #######准备FM#####
                        #                                         if run_parameter['re_mode'] == 0:    #Weiner
                        #                                             w3 = w1
                        #                                             FM = 1/(Msum+w3) *Msum_mask
                                                                    
                        #                                             # FM_,gm_1 = FM_fix_error(FM,w3=0.03,size=27)
                        #                                             # FM_[FM_!=0.03]=1
                                        
                        #                                             FM2 = FM#*FM_
                                            
                        #                                             # _U.save_tif_v21(f'FM.tif',abs(FM),run_parameter)
                        #                                             _U.save_tif_v21(f'⑦FM-{order_flag}-{notch}.tif',abs(FM),run_parameter)
                        #                                             # _U.save_tif_v21('FM_（FMmap用来去除异常值）.tif',abs(FM_),run_parameter)
                        #                                             _U.save_tif_v21(f'⑦FM2-{order_flag}-{notch}.tif',abs(FM2),run_parameter)

                        #                                             fiamge = FZall*FM2*(Apo**1)*Msum_mask
                        #                                             finalr,finalk = _WR.Final_ifft(opt,fiamge,Method_c,w3,notch,order_flag,direction,run_parameter=run_parameter,filename=filename,p1_i_=p1_i_)

                        #                                         elif run_parameter['re_mode'] == 2 or run_parameter['re_mode']==3 :
                        #                                             #notchall
                        #                                             # Msum_mask = np.ones_like(Msum_mask);
                        #                                             # Msum_mask[FZall==0]=0;
                        #                                             w1=0.5
                        #                                             w2=0.1
                        #                                             w3 = str(np.round(w1,4))+'_'+str(np.round(w2,4))
                        #                                             run_parameter['w1'] = w1
                        #                                             run_parameter['w2'] = w2
                        #                                             #16-512
                        #                                             Apo1 = imread(r'E:\4TB图片\python 实验数据\多层\2023\验证性试验\7月\7.26 MP-SIM仿真-3D卷积\method=2\Apo1.tif')
                        #                                             Apo2 = imread(r'E:\4TB图片\python 实验数据\多层\2023\验证性试验\7月\7.26 MP-SIM仿真-3D卷积\method=2\Apo2.tif')

                        #                                             # Apo1 = imread(r'E:\4TB图片\python 实验数据\多层\2023\验证性试验\5月\4PS-3DSIM\Apo1.tif')
                        #                                             # Apo2 = imread(r'E:\4TB图片\python 实验数据\多层\2023\验证性试验\5月\4PS-3DSIM\Apo2.tif')

                        #                                             #Apo-1024-21
                        #                                             #Apo1 = imread(r'E:\4TB图片\python 实验数据\多层\2023\验证性试验\8月\8.10 真实3DSIM加MPSIM\6. 3DSIM-2DS\results\2.【matlab-open-3dsim】\Apo1-1024-21.tif')
                        #                                             #Apo2 = imread(r'E:\4TB图片\python 实验数据\多层\2023\验证性试验\8月\8.10 真实3DSIM加MPSIM\6. 3DSIM-2DS\results\2.【matlab-open-3dsim】\Apo2-1024-21.tif')

                        #                                             #Apo-1024-20
                        #                                             # Apo1 = imread(r'E:\4TB图片\python 实验数据\多层\2023\验证性试验\!4P-SSIM\1、0428\Apo1-1024-20.tif')
                        #                                             # Apo2 = imread(r'E:\4TB图片\python 实验数据\多层\2023\验证性试验\!4P-SSIM\1、0428\Apo2-1024-20.tif')

                        #                                             Filter1 =  Apo1/(notchall+w1) #* Msum_mask; 

                        #                                             Filter2 =  Apo2/(notchall+w2) #* Msum_mask; 
                        #                                             fiamge = FZall*Filter1*Filter2;

                        #                                             FZall_r = (_U.ifftn(FZall)).real
                        #                                             FZall_r[FZall_r<0]=0
                        #                                             FZall_r=FZall_r/np.max(FZall_r)

                        #                                             FZall_1_r = (_U.ifftn(FZall*Filter1)).real
                        #                                             FZall_1_r[FZall_1_r<0]=0
                        #                                             FZall_1_r=FZall_1_r/np.max(FZall_1_r)
                        #                                             _U.save_tif_v21(f'【】Filter1-{w1}.tif',Filter1,run_parameter)
                        #                                             _U.save_tif_v21(f'【】Filter2-{w2}.tif',Filter2,run_parameter)
                        #                                             _U.save_tif_v21('【】Filter12-.tif',Filter1*Filter2,run_parameter)
                        #                                             _U.save_tif_v21('【】FZall-r-.tif',FZall_r ,run_parameter) #只做了notch

                        #                                             _U.save_tif_v21('【】FZallxFilter1-r-.tif',FZall_1_r,run_parameter)

                        #                                             _U.save_tif_v21('【】FZall-k.tif',abs(FZall),run_parameter)
                        #                                             finalr,finalk = _WR.Final_ifft(opt,fiamge,Method_c,w3,notch,order_flag,direction,run_parameter=run_parameter,filename=filename)

                        # elif run_parameter['decon_meth']=='XC':   #先除 再移动
                        #     # bhs,qiezhiname = _WR.geneate_qiezhi(opt,opt.kuodapz,opt.EEdge,qiezhi=1)
                        #     OTFmask3D_all = _DP.otf_mask3D_SIM(opt,opt.nx,opt.ny,opt.nz_re,PGxy,paddingxy=2)
                        #     _U.save_tif_v21('OTFmask3D_all.tif',abs(OTFmask3D_all),run_parameter)

                        #     #遍历notch参数
                        #     for notch0th in run_parameter['notch0ths']:  #0阶notch参数
                        #         run_parameter['notch0th'] = notch0th   
                        #         for attFWHM0 in run_parameter['attFWHM0s']:
                        #             run_parameter['attFWHM0'] = attFWHM0
                        #             for notch1th_pang in run_parameter['notch1th_pangs']: #1阶旁边
                        #                 run_parameter['notch1th_pang'] = notch1th_pang
                        #                 for attStrFZ in run_parameter['attStrFZs']:#2阶
                        #                     run_parameter['attStrFZ'] = attStrFZ
                        #                     for attFWHMFZ in run_parameter['attFWHMFZs']:
                        #                         run_parameter['attFWHMFZ'] = attFWHMFZ
                        #                         for attStr in run_parameter['attStrs']:#1阶
                        #                             run_parameter['attStr'] = attStr
                        #                             for attFWHM in run_parameter['attFWHMs']:
                        #                                 run_parameter['attFWHM'] = attFWHM
                                                        
                        #                                 # '【构建分母】'
                        #                                 Msum,FZotf,Msum2,Msum_mask,notchall,Msum_3 = _WR.GetMsum(opt,Irtest,Irtestxy,Irtestz,cfai,PGxy,PGz,OTFmask3D_all,chushi=Method_c,flagg=order_flag,Debug=isdebug[5],direction=direction,notch=notch,run_parameter=run_parameter)
                        #                                 _U.save_tif_v21(f'⑦Msum-{order_flag}-{notch}.tif',abs(Msum),run_parameter)
                        #                                 _U.save_tif_v21('⑤Msum_mask.tif',abs(Msum_mask),run_parameter)

                        #                                 OTFmask3D_all_sum = np.sum(OTFmask3D_all,axis=0)
                        #                                 OTFmask3D_all_sum[OTFmask3D_all_sum!=0]=1
        
                        #                                 _U.save_tif_v21(r'OTFmask3D_all_sum.tif',OTFmask3D_all_sum,run_parameter=run_parameter)
                        #                                 for w1 in opt.W:  # 遍历维纳参数
                        #                                     # for qiezhii in [4]: #切趾的方式  #切趾的方式 0:无切趾  1:   2:   3:bhs   4 bhs_Z
                        #                                     # Apo,qiezhiname = _WR.geneate_qiezhi(opt,opt.kuodapz,opt.EEdge,qiezhi=qiezhii,lenz=run_parameter['Apo_Z'])#lenz 越小越宽
                        #                                     # Zjust =    # Zjust 大 轴向越衰减
                        #                                     # Apo = _WF.bhs_3D_Z(opt,plong,1.5*opt.fc,kuodaz=opt.kuodapz, Zjust=opt.Apoz )
                        #                                     try:        
                        #                                         Apo = opt.Apo
                        #                                         qiezhiname = 'loadApo'
                        #                                         _U.save_tif_v21(f'Apo-{qiezhiname}.tif',abs(Apo),run_parameter)
                        #                                     except:
                        #                                         Apo = _WF.bhs_3D_Z(opt,opt.plongxs*opt.plong,opt.fcxs*opt.fc,kuodaz=opt.kuodapz, Zjust=opt.Apoz )
                        #                                         qiezhiname = 'bhs_3D_Z'
                        #                                         Zjust = opt.Apoz    # Zjust 大 轴向越衰减
                        #                                         _U.save_tif_v21(f'Apo-{Zjust}.tif',abs(Apo),run_parameter)

                        #                                      #对切趾函数插值
                        #                                     Apo = _U.Interpolate(Apo,2*opt.nx,2*opt.ny,opt.nz_re)
                        #                                     run_parameter['qiezhiname'] = qiezhiname 
                                                            
                        #                                     w3 = w1
                                                            
                        #                                     '空域重构'
                        #                                     _,fiamge = _WR.recon_konyu(opt,sk,Msum,Msum_3,Irtest,Irtest_1,flagg=order_flag,run_parameter=run_parameter,w3=w3,notch=notch,direction=direction)
                        #                                     # fiamge = _WR.recon_konyu2(opt,sk,Irtest,flagg=order_flag,notch=notch,run_parameter=run_parameter,w3=w3,direction=direction)

                        #                                     fiamge = fiamge*OTFmask3D_all_sum*Apo   #切趾  以后这个切趾可以放到Final——ifft中，截取了16层之后再加
                        #                                     finalr,finalk = _WR.Final_ifft(opt,fiamge,Method_c,w3,notch,order_flag,direction,run_parameter=run_parameter,filename=filename,p1_i_=p1_i_)

                        if run_parameter['decon_meth'] == '4B':
                            #生成FZotf
                            FZotf = _WR.GetFZotf(opt,Irtest,order_flag=order_flag,run_parameter=run_parameter,direction=direction) #仅仅是生成了各个位置上的OTF，没有notch 没有mask 没有平方
                            
                            #移动分子
                            FZall = _WR.GetFZ(opt,ck,Irtest,order_flag=order_flag,run_parameter=run_parameter,direction=direction)
                            _U.save_tif_v21(r'FZotf.tif',abs(FZotf),run_parameter)
                            _U.save_tif_v21(r'FZall.tif',abs(FZall),run_parameter)
                            
                            #遍历notch参数
                            for notch0th in run_parameter['notch0ths']:  #0阶notch参数
                                run_parameter['notch0th'] = notch0th   
                                for attFWHM0 in run_parameter['attFWHM0s']:
                                    run_parameter['attFWHM0'] = attFWHM0
                                    for notch1th_pang in run_parameter['notch1th_pangs']: #1阶旁边
                                        run_parameter['notch1th_pang'] = notch1th_pang
                                        for attStrFZ in run_parameter['attStrFZs']:#2阶
                                            run_parameter['attStrFZ'] = attStrFZ
                                            for attFWHMFZ in run_parameter['attFWHMFZs']:
                                                run_parameter['attFWHMFZ'] = attFWHMFZ
                                                for attStr in run_parameter['attStrs']:#1阶
                                                    run_parameter['attStr'] = attStr
                                                    for attFWHM in run_parameter['attFWHMs']:
                                                        run_parameter['attFWHM'] = attFWHM
                                                        # if run_parameter['B4_double']==1:
                                                        #     Wlist = opt.W1
                                                        # elif run_parameter['B4_double']==0:
                                                        #     Wlist = opt.W
                                                            
                                                        # for w1 in Wlist:  #遍历wiener
                                                        
                                                        #分方向，第一次filter--notch、mask、cfai组合
                                                        w1=1
                                                        Wiener1 = _WR.GetWiener1(opt,_U.ToNumpy(FZall),_U.ToNumpy(FZotf),PGxy,PGz,cfai,w1,order_flag,run_parameter,notch,direction=[0,1,2])
                                                        _U.save_tif_v21('Wiener1-k.tif',abs(Wiener1),run_parameter)
                                                        _U.save_tif_v21('Wiener1-r.tif',np.real(_U.ifftn(Wiener1)),run_parameter)
                                                        #第二次，组合为整理

                                                        for w2 in opt.W:
                                                            filename2 = filename#+'-'+str(w2)
                                                            Msum_all = _WR.GetWiener2(opt,_U.ToNumpy(Wiener1),_U.ToNumpy(FZotf),PGxy,PGz,cfai,w2,order_flag,run_parameter,notch,direction=[0,1,2])
                                                            
                                                            Apo = _WF.bhs_3D_Z(opt,opt.plongxs*opt.plong,opt.fcxs*opt.fc,kuodaz=opt.kuodapz, Zjust=opt.Apoz )

                                                            if run_parameter['mear']==2:
                                                                Apo = Apo**2
                                                            _U.save_tif_v21(f'Apo-r{opt.Apoz}.tif',Apo,run_parameter)

                                                            Wiener2 = Apo * Wiener1 / Msum_all
                                                            
                                                            finalr,finalk = _WR.Final_ifft(opt,_U.ToNumpy(Wiener2),Method_c,w2,notch,order_flag,direction,run_parameter=run_parameter,filename=filename2,p1_i_=p1_i_)

                        
    #将所有Dr文件夹的文件都遍历完后           
    if run_parameter['just_cfai']: #用于二次配准
        return cfai,PGxy,PGz,plong,same_region_fai,same_region_pxy,Irtest
    
    from datetime import datetime
    
    now = datetime.now().strftime("%H-%M-%S")
    print(now)
    txt_path = _U.save_path(f'run_parameter-{filename}-{now}.txt',run_parameter)
    _U.save_txt(txt_path,run_parameter)
    
    txt_path = _U.save_path(f'PARA-{filename}-{now}.txt',run_parameter)
    PARA = {}
    PARA['PGxy'] = PGxy
    PARA['P_long'] = P_long
    # PARA['PGz'] = PGz
    PARA['cfai'] = cfai    
    _U.save_txt(txt_path,PARA)
    
    return opt,cfai,PGxy,PGz,plong,finalr,finalk,same_region_fai,same_region_pxy#,Msum,Msum_mask




