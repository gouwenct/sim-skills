import numpy as np
import torch 
import pickle   #写入txt字典

import Utils as _U
import WindowF as _WF
import DataProcess as _DP
import Para_Estimate as _PE
from tifffile import imread

from torch.fft import fftshift as tfftshift
from torch.fft import ifftshift as tifftshift
from numpy.fft import fftshift,ifftshift 

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftn = np.fft.fftn
ifftn = np.fft.ifftn

tfft2 = torch.fft.fft2
tifft2 = torch.fft.ifft2
tfftn = torch.fft.fftn
tifftn = torch.fft.ifftn




#产生相位矩阵
def IrTEST(opt,PGxy,PGz,kuodaxy=2,kuodaz=2,Zchushu=1,pz_true=0):
    '''
    统一的移动相位mat
    传入参数估计的PGxy，PGz
    这里的PGz是使用的不同的PGz
    V2.1 增加仅xy z上的移动
    '''
    nz= opt.nz_re        #(opt.nz+2*opt.Zpadding)*kuodaz
    nx=opt.nx*kuodaxy
    ny=opt.ny*kuodaxy
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=kuodaxy,kuodaz=kuodaz) #原点在中心处  -nx/2 ,nx/2
    print(f"IrTestZ范围={ZV[0]}~{ZV[-1]}")
    print(f"IrTestx范围={XV[0][0]}~{XV[-1][0]}")
    Irtest = []
    Irtestxy = []
    Irtestz = []
    for i in range(opt.Nangles):#3个方向
        Pxyabs = PGxy[i]/nx
        Pzabs = PGz[i]/(Zchushu)
        for j in range(opt.Nshifts):#7阶   0  -2  2  (-1,-1)  (-1,1） （1，-1） （1，1）
            Pxy = Pxyabs[j]
            Pz = Pzabs[j]
            shiftmat = np.zeros((nz,nx,ny),dtype=opt.dtype_com)
            if pz_true!=0 and (j==3 or j==5 ):  #真实数据中 3,5为+
                Pz = -pz_true/(Zchushu)
            if pz_true!=0 and (j==4 or j==6 ):  #其他为-
                Pz = pz_true/(Zchushu)
            shiftmat = np.exp(2j*np.pi*(Pxy[0]*XV+Pxy[1]*YV +Pz*ZV) ) #* np.exp(2j*np.pi*Pz*ZV)

            if opt.flag_N==5:
                shiftmatxy = np.exp(2j*np.pi*(Pxy[0]*XV+Pxy[1]*YV) ) 
                shiftmatz = np.exp(2j*np.pi*(Pz*ZV) )
                Irtestxy.append(shiftmatxy)
                Irtestz.append(shiftmatz)
            Irtest.append(shiftmat)
    return Irtest,Irtestxy,Irtestz     #返回为列表

def IrTEST3D_SIM_z(opt,pz_true=0,Zchushu=1,kuodaxy=2,kuodaz=2):
    '''
    
    '''
    nz = opt.nz_re        
    nx =opt.nx*kuodaxy
    ny =opt.ny*kuodaxy
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=kuodaxy,kuodaz=kuodaz) #原点在中心处  -nx/2 ,nx/2
    print(f"IrTestZ范围={ZV[0]}~{ZV[-1]}")
    
    Pz = pz_true/(Zchushu)
    shiftmatz = np.exp(2j*np.pi*(Pz*ZV) )

    return shiftmatz

def IrTEST3D_SIM(opt,PGxy,PGz,kuodaxy=2,kuodaz=2,Zchushu=1,pz_true=0):
    '''
    统一的移动相位mat
    传入参数估计的PGxy，PGz
    这里的PGz是使用的不同的PGz
    V2.1 增加仅xy z上的移动
    '''
    nz= opt.nz_re        #(opt.nz+2*opt.Zpadding)*kuodaz
    nx=opt.nx*kuodaxy
    ny=opt.ny*kuodaxy
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=kuodaxy,kuodaz=kuodaz) #原点在中心处  -nx/2 ,nx/2
    print(f"IrTestZ范围={ZV[0]}~{ZV[-1]}")
    print(f"IrTestx范围={XV[0][0]}~{XV[-1][0]}")
    Irtest = []
    Irtest_1 = []
    for i in range(opt.Nangles):#3个方向
        Pxyabs = PGxy[i]/nx
        for j in range(opt.Nshifts):#5阶   0  -2  2 -1 1
            Pxy = Pxyabs[j]
            shiftmat = np.zeros((nz,nx,ny),dtype=opt.dtype_com)

            shiftmat = np.exp(2j*np.pi*(Pxy[0]*XV+Pxy[1]*YV ) ) #
            Irtest.append(shiftmat)

            shiftmat2 = np.exp(2j*np.pi*(-1*Pxy[0]*XV-1*Pxy[1]*YV ) ) #
            Irtest_1.append(shiftmat2)
    return Irtest,Irtest_1     #返回为列表



def choose_jiesu(opt,flagg):
    if flagg==0:              #0
        # print('MsumFM 仅含0阶')
        jiesu=[0]

    elif flagg==1 or flagg==3:  # ±1 or 0±1              
        if opt.Nshifts==5:       ##slice ±1阶
            # print('slice FM 仅含±1阶 [3,4]')
            jiesu=[3,4]
        elif opt.flag_N == 7 and opt.Nshifts==7:   ##mp ±1
            # print('MP FM 仅含±1阶 [3,4,5,6]')
            jiesu=[3,4,5,6]
        else:
            raise '阶数错误，无1阶'
        if flagg ==3:
            print('0±1')      
            jiesu.insert(0,0)

    elif flagg == 2:            #0 ±2
        print('±2阶')
        jiesu = [1,2]
    elif flagg ==4 :
        print('0±2阶')
        jiesu = [0,1,2]

    elif flagg == 5 or flagg==6:  #±1
        if opt.Nshifts==5:       ##slice ±1阶
            # print('slice FM 仅含±1阶 [3,4]')
            jiesu=[1,2,3,4]
        elif opt.flag_N == 7 and opt.Nshifts==7 :   ##mp ±1
            # print('MP FM 仅含±1阶 [3,4,5,6]')
            jiesu=[1,2,3,4,5,6]

        elif opt.Nshifts==3 and opt.flag_N == 3:
            jiesu = [1,2]

        if flagg ==6:
            print('0±1±2')
            jiesu.insert(0,0)
            
        else:
            print('±1±2')
    return jiesu

def choose_notch(opt,notch,i,j,PGxy,PGz,Astrbase,FWHMbase):
    'HIFI中使用的notch'
    control=1
    #HIFI winer 都用的notch
    if notch==1:       
        if j==0: #0jie  没有施加衰减
            #2D
            if opt.flag_N==3: 
                Astr = 0*Astrbase   
                fwhm = FWHMbase
            #3D
            elif opt.flag_N==5 or opt.flag_N==7:
                Astr = 0*Astrbase
                fwhm = FWHMbase

            
        else:    #其他阶  外面是什么强度就是什么强度
            #2D
            if opt.flag_N==3: 
                Astr = Astrbase/1
                fwhm = FWHMbase
            #3D
            elif opt.flag_N==5 or opt.flag_N==7:
                Astr = Astrbase/1
                # Astr = 0.9/1.05    
                fwhm = FWHMbase
    
    #HIFI特有的notch 第二个
    elif notch==2:      
        if j==0:    
            #2D
            if opt.flag_N==3: 
                Astr = Astrbase/1.05
                fwhm = FWHMbase
            #3D
            elif opt.flag_N==5 or opt.flag_N==7:
                Astr = Astrbase/1
                # Astr = 0.9/1.05
                fwhm = 1 

        else:         #其他阶
            #2D
            if opt.flag_N==3:
                Astr = Astrbase/1.15
                fwhm = FWHMbase
            #3D
            elif opt.flag_N==5 or opt.flag_N==7:  
                # Astr = Astrbase/1
                # fwhm = 1

                # Astr = Astrbase/1.15
                # fwhm = 0.85
                # control = 1.2

                Astr = Astrbase/1
                fwhm = FWHMbase
                control = 1.2                             

    else:   #无notch
        Astr = 0
        fwhm = 1


    Atteg = _WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=Astr,fwhm=fwhm)/control #2D
    if j>2: #±1阶
        Atteg1 = _WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=opt.pz,Astr=Astr,fwhm=fwhm)/control #2D
        Atteg2 = _WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=-opt.pz,Astr=Astr,fwhm=fwhm)/control #2D
        Atteg = Atteg1*Atteg2
    # _U.save_tif(f'Atteg-{i,j}-{np.round(Astr,2),fwhm}.tif',Atteg)
    return Atteg
#分母部分
def GetMsum(opt,Irtest,Irtestxy,Irtestz,cfai,PGxy,PGz,OTFmask3D_all,run_parameter=[],chushi=5,flagg=4,Debug=0,direction=[0,1,2],notch=0):
    '''
    相比于GetMsum 多了GPU
        s=增加一个功能，flagg=0 表示只求0阶
                   flagg=1 表示求其他阶，
                   因为要使用不同维纳参数，所以分开解卷积，再频域相加
    chushi = 4, 用方法2
    chushi = 5，用方法1
    valAttenuation(opt,dist,px,py,Astr=0.9,fwhm=1) 给除了0阶都打上notch
    '''
    nx = opt.nx*opt.kuodaxy
    ny = opt.ny*opt.kuodaxy
    nz = opt.nz_re          
    Nshifts=opt.Nshifts                #5
    nc = opt.nc
    OTFmask_Meth = run_parameter['Msum_maskMove']

    FZotf = torch.zeros((nc,nz,nx,ny),dtype=opt.tdtype_real)     #
    Msum = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)
    Msum2 = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)
    Msum_3= torch.zeros((3,nz,nx,ny),dtype=opt.tdtype_real)   #3个方向分别存储
    OTFnotchall = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)
    # SUMnotch = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)
    MsumMask = torch.zeros((nz,nx,ny),dtype=torch.bool)
    
    "1 OTF转psf"
    OTF3D = _U.ToTensor(opt.OTF[[0,2]]).to(device=opt.device)  #取出0阶和1阶来
    PSF3D_02th = tfftshift(_U.tifftn(OTF3D[0])).real
    PSF3D_1th = tfftshift(_U.tifftn(OTF3D[1])).real
    
    "2 OTFmask转psfmask"
    if OTFmask_Meth ==1:        #"生成OTFmask"    
        pass
    elif OTFmask_Meth ==2:      #"移动式OTFmask"
        pass
        PGxy2 = np.zeros((3,5,2))
        OTFmask3D = _DP.otf_mask3D_SIM_PE(opt,opt.nx,opt.ny,opt.nz_re,PGxy2,paddingxy=2)
        OTFmask3D = _U.ToTensor(OTFmask3D).to(torch.uint8).to(device=opt.device)
        PSF3Dmask_02th = tfftshift(_U.tifftn(OTFmask3D[0])).real
        PSF3Dmask_1th = tfftshift(_U.tifftn(OTFmask3D[2])).real

    '3、0阶1阶notch' #open-3DSIM用
    if run_parameter['re_mode']==3:
        Mask021 = _DP.otf_mask3D_SIM_PE(opt,opt.nx,opt.ny,opt.nz_re,paddingxy=2)

        # _U.save_tif('Mask021.tif',Mask021)
        notch0 = _WF.Notch_fliter(opt,PGxy,0,0,0,Astr= run_parameter['attStr'],fwhm = run_parameter['attFWHM'] )
        # notch0 = notch0/np.max(notch0)
        Mask0 =  Mask021[0]
        OTFtemp = opt.OTF[0]#imread(r'E:\matlab2-code\open-3DSIMv2.2\Open_3DSIM_MATLAB_v2.2\OTFem.tif')# np.copy(OTF3D[0].cpu())  #0阶OTF
        OTFtemp[OTFtemp>0.5]=0.5
        notch0 = notch0*Mask0*OTFtemp
        notch0_ifft = fftshift(_U.ifftn(notch0)).real
        notch0_ifft = _U.ToTensor(notch0_ifft).to(device=opt.device)
        ##
        #
        #轴向移动矩阵
        PGzz,_ = _DP.get_pzp1(opt,PGxy)
        Ir_z = IrTEST3D_SIM_z(opt,pz_true=PGzz[0,3],Zchushu=opt.nz_re,kuodaxy=2,kuodaz=1)
        Ir_z_ = IrTEST3D_SIM_z(opt,pz_true=-PGzz[0,3],Zchushu=opt.nz_re,kuodaxy=2,kuodaz=1)
        tIr_z =  _U.ToTensor(Ir_z).to(device=opt.device)
        tIr_z_ =  _U.ToTensor(Ir_z_).to(device=opt.device)

        _U.save_tif(' CHECK notch0.tif',notch0)
    # _U.save_tif(' CHECK notch1.tif',notch1)
    # sd= sd+1
    
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)

    for i in direction:              #调试dirct 控制方向
        for j in (jiesu):  #调试
            "【移动矩阵】"
            filename = str(i)+str(j)+'.tif'
            tIrtest =_U.ToTensor(Irtest[i*Nshifts+j]).to(device=opt.device)            #CPU-GPU
            # if opt.flag_N==7:
            #     tIrtestxy = _U.ToTensor(Irtestxy[i*opt.Nshifts+j]).to(device=opt.device)   #GPU
            #     tIrtestz = _U.ToTensor(Irtestz[i*opt.Nshifts+j]).to(device=opt.device)   #GPU

            '移动-【mask】'
            if OTFmask_Meth == 1:        #"生成OTFmask"    
                OTFmask3D_move = OTFmask3D_all[i*5+j]                         #CPU
           
            elif OTFmask_Meth==2:       #移动式
                if j <=2: #2阶
                    PSFmask3D = (PSF3Dmask_02th)

                elif j >2 and j<5:  #1th
                    PSFmask3D = (PSF3Dmask_1th)

                OTFmask3D_move =  abs(_U.tfftn( tifftshift(PSFmask3D*tIrtest)) )
                OTFmask3D_move[ (OTFmask3D_move)> 0.1]=1
                OTFmask3D_move[  OTFmask3D_move != 1]=0
                OTFmask3D_move= OTFmask3D_move.to(torch.bool).cpu()           #GPU-CPU

            

            '移动-【OTF】'
            if j<2:
                PSF3D = PSF3D_02th
            elif j >2 and j<5:  #1th
                PSF3D = PSF3D_1th

            OTF_move = abs(_U.tfftn(tifftshift (PSF3D*tIrtest) )).cpu()           #GPU -> CPU
            # OTF_move = _U.normalize(OTF_move) #归一化   分子没做归一化，分母也不做; 1.3改动 分子不敢做，但分母做 
                                                                            #;但从仿真结果来看 不做效果更好(感觉是OTF的问题)
            # _U.save_tif(f'OTF_move-{filename}-.tif',abs(OTF_move).numpy())
            # _U.save_tif(f'OTFmask3D_move-{filename}-.tif',OTFmask3D_move.numpy())
            torch.cuda.empty_cache()
            '【otf_move and mask】------------------【考虑要不要删去 分母上不要mask】'
            FZotf[i*Nshifts+j] = abs(OTF_move)#*(OTFmask3D_move)

            if chushi==4:#启用方法2
                FZotf[i*Nshifts+j] = FZotf[i*Nshifts+j]*cfai[i,j,0]

            '制作【notch】' #3.3修正mode=0 将所有notch应用到FZ上，fm不做notch
            if run_parameter['re_mode']==1 and notch!=0:  #HIFI
                Atteg1 = 1#choose_notch(opt,notch,i,j,PGxy,PGz,run_parameter['attStrFZ'],run_parameter['attFWHMFZ'])    #notch=1
                Atteg2 = 1#choose_notch(opt,notch+1,i,j,PGxy,PGz,run_parameter['attStrFZ'],run_parameter['attFWHMFZ'])  #notch=2

            elif run_parameter['re_mode']==0 and notch!=0 and j <=2: #2阶notch 
                Atteg1 = 1#choose_notch(opt,notch,i,j,PGxy,PGz,run_parameter['attStrFZ'],run_parameter['attFWHMFZ'])    
                Atteg2 = 1

            elif run_parameter['re_mode']==0 and notch!=0 and j >2: # 1阶notch 
                Atteg1 = 1#choose_notch(opt,notch,i,j,PGxy,PGz,run_parameter['attStr'],run_parameter['attFWHM'])  
                Atteg2 = 1

            elif run_parameter['re_mode']==2 and notch!=0:

                #3DSIM 因为1阶缘故多了两个if情况
                if j<=2: #2阶
                    Atteg1 = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,0,i,j))
                    # _U.save_tif(f'Atteg1-{i,j}.tif',Atteg1.numpy())
                    #mask
                    Atteg1 = Atteg1*OTFmask3D_move
                if j>2:  #1阶
                    Atteg1 = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,PGz[i,j],i,j))
                    Atteg1_ = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,-PGz[i,j],i,j))
                    
                    OTFmask3D_move1 = np.copy(OTFmask3D_move)
                    OTFmask3D_move1[:opt.nz_re//2]=0   #保留后半部分
                    OTFmask3D_move1_ = np.copy(OTFmask3D_move)
                    OTFmask3D_move1_[opt.nz_re//2+1:]=0  #保留前半部分
                    
                    #组合
                    Atteg1 = Atteg1*OTFmask3D_move1
                    Atteg1_= Atteg1_*OTFmask3D_move1_
                    # _U.save_tif(f'2Atteg1_-{i,j}.tif',Atteg1_ )  
                    # _U.save_tif(f'2Atteg1-{i,j}.tif',Atteg1 )
                    Atteg1 = 0.5*(Atteg1+Atteg1_)
                    # _U.save_tif(f'Atteg1all-{i,j}.tif',Atteg1 )
                    
                
                # _U.save_tif(f'Atteg1-mask-{i,j}.tif',Atteg1.numpy())
                OTF_move[OTF_move>0.4]=0.4 
                OTF_move_notch = (OTF_move)*Atteg1
                # _U.save_tif(f'OTF_move_notch-{i,j}.tif',OTF_move_notch)
                #OTF_move_notch 

                if j==0:
                    OTFnotchall += abs(OTF_move_notch)/2;
                else:
                    OTFnotchall += abs(OTF_move_notch)
                Atteg2 = 1

            elif run_parameter['re_mode']==3 and notch!=0: #notch、OTF、OTFmask一起移动
                if j == 0 :
                    OTF_move_notch = notch0
                elif j <=2 : #0阶2阶
                    OTF_move_notch = (_U.tfftn(tifftshift (notch0_ifft*tIrtest) )).cpu() 
                
                elif j>2 :#1阶
                    #xy移动
                    # OTF_XYmove_ = 0.5*(_U.fftn(ifftshift (notch0_ifft*Irtest[i*Nshifts+j]) )) 
                    # OTF_XYmove_ifft = fftshift(_U.ifftn(OTF_XYmove_))
                    #z移动
                    # OTF_move_notch = (_U.fftn(ifftshift ( OTF_XYmove_ifft*Ir_z ) )) + (_U.fftn(ifftshift ( OTF_XYmove_ifft*Ir_z_ ) ))

                    OTF_XYmove_ = notch0_ifft*tIrtest
                    OTF_move_notch = 0.5*_U.tfftn(tifftshift(OTF_XYmove_* tIr_z +  OTF_XYmove_* tIr_z_)).cpu() 
                
                OTF_move_notch = _U.ToNumpy(OTF_move_notch)
                OTF_move_notch[OTF_move_notch<1e-9]=0
                # _U.save_tif(f'OTF_move_notch-{i,j}.tif',abs(OTF_move_notch))
                #OTF_move_notch

                if j==0:
                    OTFnotchall += abs(OTF_move_notch)/2;
                    Atteg1 =abs(OTF_move_notch)/2;
                else:
                    OTFnotchall += abs(OTF_move_notch)
                    Atteg1 = abs(OTF_move_notch)
                
                Atteg2 = 1

                # _U.save_tif(f'OTFnotchall-{i,j}.tif',abs(OTFnotchall))
            else:
                Atteg1 = 1
                Atteg2 = 1


            '各阶相加'

            Msum  += abs(FZotf[i*Nshifts+j])**2*Atteg1            #  HIFI中的第一个 /wiener
            Msum2 += abs(FZotf[i*Nshifts+j])**2*Atteg2            #  HIFI中的第二个
            MsumMask = MsumMask | OTFmask3D_move
            # SUMnotch += Atteg1
            Msum_3[i] += abs(FZotf[i*Nshifts+j])**2*Atteg1
            if Debug==1:  #isdebug[5]
                pass

        # pass
    # _U.save_tif_v21('SUMnotch.tif',abs(SUMnotch),run_parameter)        
    return Msum.numpy(),FZotf.numpy(),Msum2.numpy(),MsumMask.numpy(),OTFnotchall.numpy(),Msum_3.numpy()

def GetFZall(opt,sk,FZotf,Irtest,Irtestxy,Irtestz,cfai,Msum,PGxy,PGz,OTFmask3D_all,run_parameter=[],chushi=5,flagg=3,Debug=1,direction=[0,1,2],notch=0):
    #opt,sk,FZotf,Irtest,kuodaxy=2,kuodaz=2,Nshifts=7,flagg=3
    #测试模型 新加了Msum，bhs 两个变量来查看每个分量的空域情况
    #将0阶与其他阶分别处理mask 
    
    '''
    flagg=3 表明所有阶都用到；0 表示仅仅用到0阶；1表示仅仅用高频阶
    OTFmask_cha考虑进来
    yuzhi 1：还是移动0阶 ; 
          2：因为公式生成的可能对不上 可以用公式生成0阶，然后跟着移动
           ：选择用公式生成，
    '''
    if opt.device != 'cpu':
        torch.cuda.empty_cache()

    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Nshifts=opt.Nshifts
    OTFmask_Meth = run_parameter['Msum_maskMove']

    sk = _U.ToTensor(sk)
    FZotf = _U.ToTensor(FZotf)
    FZall =torch.zeros((nz,nx,ny),dtype=torch.complex64)                            #CPU
    # FZall2 =torch.zeros((nz,nx,ny),dtype=torch.complex128)   #只是看看FZ情况
    FZnotchall = torch.zeros((nz,nx,ny),dtype=torch.float32)
    "1、OTF转PSF"
    #FZotf
    
    "2 OTFmask转psfmask"
    if OTFmask_Meth ==1:        #"生成OTFmask"    
        # OTFmask3D_all = _DP.otf_mask3D_SIM(opt,opt.nx,opt.ny,opt.nz_re,PGxy,paddingxy=2)
        # # tempall = np.zeros_like(OTFmask3D_all)
        # for i in range(15):
        #     temp = _PE.Cir_dilate_3D(OTFmask3D_all[i].astype(np.uint8),size=3)
        #     temp = _PE.Cir_Erode_3D(temp,size=3)
        #     OTFmask3D_all[i] = temp.astype(np.bool_)
        pass

    elif OTFmask_Meth ==2:      #"移动式OTFmask"
        PGxy2 = np.zeros((3,5,2))
        OTFmask3D = _DP.otf_mask3D_SIM_PE(opt,opt.nx,opt.ny,opt.nz_re,PGxy2,paddingxy=2)
        OTFmask3D = _U.ToTensor(OTFmask3D).to(torch.uint8).to(device=opt.device)
        PSF3Dmask_02th = tfftshift(_U.tifftn(OTFmask3D[0])).real
        PSF3Dmask_1th = tfftshift(_U.tifftn(OTFmask3D[2])).real

    "3、SK横向补零"
    sr_pad = torch.zeros((nc,nz,nx,ny),dtype=opt.tdtype_com)#.to(device=opt.device)  #CPU
    sk_pad = _U.pad_or_cut(sk,x=nx,y=ny,z=nz)                                        #CPU
    for i in range(nc):
        sr_pad[i] = (_U.tifftn(sk_pad[i].to(device=opt.device)).cpu() )             #GPU->CPU
    del sk_pad 
    sr_pad = sr_pad.to(device=opt.device)                                            #CPU->GPU  
                         
    

    '4、0阶1阶notch' #open-3DSIM用
    if run_parameter['re_mode']==3:
        Mask021 = _DP.otf_mask3D_SIM_PE(opt,opt.nx,opt.ny,opt.nz_re,paddingxy=2)

        # _U.save_tif('Mask021.tif',Mask021)
        notch0 = _WF.Notch_fliter(opt,PGxy,0,0,0,Astr= run_parameter['attStrFZ'],fwhm = run_parameter['attFWHMFZ'] )
        notch0 = notch0/np.max(notch0)
        Mask0 =  Mask021[0]
        OTFtemp = opt.OTF[0]#imread(r'E:\matlab2-code\open-3DSIMv2.2\Open_3DSIM_MATLAB_v2.2\OTFem.tif')# np.copy(OTF3D[0].cpu())  #0阶OTF
        OTFtemp[OTFtemp>0.5]=0.5
        notch0 = notch0*Mask0*(OTFtemp**0)
        notch0 = notch0/np.max(notch0)
        notch0_ifft = fftshift(_U.ifftn(notch0)).real
        notch0_ifft = _U.ToTensor(notch0_ifft).to(device=opt.device)

        ##
        #
        #轴向移动矩阵
        PGzz,_ = _DP.get_pzp1(opt,PGxy)
        Ir_z = IrTEST3D_SIM_z(opt,pz_true=PGzz[0,3],Zchushu=opt.nz_re,kuodaxy=2,kuodaz=1)
        Ir_z_ = IrTEST3D_SIM_z(opt,pz_true=-PGzz[0,3],Zchushu=opt.nz_re,kuodaxy=2,kuodaz=1)
        tIr_z =  _U.ToTensor(Ir_z).to(device=opt.device)
        tIr_z_ =  _U.ToTensor(Ir_z_).to(device=opt.device)

        _U.save_tif('FZnotch0.tif',notch0)
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)
    
    for i in direction:
        for j in jiesu:

            '【移动矩阵】'
            filename = str(i)+str(j)
            tIrtest = _U.ToTensor(Irtest[i*opt.Nshifts+j]).to(device=opt.device)   #GPU
            # if opt.flag_N==7:
            #     tIrtestxy = _U.ToTensor(Irtestxy[i*opt.Nshifts+j]).to(device=opt.device)   #GPU
            #     tIrtestz = _U.ToTensor(Irtestz[i*opt.Nshifts+j]).to(device=opt.device)   #GPU

            '移动-【mask】'
            if OTFmask_Meth == 1:        #"生成OTFmask"    
                OTFmask3D_move = OTFmask3D_all[i*5+j]                         #CPU
           
            elif OTFmask_Meth==2:       #移动式
                if j <=2: #2阶
                    PSFmask3D = (PSF3Dmask_02th)

                elif j >2 and j<5:  #1th
                    PSFmask3D = (PSF3Dmask_1th)

                OTFmask3D_move =  abs(_U.tfftn( tifftshift(PSFmask3D*tIrtest)) )
                OTFmask3D_move[ (OTFmask3D_move)> 0.9]=1
                OTFmask3D_move[  OTFmask3D_move != 1]=0
                OTFmask3D_move= OTFmask3D_move.to(torch.bool).cpu()           #GPU-CPU


            '移动-【sk】'
            sk_shift = (_U.tfftn(sr_pad[i*opt.Nshifts+j]*tIrtest)).cpu()                 #GPU-CPU
            # _U.save_tif_v21(f'sk_shift-{i,j}.tif',abs(sk_shift),run_parameter)
            # FZall2 += sk_shift*OTFmask3D_move
            torch.cuda.empty_cache()

            '制作【notch】'
            if notch == 0:
                Atteg1 = 1  # 所有都不加
    
            elif notch==1 and run_parameter['re_mode']==2 : #open-3DSIM
                if j<=2:
                    Atteg1 = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,0,i,j))
                    # _U.save_tif(f'Atteg1-{i,j}.tif',Atteg1.numpy())
                    #mask
                    Atteg1 = Atteg1*OTFmask3D_move
                if j>2:#
                    Atteg1 = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,PGz[i,j],i,j))
                    Atteg1_ = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,-PGz[i,j],i,j))
                    
                    OTFmask3D_move1 = np.copy(OTFmask3D_move)
                    OTFmask3D_move1[:opt.nz_re//2]=0   #保留后半部分
                    OTFmask3D_move1_ = np.copy(OTFmask3D_move)
                    OTFmask3D_move1_[opt.nz_re//2+1:]=0  #保留前半部分
                    
                    #组合
                    Atteg1 = Atteg1*OTFmask3D_move1
                    Atteg1_= Atteg1_*OTFmask3D_move1_
                    # _U.save_tif(f'2Atteg1_-{i,j}.tif',Atteg1_ )  
                    # _U.save_tif(f'2Atteg1-{i,j}.tif',Atteg1 )
                    Atteg1 = 1*(Atteg1+Atteg1_)

                OTFtemp = FZotf[i*opt.Nshifts+j]
                OTFtemp[OTFtemp>0.3]=0.3

                Atteg1 = abs(Atteg1*OTFmask3D_move*abs(OTFtemp)**0)  #
            
            elif notch==1 and run_parameter['re_mode']==3:
                if j == 0:
                    Atteg1 = notch0
                    
                elif j<=2: #2

                    # Atteg1 = (_U.fftn(ifftshift (notch0_ifft*Irtest[i*Nshifts+j]) )) 

                    Atteg1 = _U.tfftn(tifftshift (notch0_ifft*tIrtest )).cpu() 

                else:  #1

                    # OTF_XYmove_ = (_U.tfftn(tifftshift (notch0_ifft*Irtest[i*Nshifts+j]) )) 
                    # OTF_XYmove_ifft = tfftshift(_U.tifftn(OTF_XYmove_))

                    OTF_XYmove_ = notch0_ifft*tIrtest
                    Atteg1 = 0.5*_U.tfftn(tifftshift(OTF_XYmove_* tIr_z +  OTF_XYmove_* tIr_z_)).cpu() 

                    # Atteg1 = 0.5*( (_U.fftn(ifftshift( OTF_XYmove_ifft*Ir_z ))) + (_U.fftn(ifftshift ( OTF_XYmove_ifft*Ir_z_ ) )))
                Atteg1 = _U.ToNumpy(Atteg1)
                Atteg1 = (Atteg1).real
                Atteg1[Atteg1<1e-9] = 0


            elif notch ==1 and j>0 and j <=2: # 2阶的notch
                Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
            
            elif notch ==1 and j>2:           # 1阶的notch
                Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU

            else:
                Atteg1 = 1

            '对skshift的处理'

            #对0阶的抑制
            if j==0 and run_parameter['notch0th']!= 0: #0阶也加 （注意细胞重构时的控制）
                att = run_parameter['notch0th']
                print(f'注意这里施加了0频抑制-{i,j}-{att}')
                Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) #CPU
            
            #'对sk1阶旁边亮点的抑制'
            if opt.flag_N==5 and j>=3 and run_parameter['notch1th_pang']!=0:
                att = run_parameter['notch1th_pang']
                print(f'注意这里施加了1阶旁边的0频抑制-{i,j}-{att}')
                Atteg0 = _U.ToTensor(_WF.valAttenuation(opt,0,0,pz=PGz[i,j],Astr=run_parameter['notch1th_pang'],fwhm=1))
                sk_shift = (sk_shift*1)*Atteg0
                # _U.save_tif(f'Atteg0-{i,j}.tif',Atteg0)
        

            if run_parameter['re_mode']==2 or run_parameter['re_mode']==3:
                FZ = sk_shift*abs(Atteg1)
                # _U.save_tif(f'Atteg1-{i,j}.tif',abs( _U.ToNumpy(Atteg1) ))
                # _U.save_tif(f'FZ-{i,j}.tif',abs(  _U.ToNumpy(FZ)))
            else:
                '合并【OTF,notch与mask】'

                #腐蚀一圈
                OTFmask3D_move_cir = _U.Cir_Erode_3D(OTFmask3D_move,size=3)

                FzOTF_notch = (FZotf[i*opt.Nshifts+j])*Atteg1*OTFmask3D_move_cir  
                if notch==1 and i==1 and (j==3 or j==2):
                    pass
                    # _U.save_tif_v21(f'Atteg1-{i,j}.tif',Atteg1,run_parameter)
                    # _U.save_tif_v21(f'FzOTF_notch-{i,j}.tif',FzOTF_notch,run_parameter)

                FZ = sk_shift*FzOTF_notch 


            '合并FZall'
            FZall += FZ
            FZnotchall += Atteg1

    notch0th_ = run_parameter['notch0th'] #0
    attFWHM0_ = run_parameter['attFWHM0']
    attStrFZ_ = run_parameter['attStrFZ'] #2
    attFWHMFZ_ = run_parameter['attFWHMFZ']
    attStr_ = run_parameter['attStr']
    attFWHM_ = run_parameter['attFWHM'] #1
    notch1th_pang = run_parameter['notch1th_pang'] #1'

    FZnotchall = FZnotchall[0]
    if notch==0:
        _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang}).tif',abs(FZnotchall),run_parameter)        
    else:
        _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang},{attStrFZ_}_{attFWHMFZ_}-{attStr_}-{attFWHM_}).tif',abs(FZnotchall),run_parameter)        
    return _U.ToNumpy(FZall)#.numpy()




def recon_konyu(opt,sk,Msum,Msum_3,Irtest,Irtest_1,flagg,run_parameter,w3,notch=0,direction=[0,1,2]):
    '''
    空域重建方法
    '将分母逆向移动'
    'sk/分母'    
    'sk->sk_move (这个应该是在空域)'

    '''
    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Nshifts=opt.Nshifts

    #Msum转空域
    Msumt = abs(_U.ToTensor(Msum).to(device=opt.device))  #取出0阶和1阶来
    # print(f'Msumt--{(Msumt)[8,255,256]}')
    # print(f'Msumt--{(Msumt)[8,257,256]}')

    Msumr = tfftshift(_U.tifftn(Msumt)).real    # GPU 需要取real

    # print(f'Msumrt--{_U.tfftn( tifftshift(Msumr))[8,255,256]}')
    # print(f'Msumrt--{_U.tfftn( tifftshift(Msumr))[8,257,256]}')
    # _U.save_tif_v21(r'Msum.tif',Msum,run_parameter)

    del Msumt,Msum

    #Msum_3转空域
    # Msum_3t = _U.ToTensor(Msum_3).to(device=opt.device)  #取出0阶和1阶来
    # Msum_3r = torch.zeros_like(Msum_3t)
    # for i in range(3):
    #     Msum_3r[i] = tfftshift(_U.tifftn(Msum_3t[i])).real
    # del Msum_3t,Msum_3

    #"SK横向补零"
    sk =  _U.ToTensor(sk)
    sk_pad = _U.pad_or_cut(sk,x=nx,y=ny,z=nz)                  #CPU

    #FZOTF #这个OTF应该是没有删去的
    FZOTF02 = torch.clone(_U.ToTensor(opt.OTF[0]) ).to(device=opt.device)            
    FZOTF1  = torch.clone(_U.ToTensor(opt.OTF[2]) ).to(device=opt.device)

    #OTFmask 
    OTFmask02 = torch.clone(_U.ToTensor(opt.OTFmask3D[0]) ).to(device=opt.device)            
    OTFmask1  = torch.clone(_U.ToTensor(opt.OTFmask3D[2]) ).to(device=opt.device) 

    #return : Wiener_sum
    Wiener_sum = torch.zeros((nz,nx,ny),dtype=opt.tdtype_com)  #CPU
    FZnotchall = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)

    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)

    for i in direction:              #调试dirct 控制方向
        for j in jiesu:              #调试
            "【移动矩阵】"
            filename = str(i)+str(j)+'.tif'
            tIrtest =_U.ToTensor(Irtest[i*Nshifts+j]).to(device=opt.device)
            tIrtest_1 =_U.ToTensor(Irtest_1[i*Nshifts+j]).to(device=opt.device)            #CPU-GPU

            '移动分母'
            Msum_r_move = (Msumr*tIrtest_1)
            Msum_move =  abs(_U.tfftn( tifftshift(Msum_r_move)) )                 #GPU
            # Msum_move =  abs(_U.tfftn( tifftshift(Msum_3r[i]*tIrtest_1)) )                 #GPU
            del Msum_r_move,tIrtest_1
            
            'notch'
            if j==0 : #0阶时 判断加不加（注意细胞重构时的控制）
                # att = run_parameter['notch0th']
                # print(f'注意这里施加了0频抑制-{i,j}-{att}')
                Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) #CPU
            # elif j==0 and run_parameter['notch0th']== 0:
            #     att = run_parameter['notch0th']
            #     print(f'注意这里施加了0频抑制-{i,j}-{att}')
            #     Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) #CPU
            elif j!=0:
                if notch == 0:
                    Atteg1 = torch.ones_like(sk_pad[0],dtype=opt.tdtype_real)  # 所有都不加
                elif notch ==1 and j>0 and j <=2: # 2阶的notch
                    Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
                elif notch ==1 and j>2:           # 1阶的notch
                    Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU
                else:
                    Atteg1 = torch.ones_like(sk_pad[0],dtype=opt.tdtype_real)
            '合并fznotch'
            # FZnotchall += Atteg1

            'sk/分母'
            FZ = (Atteg1 * sk_pad[i*Nshifts+j]).to(device=opt.device)
            #大家都使用psf卷积形式
            # FZr  = _U.tifftn(FZ).real
            # if j<3 :
            #     psf = tfftshift(_U.tifftn(FZOTF02)).real
            # elif j>=3:
            #     psf = tfftshift(_U.tifftn(FZOTF1)).real

            # FZr_ = _U.T_conv3D(FZr,psf)
            # FZ_  = _U.tfftn(FZr_)
            # Wiener_sk =  FZ_ / (Msum_move + w3)

            #频域处理
            if j <3 :                          # 02阶
                FZ_ = (FZ*FZOTF02*OTFmask02)
                Wiener_sk =  FZ_ / (Msum_move + w3)     
                del FZ,FZ_
            elif j>=3:   
                FZ_ = (FZ*FZOTF1*OTFmask1)       #1阶
                Wiener_sk =  FZ_ / (Msum_move + w3)
                del FZ,FZ_

            '移动回来Wiener_sk>Wiener_sk_move'
            Wiener_sr = _U.tifftn(Wiener_sk)
            Wiener_sr_move = ((Wiener_sr)*tIrtest)#.real
            
            del Wiener_sr,Wiener_sk,Msum_move,tIrtest
            # Wiener_sr_move[Wiener_sr_move<0]=0
            Wiener_sum += Wiener_sr_move.cpu().real   #4b这里是加了real 的,因为它是对称的在移动，所以加real，
                                                      #这里我是单边移动，所以没加real，那到底应该不应该加呢
                                                      #

    Wienek_sum = _U.tfftn(Wiener_sum)
    Wiener_sum = Wiener_sum
    # Wiener_sum[Wiener_sum<0]=0
    _U.save_tif_v21(f'WieneR_sum-.tif',Wiener_sum,run_parameter)
    _U.save_tif_v21(f'WieneK_sum-.tif',abs(Wienek_sum),run_parameter)

    # notch0th_ = run_parameter['notch0th'] #0
    # attFWHM0_ = run_parameter['attFWHM0']
    # attStrFZ_ = run_parameter['attStrFZ'] #2
    # attFWHMFZ_ = run_parameter['attFWHMFZ']
    # attStr_ = run_parameter['attStr']
    # attFWHM_ = run_parameter['attFWHM'] #1
    # notch1th_pang = run_parameter['notch1th_pang'] #1'

    # if notch==0:
    #     _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang}).tif',abs(FZnotchall[0]),run_parameter)        
    # else:
    #     _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang},{attStrFZ_}_{attFWHMFZ_}-{attStr_}-{attFWHM_}).tif',abs(FZnotchall[0]),run_parameter) 
    
    return _U.ToNumpy(Wiener_sum),_U.ToNumpy(Wienek_sum)




def recon_konyu2(opt,sk,Irtest,flagg,notch,run_parameter,w3,direction=[0,1,2]):
    '''
    空域重建方法
    'SK*OTF*OTFmaks / (OTF^2+w3)'
    '制作一个上面的wiener-mask'  这个组合的结果 可能需要改进 见01-11
    '移动wiener-sk、wiener-mask'
    '空域组合'                                  ‘空域组合 还是FFT到频域再组合？’
    'FFT到频域，再相除wiener-mask，乘上切趾Apo'   
    '返回频谱'

    '''
    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Nshifts=opt.Nshifts

    # #Msum转空域
    # Msumt = _U.ToTensor(Msum).to(device=opt.device)  #取出0阶和1阶来
    # Msumr = tfftshift(_U.tifftn(Msumt)).real
    # del Msumt,Msum

    # #Msum_3转空域
    # Msum_3t = _U.ToTensor(Msum_3).to(device=opt.device)  #取出0阶和1阶来
    # Msum_3r = torch.zeros_like(Msum_3t)
    # for i in range(3):
    #     Msum_3r[i] = tfftshift(_U.tifftn(Msum_3t[i])).real
    # del Msum_3t,Msum_3
    # Apo,qiezhiname = geneate_qiezhi(opt,opt.kuodapz,10,qiezhi=4,lenz=run_parameter['Apo_Z'])#lenz 越小越宽
    Apo = _WF.bhs_3D_Z(opt,opt.plongxs*opt.plong,opt.fcxs*opt.fc,kuodaz=opt.kuodapz, Zjust=opt.Apoz )

    Apo = _U.ToTensor(Apo).to(device=opt.device)

    #"SK横向补零"
    sk =  _U.ToTensor(sk)
    sk_pad = _U.pad_or_cut(sk,x=nx,y=ny,z=nz)                  #CPU

    #FZOTF #这个OTF应该是没有删去的
    FZOTF02 = torch.clone(_U.ToTensor(opt.OTF[0]) ).to(device=opt.device)            
    FZOTF1  = torch.clone(_U.ToTensor(opt.OTF[2]) ).to(device=opt.device)

    #OTFmask 
    OTFmask02 = torch.clone(_U.ToTensor(opt.OTFmask3D[0]) ).to(device=opt.device)            
    OTFmask1  = torch.clone(_U.ToTensor(opt.OTFmask3D[2]) ).to(device=opt.device) 
    OTFmaskr02 = tfftshift(_U.tifftn(OTFmask02*Apo**1)).real
    OTFmaskr1  = tfftshift(_U.tifftn(OTFmask1*Apo**1)).real

    #return : Wiener_sum
    Wiener_sum = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)  #CPU
    OTFmaskr_sum = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)

    Wienek_sum = torch.zeros((nz,nx,ny),dtype=opt.tdtype_com)  #CPU
    OTFmaskk_sum = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)


    
    'notch去噪'
    #对2阶 notch去噪
    if notch == 0:
        Atteg1 = 1  # 所有都不加
        Atteg2 = 1
    elif notch==1 and run_parameter['re_mode']==3:
        if j == 0:
            Atteg1 = notch0
            
        elif j<=2: #2

            # Atteg1 = (_U.fftn(ifftshift (notch0_ifft*Irtest[i*Nshifts+j]) )) 

            Atteg1 = _U.tfftn(tifftshift (notch0_ifft*tIrtest )).cpu() 

        else:  #1

            # OTF_XYmove_ = (_U.tfftn(tifftshift (notch0_ifft*Irtest[i*Nshifts+j]) )) 
            # OTF_XYmove_ifft = tfftshift(_U.tifftn(OTF_XYmove_))

            OTF_XYmove_ = notch0_ifft*tIrtest
            Atteg1 = 0.5*_U.tfftn(tifftshift(OTF_XYmove_* tIr_z +  OTF_XYmove_* tIr_z_)).cpu() 

            # Atteg1 = 0.5*( (_U.fftn(ifftshift( OTF_XYmove_ifft*Ir_z ))) + (_U.fftn(ifftshift ( OTF_XYmove_ifft*Ir_z_ ) )))
        Atteg1 = _U.ToNumpy(Atteg1)
        Atteg1 = (Atteg1).real
        Atteg1[Atteg1<1e-9] = 0

    elif notch ==1 : # 2阶的notch
        Atteg2 = _U.ToTensor(_WF.valAttenuation(opt,0,0,pz=0,Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
        # 1阶的notch
        Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,0,0,pz=0,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU

    else:
        Atteg1 = 1  # 所有都不加
        Atteg2 = 1

    #对0阶的抑制
    if run_parameter['notch0th']!= 0: #0阶也加 （注意细胞重构时的控制）
        att = run_parameter['notch0th']
        print(f'注意这里施加了0频抑制')
        Atteg0 = _U.ToTensor(_WF.valAttenuation(opt,0,0,pz=0,Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) #CPU
    else:
        Atteg0 = np.zeros((2*opt.nx,2*opt.ny))
    #'对sk1阶旁边亮点的抑制'
    if run_parameter['notch1th_pang']!=0:
        att = run_parameter['notch1th_pang']
        print(f'注意这里施加了1阶旁边的0频抑制')
        Atteg1_ = _U.ToTensor(_WF.valAttenuation(opt,0,0,pz=0,Astr=run_parameter['notch1th_pang'],fwhm=1))

    else:
        Atteg1_ = 1

    # notch_filter_all= (Atteg0+Atteg2+Atteg1+Atteg1_)-4
    # # notch_filter_all = notch_filter_all[0]
    # _U.save_tif_v21(r'notch_filter_all.tif',notch_filter_all,run_parameter)

    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)
    method=1
    for i in direction:              #调试dirct 控制方向
        for j in jiesu:              #调试
            "【移动矩阵】"
            filename = str(i)+str(j)+'.tif'
            tIrtest =_U.ToTensor(Irtest[i*Nshifts+j]).to(device=opt.device)
            # tIrtest_1 =_U.ToTensor(Irtest_1[i*Nshifts+j]).to(device=opt.device)            #CPU-GPU

            # '移动分母'
            # Msum_move =  abs(_U.tfftn( tifftshift(Msumr*tIrtest_1)) )                 #GPU
            # Msum_move =  abs(_U.tfftn( tifftshift(Msum_3r[i]*tIrtest_1)) )                 #GPU


            'sk去除OTF影响'
            if j == 0:
                notch_filter = Atteg0
            elif j<3:
                notch_filter = Atteg2
            elif j>=3:
                notch_filter = Atteg1

            FZ = (sk_pad[i*Nshifts+j]*notch_filter).to(device=opt.device)

            if j <3 :               # 02阶
                Wiener_sk = (FZ*torch.conj(FZOTF02)* OTFmask02) / (FZOTF02**2 + w3) 
                OTFmaskr_move = OTFmaskr02*tIrtest
            elif j>=3:              #1阶
                Wiener_sk = (FZ*torch.conj(FZOTF1) * OTFmask1) / (FZOTF1**2 + w3)  
                OTFmaskr_move = OTFmaskr1 *tIrtest
            del FZ


            '移动z正确位置Wiener_sk>Wiener_sk_move'
            Wiener_sr = _U.tifftn(Wiener_sk)
            Wiener_sr_move = Wiener_sr*tIrtest



            '空域组合 还是 频域组合'
            #空域组合 直接相加
            if method==1:
                Wiener_sum += Wiener_sr_move.cpu().real
                OTFmaskr_sum += OTFmaskr_move.cpu().real


            #频域组合 FFT到频域相加
            elif method==2:
                fre_band =  _U.tfftn(Wiener_sr_move).cpu()#*Atteg1_
                Wienek_sum+= fre_band

                temp = abs(_U.tfftn( tifftshift(OTFmaskr_move) ) ).cpu()
                OTFmaskk_sum+= temp

            #保存临时文件
            if i ==0:
                pass
                # _U.save_tif_v21(f'Wiener_sk-{i,j}.tif',abs(Wiener_sk.cpu()),run_parameter)

                # _U.save_tif_v21(f'temp-{i,j}.tif',temp,run_parameter)

    if method==1:
        Wienek_sum = _U.tfftn(Wiener_sum.to(device=opt.device)).cpu()
        OTFmaskk_sum = abs(_U.tfftn( tifftshift(OTFmaskr_sum.to(device=opt.device))) ).cpu()
    
    '除OTFmaskk_sum'
    # OTFmaskk_sum_mask = torch.clone(OTFmaskk_sum)
    # OTFmaskk_sum_mask[OTFmaskk_sum_mask!=0] = 1

    # OTFmaskk_sum_FM = 1/(OTFmaskk_sum)
    # OTFmaskk_sum_FM = 1/(OTFmaskk_sum+0.05)
    OTFmaskk_sum_FM = 1/(OTFmaskk_sum+0.1)

    Wienek_fine = Wienek_sum * OTFmaskk_sum_FM


    # Wiener_sum = Wiener_sum
    # Wiener_sum[Wiener_sum<0]=0

    Wienerr_fine = _U.tifftn(Wienek_sum).real
    Wienerr_fine[Wienerr_fine<0]=0

    _U.save_tif_v21(f'Wienerr_fine-.tif',abs((Wienerr_fine)),run_parameter)
    _U.save_tif_v21(f'OTFmaskk_sum_FM-.tif',abs((OTFmaskk_sum_FM)),run_parameter)

    _U.save_tif_v21(f'OTFmaskk_sum-.tif',abs((OTFmaskk_sum)),run_parameter)
    _U.save_tif_v21(f'Wienek_sum-.tif',abs((Wienek_sum)),run_parameter)

    return _U.ToNumpy(Wienek_fine)


#sk修正 c fai 影响
def getsk(opt,ck,cfai,methods=0,Debug=0):
    #修正C 、fai系数
    sk = np.zeros_like(ck)
    for i in range(opt.Nangles):
        for j in range(opt.Nshifts): #7,5,3

            if j == 0:
                assert cfai[i,j,0]==1 and cfai[i,j,1]==0 

            if methods == 4:  #方法2
                sk[i*opt.Nshifts+j] = ck[i*opt.Nshifts+j]/( np.exp(1j* cfai[i,j,1] ))
            elif methods == 5:  #方法1
                sk[i*opt.Nshifts+j] = ck[i*opt.Nshifts+j]/( cfai[i,j,0] * np.exp(1j* cfai[i,j,1] ))

    if Debug == 1:
        for i in range(7):
            _U.save_tif(f'⑥sk—all方向-{i}.tif',abs(sk[i]))
    return sk



# 产生Apo
# def geneate_qiezhi(opt,kuodapz,plong,qiezhi=4,lenz=2):
#     # 0 
#     if qiezhi==0:  #无切趾
#         Apo = 1   #FZall*FM
#         qiezhiname='non'
#     if qiezhi ==1: #1apod  突然截断的切趾
#         Apo = _WF.apod(opt)
#         # _U.save_tif("⑧ apod.tif",Apo)
#         qiezhiname='apod'
#     elif qiezhi ==2:   #2高斯切趾
#         Apo = _WF.build_gaussian_3D(opt,0,1,kuodaxy=2,kuodaz=opt.kuodapz)
#         # _U.save_tif('⑧ gaosi.tif',Apo)
#         qiezhiname = 'gaosi'
#     elif qiezhi ==3:   #3bhs无轴向
#         Apo = _WF.bhs_3D(opt,plong,opt.fc,kuodaz=kuodapz)
#         # _U.save_tif("⑧ bhs.tif",Apo)
#         qiezhiname = 'bhs'

#     elif qiezhi ==4:   #4 bhs有轴向
#         Apo = _WF.bhs_3D_Z(opt,plong,opt.fc,kuodaz=kuodapz,lenz=lenz)
#         # _U.save_tif("⑧ bhs_Z.tif",Apo)
#         qiezhiname = 'bhs_Z'
#     return Apo,qiezhiname

def Just_piexlsize(data_k,dx,nx):
    #将3D的piexlsize 变为MP对应的piexlsize
    dx_3D = dx//2   #重构后dx由72 变为 36
    dx_MP = 30          #一般为30
    rate = dx_MP/dx_3D
    
    nx_recon = 2*nx   #重构后由512变为1024
    nx_just = int(nx_recon/rate)     #调整1024 为1228.8
    
    #对data_K 的xy进行调整
    assert len(data_k.shape)==3
    nz,_,_ = data_k.shape
    # print("just:")
    # print(data_k.shape)
    # print(rate,nx_just)
    data_k_just = _U.pad_or_cut(data_k,x=nx_just,y=nx_just,z=nz)
    return data_k_just           #返回xy补零的频域 = 空域插值

#文件夹命名
def Final_ifft(opt,fimage_k,chushi,w1,notch,flagg,direction,run_parameter,filename,p1_i_):
    """
    传入频域，逆变换
    """

    if chushi ==4:
        strt = '法2'
    elif chushi==5:
        strt = '法1'
    
    #当前阶数
    if flagg == 3:
        strt2 = '0±1th'
    elif flagg == 0:
        strt2 = '(0)th'
    elif flagg == 1:
        strt2 = '(±1)th'
    elif flagg == 2:
        strt2 = '(±2)th'
    
    elif flagg == 4:
        strt2 = '(0±2)th'
    elif flagg == 5:
        strt2 = '(±1±2)th'
    
    elif flagg == 6:
        strt2 = ''
    #当前方向
    if direction ==[0,1,2]:
        dire=''
    elif direction==[0]:
        dire='dir1'
    elif direction==[1]:
        dire='dir2'
    elif direction==[2]:
        dire='dir3'
    

    #控制小数点位数
    if isinstance(w1, str):
        w2 = w1
    elif isinstance(w1, float) and w1 >= 1e-3:     
        w2 = (np.round(w1,5))
    elif isinstance(w1, float):
        # w2 = (np.round(w1,20))
        w2 = f"{w1:.0e}"
    else:
        w2=w1

    #re_mode
    if run_parameter['re_mode']==0:
        re_mode='Wien'
    elif run_parameter['re_mode']==1:
        re_mode='HIFI'
    elif run_parameter['re_mode']==2:
        re_mode='open-3DSIM'
    elif run_parameter['re_mode']==3:
        re_mode='open-3DSIM2'
    else :
        re_mode = 'other algorithms'
        
    decon_meth = run_parameter['decon_meth']
    #notch
    notch0th = _U.fill0(run_parameter['notch0th'])
    attFWHM0 = _U.fill0( run_parameter['attFWHM0'])
    notch1th_pang = _U.fill0(run_parameter['notch1th_pang'])
    
    Apo_Z = opt.Apoz#run_parameter['Apo_Z']  #记录Apo_Z宽度
    Apo_XY = opt.fcxs
    ##############################文件处理##########################################
    
    #空域处理
    temp_r = _U.ifftn((fimage_k)) #).real.astype(opt.dtype_real)
    
    #在空域中裁掉重复的部分
    if opt.Zpadding!=0:
        temp_r_cai = temp_r[1*opt.Zpadding:-1*opt.Zpadding]
    else:
        temp_r_cai = temp_r

    #取real or abs
    temp_abs_cai = abs(np.copy(temp_r_cai).astype(opt.dtype_real))   #看看abs的效果
    temp_r_cai = temp_r_cai.real.astype(opt.dtype_real) #real

    #截断
    # temp_abs_cai_j[temp_rabs_cai<0] = 0   #截断
    temp_r_cai_j = np.copy(temp_r_cai)
    temp_r_cai_j[temp_r_cai_j<0] = 0   #截断

    #归一化_n
    # temp_abs_cai_j_n = temp_abs_cai_j/np.max(temp_abs_cai_j)
    temp_r_cai_j_n = np.copy(temp_r_cai_j)
    temp_r_cai_j_n = temp_r_cai_j_n/np.max(temp_r_cai_j_n)
    ##################################################################
    ##################################################################
    #调整piexlsize大小与MP相同
    #空域上下采样 = 频域插值
    fimage_k_just = Just_piexlsize(fimage_k,opt.dx,opt.nx)
    temp_r_just = _U.ifftn((fimage_k_just))
    
    #在空域中裁掉重复的部分
    if opt.Zpadding!=0:
        temp_r_just_cai = temp_r_just[1*opt.Zpadding:-1*opt.Zpadding]
    else:
        temp_r_just_cai = temp_r_just
    
    #real
    temp_r_just_cai = temp_r_just_cai.real.astype(opt.dtype_real) #real
    
    #截断
    temp_r_just_cai_j = np.copy(temp_r_just_cai)
    temp_r_just_cai_j[temp_r_just_cai_j<0] = 0   #截断
    
    #归一化
    temp_r_just_cai_j_n = np.copy(temp_r_just_cai_j)
    # temp_r_just_cai_j_n = temp_r_just_cai_j_n/np.max(temp_r_just_cai_j_n)   #2D/MF 不要归一化
    ##################################################################
    
    #裁剪跟MP一样大小-不用
    # temp_r_just_cai_j_n_860 = _U.pad_or_cut(temp_r_just_cai_j_n,x=860,y=860,z=opt.nz_re)
    
    #FFT
    fimage_k_cai = _U.fftn(temp_r_cai)

    version = opt.version
    data_name = run_parameter['data_name']
    dirname = run_parameter['dir_name']#[0:3]
    
    #保存空域
    if notch==0: #没有12阶notch 但肯可能有0阶notch，1阶旁

        filenamerreal_cai = f'{dirname}_{version}-3D-{decon_meth}-W={w2}+({notch}-{notch0th}_{attFWHM0}-{notch1th_pang},)-{dire}_{strt2}-Apo_Z={Apo_Z}-{re_mode}-{filename}.tif'
        
        filenamerreal_cai2 = f'{data_name}-{dirname}-W{w2}+({notch0th})-AZ{Apo_Z}-{Apo_XY}-{version}.tif' 
    
    else:#notch参数
        attStrFZ = _U.fill0(run_parameter['attStrFZ'])  #2
        attFWHMFZ =  _U.fill0(run_parameter['attFWHMFZ'])

        attStr = _U.fill0(run_parameter['attStr'])      #1
        attFWHM =  _U.fill0(run_parameter['attFWHM'])

        filenamerreal_cai = f'{dirname}_{version}-3D-{decon_meth}-W={w2}+({notch}-{notch0th}_{attFWHM0}-{notch1th_pang},-{attStrFZ}_{attFWHMFZ}-{attStr}_{attFWHM})-{dire}_{strt2}-ApoZ={Apo_Z}-{re_mode}-{filename}.tif'    
        
        filenamerreal_cai2 = f'{data_name}-{dirname}-W{w2}+({notch0th})-AZ{Apo_Z}-{Apo_XY}-{version}.tif' 


    filenamerreal_cai_j = filenamerreal_cai[:-4]+''+'.tif'
    # filenamerreal_cai_j_n = filenamerreal_cai_j[:-4]+''+'.tif'
    filenamerreal_just_cai_j_n = filenamerreal_cai_j[:-4]+''+'.tif'
    filenamerreal_just_cai_j_n_860 = filenamerreal_cai_j[:-4]+'-piexl归一化'+'-860'+'.tif'

    filenamek = '_Klog-'+filenamerreal_cai[6:]
    filenamek2_cai = '_Kcai-'+filenamerreal_cai[6:]
    filenamek2 = '_KV-'+filenamerreal_cai[6:]

    Dr_sub = run_parameter['Dr_sub']
    # _U.save_tif_v21(filenamerreal_cai,temp_r_cai,run_parameter,sub_filename='ORI')
    # _U.save_tif_v21(filenamerreal_cai_j,temp_r_cai_j,run_parameter,sub_filename='j')
    # _U.save_tif_v21(filenamerreal_cai_j_n,temp_r_cai_j_n,run_parameter,sub_filename='n-'+ Dr_sub)
    
    
    
    if run_parameter['mear']==1:
        _U.save_tif_v21('留负'+filenamerreal_cai_j,temp_r_just_cai,run_parameter,sub_filename='n-just-'+ Dr_sub,subsub_filename=f"留负{data_name}"+"_"+f"{dirname}"+"_W="+f"{w2}"+f"NF0=({notch0th})_t")
        _U.save_tif_v22(f'留负-{filename}.tif',temp_r_just_cai,run_parameter,sub_filename='just-'+ Dr_sub,subsub_filename="留负"+filenamerreal_cai2[:-4])
    else:
        _U.save_tif_v21(filenamerreal_cai_j,temp_r_just_cai_j,run_parameter,sub_filename='n-just-'+ Dr_sub,subsub_filename=f"{data_name}"+"_"+f"{dirname}"+"_W="+f"{w2}"+f"NF0=({notch0th})_t")
        _U.save_tif_v22(f'{filename}.tif',temp_r_just_cai_j,run_parameter,sub_filename='just-'+ Dr_sub,subsub_filename=filenamerreal_cai2[:-4])
    
    # _U.save_tif_v22(filenamerreal_cai_j,temp_r_just_cai_j,run_parameter,sub_filename='just-'+ Dr_sub,subsub_filename=f"{data_name}"+"_"+f"{dirname}"+"_W="+f"{w2}"+f"NF0=({notch0th})_t")
    # _U.save_tif_v21(filenamerreal_just_cai_j_n_860,temp_r_just_cai_j_n_860,run_parameter,sub_filename='n-just-860'+ Dr_sub)
    
    
    

    #保存频域
    if p1_i_!=-1 or p1_i_==4:
        if opt.Zpadding!=0:
            # _U.save_tif(filenamek,_U.visual(fimage_k_cai))  
            # _U.save_tif_v21(filenamek2_cai,abs(fimage_k_cai),run_parameter)  
            _U.save_tif_v21(filenamek2,abs(fimage_k).astype(np.float32),run_parameter,sub_filename='K-'+ Dr_sub,subsub_filename=f'{w2}+{notch0th}')
            _U.save_tif_v22(filenamek2,abs(fimage_k).astype(np.float32),run_parameter,sub_filename='K-'+ Dr_sub,subsub_filename=f'{w2}+{notch0th}')

        else:
            # _U.save_tif(filenamek,_U.visual(fimage_k))  
            _U.save_tif_v21(filenamek2,abs(fimage_k).astype(np.float32),run_parameter,sub_filename='K-'+ Dr_sub,subsub_filename=f'{w2}+{notch0th}')   
            _U.save_tif_v22(filenamek2,abs(fimage_k).astype(np.float32),run_parameter,sub_filename='K-'+ Dr_sub,subsub_filename=f'{w2}+{notch0th}')   

    #写入字典
    # _U.save_txt(f"{filenamerreal_cai[:-4]}.txt",run_parameter)
    return temp_r_cai,fimage_k_cai





##############2023-12-22-XYD-4方法#############
def GetFZotf(opt,Irtest,order_flag,run_parameter,direction=[0,1,2]):
    """
    读取opt.OTF,根据Irtest移动OTF
    """
    
    nx = opt.nx*opt.kuodaxy
    ny = opt.ny*opt.kuodaxy
    nz = opt.nz_re          
    Nshifts=opt.Nshifts                #5
    nc = opt.nc
    # OTFmask_Meth = run_parameter['Msum_maskMove']

    FZotf = torch.zeros((nc,nz,nx,ny),dtype=opt.tdtype_real)     #
    # OTFnotchall = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)

    "1 OTF转psf"
    if run_parameter['mear']==1:
        pass
    else:
        opt.OTF[0] = opt.OTF[0] * opt.OTFmask3D[0]
        opt.OTF[1] = opt.OTF[1] * opt.OTFmask3D[1]
        opt.OTF[2] = opt.OTF[2] * opt.OTFmask3D[2]
    
    
    
    OTF3D = _U.ToTensor(opt.OTF[[0,2]]).to(device=opt.device)  #取出0阶和1阶来
    PSF3D_02th = tfftshift(_U.tifftn(OTF3D[0])).real
    PSF3D_1th = tfftshift(_U.tifftn(OTF3D[1])).real
    
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,order_flag)

    for i in direction:              #调试dirct 控制方向
        for j in (jiesu):  #调试
            if j == 0:
                FZotf[i*Nshifts+j] = OTF3D[0]#opt.OTF[0]
            else:
                "【移动矩阵】"
                filename = str(i)+str(j)+'.tif'
                tIrtest =_U.ToTensor(Irtest[i*Nshifts+j]).to(device=opt.device)            #CPU-GPU

                
                '移动-【OTF】'
                if j<=2:  # 0 1 2 
                    PSF3D = PSF3D_02th
                elif j >2 :  #3 4   1th
                    PSF3D = PSF3D_1th

                OTF_move = abs(_U.tfftn(tifftshift (PSF3D*tIrtest) )).cpu()           #GPU -> CPU
                OTF_move = _U.normalize(OTF_move) #归一化   分子没做归一化，分母也不做; 1.3改动 分子不敢做，但分母做 
                                                                                #;但从仿真结果来看 不做效果更好(感觉是OTF的问题)
                # _U.save_tif(f'OTF_move-{filename}-.tif',abs(OTF_move).numpy())
                # _U.save_tif(f'OTFmask3D_move-{filename}-.tif',OTFmask3D_move.numpy())            
                'FZotf'
                FZotf[i*Nshifts+j] = abs(OTF_move)
            
    return FZotf


def GetFZ(opt,ck,Irtest,order_flag,run_parameter,direction=[0,1,2]):


    if opt.device != 'cpu':
        torch.cuda.empty_cache()

    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Nshifts=opt.Nshifts


    ck = _U.ToTensor(ck)
    # FZotf = _U.ToTensor(FZotf)
    FZall =torch.zeros((nc,nz,nx,ny),dtype=torch.complex64)                            #CPU


    "CK横向补零"
    # cr_pad = torch.zeros((nc,nz,nx,ny),dtype=opt.tdtype_com)                         #
    # ck_pad = _U.pad_or_cut(ck,x=nx,y=ny,z=nz)                                        #padding
                         
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        for j in jiesu:
            if j==0:
              ck_pad = _U.pad_or_cut(ck[i*opt.Nshifts+j],x=nx,y=ny,z=nz)   
              FZall[i*opt.Nshifts+j] = ck_pad
            else:
                '【更大】'
                ck_pad = _U.pad_or_cut(ck[i*opt.Nshifts+j],x=nx,y=ny,z=nz)
                
                '【移动矩阵】'
                filename = str(i)+str(j)
                tIrtest = _U.ToTensor(Irtest[i*opt.Nshifts+j]).to(device=opt.device)   #GPU

                '移动-【ck_pad】'
                ck_pad_GPU = ck_pad.to(device=opt.device)                     #CPU->GPU
                ck_pad_move = _U.tfftn(_U.tifftn(ck_pad_GPU)*tIrtest).cpu()                    #IFFT-移动-FFT  、 GPU->CPU

                FZall[i*opt.Nshifts+j] = ck_pad_move
    
    return FZall


def GetWiener1(opt,FZall,FZotf,PGxy,PGz,cfai,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    "按照方向相加,mask、notch"
    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Nshifts=opt.Nshifts
    
    out_K_all =  np.zeros((nz,nx,ny),dtype=np.complex64)
    
    jiesu = choose_jiesu(opt,order_flag)
    for i in [1,0,2]:
        FZ_dir = np.zeros((nz,nx,ny),dtype=np.complex64)
        # Msum_dir = w1
        for j in jiesu:
            '制作【notch】/【mask】'
            
            if j ==0:
                Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) 
            elif j!=0:
                if notch == 0:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  # 不加
                elif notch ==1 and j>0 and j <=2: # 2阶的notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
                elif notch ==1 and j>2:           # 1阶的notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU
                else:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  # 不加
                
            '【cfai】'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            
            # Atteg1 = imread(r'E:\matlab2-code\SIMreconProject-main\Sirecon\dampen_matrix_34-(1_0).tif')
            '【组合某方向】'
            scale = Atteg1*FZotf[i*Nshifts+j]*np.conj(amp)
            temp_FZ = FZall[i*Nshifts+j]*scale
            FZ_dir  += temp_FZ
            # Msum_dir += (FZotf[i*Nshifts+j]**2) * (abs(amp)**2) * Atteg1
            
        _U.save_tif(r'FZ_dir.tif',abs(FZ_dir))
        _U.save_tif(r'Atteg1.tif',Atteg1)
        temp = 1*FZ_dir
        out_K_all += temp #temp_filter
        _U.save_tif_v21(f'out_K_all-dir{i}.tif',abs(out_K_all),run_parameter)

        
    return out_K_all

def GetWiener2(opt,Wiener1,FZotf,PGxy,PGz,cfai,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    
    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Nshifts=opt.Nshifts
    
    '构建分母'
    Msum_all = w1
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        for j in jiesu:
            'amp'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            Msum_all += FZotf[i*Nshifts+j]**2 * abs(amp)**2

    '构建APO'
    return Msum_all#finral_k