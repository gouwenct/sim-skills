import numpy as np
import torch 
import pickle   #写入txt字典

import Utils as _U
import WindowF as _WF
import DataProcess as _DP
from tifffile import imread


fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftn = np.fft.fftn
ifftn = np.fft.ifftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

tfft2 = torch.fft.fft2
tifft2 = torch.fft.ifft2
tfftn = torch.fft.fftn
tifftn = torch.fft.ifftn
tfftshift = torch.fft.fftshift
tifftshift = torch.fft.ifftshift



#产生相位矩阵
def IrTEST(opt,PGxy,PGz,kuodaxy=2,kuodaz=2,Zchushu=0,pz_true=0):
    '''
    统一的移动相位mat
    传入参数估计的PGxy，PGz
    这里的PGz是使用的不同的PGz
    V2.1 增加仅xy z上的移动
    '''
    nz=(opt.nz+2*opt.Zpadding)*kuodaz
    nx=opt.nx*kuodaxy
    ny=opt.ny*kuodaxy
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=kuodaxy,kuodaz=kuodaz) #原点在中心处  -nx/2 ,nx/2
    # print(f"IrTestZ范围={ZV[0]}~{ZV[-1]}")
    # print(f"IrTestx范围={XV[0][0]}~{XV[-1][0]}")
    Irtest = []
    Irtest_1 = []
    # Irtestxy = []
    # Irtestz = []
    for i in range(opt.Nangles):#3个方向
        Pxyabs = PGxy[i]/nx
        Pzabs = PGz[i]/nz#(Zchushu)
        for j in range(opt.Norders):#7阶   0  -2  2  (-1,-1)  (-1,1） （1，-1） （1，1）
            Pxy = Pxyabs[j]        
            Pz =  Pzabs[j]
            # shiftmat = np.zeros((nz,nx,ny),dtype=opt.dtype_com)
            shiftmat = np.exp(2j*np.pi*(Pxy[0]*XV+Pxy[1]*YV + Pz*ZV),dtype=opt.dtype_com) #* np.exp(2j*np.pi*Pz*ZV)
            # if opt.flag_N=='MP':
            #     shiftmatxy = np.exp(2j*np.pi*(Pxy[0]*XV+Pxy[1]*YV+0*ZV) ) 
            #     shiftmatz = np.exp(2j*np.pi*(0*XV+0*YV+Pz*ZV) ) 
            #     Irtestxy.append(shiftmatxy)
            #     Irtestz.append(shiftmatz)
            Irtest.append(shiftmat)
            shiftmat_1 = np.exp(2j*np.pi*(-1*Pxy[0]*XV-1*Pxy[1]*YV -1*Pz*ZV ),dtype=opt.dtype_com ) #
            Irtest_1.append(shiftmat_1)
            
    return Irtest,Irtest_1 #,Irtestxy,Irtestz     #返回为列表

def choose_jiesu(opt,flagg):
    if flagg==0:              #0
        # print('MsumFM 仅含0阶')
        jiesu=[0]

    elif flagg==1 or flagg==3:  # ±1 or 0±1              
        if opt.Norders==5:       ##slice ±1阶
            # print('slice FM 仅含±1阶 [3,4]')
            jiesu=[3,4]
        elif opt.flag_N == 'MP' and opt.Norders==7:   ##mp ±1
            # print('MP FM 仅含±1阶 [3,4,5,6]')
            jiesu=[3,4,5,6]
            # jiesu=[3]
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
        if opt.Norders==5:       ##slice ±1阶
            print('slice FM [1,2,3,4]')
            jiesu=[1,2,3,4]
        elif opt.flag_N == 'MP' and opt.Norders==7 :   ##mp ±1
            # print('MP FM 仅含±1阶 [3,4,5,6]')
            jiesu=[1,2,3,4,5,6]

        elif opt.Norders==3 and opt.flag_N == '2D':
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
        if j==0: #0jie 
            #2D
            if opt.flag_N=='2D': 
                Astr = 0*Astrbase
                fwhm = FWHMbase
            #3D
            elif opt.flag_N=='3D' or opt.flag_N=='MP':
                Astr = 0*Astrbase
                fwhm = FWHMbase

            
        else:    #其他阶
            #2D
            if opt.flag_N=='2D': 
                Astr = Astrbase/1
                fwhm = FWHMbase
            #3D
            elif opt.flag_N=='3D' or opt.flag_N=='MP':
                Astr = Astrbase/1
                # Astr = 0.9/1.05    
                fwhm = FWHMbase
    
    #HIFI特有的notch 第二个
    elif notch==2:      
        if j==0:    
            #2D
            if opt.flag_N=='2D': 
                Astr = Astrbase/1.05
                fwhm = FWHMbase
            #3D
            elif opt.flag_N=='3D' or opt.flag_N=='MP':
                Astr = Astrbase/1
                # Astr = 0.9/1.05
                fwhm = 1 

        else:         #其他阶
            #2D
            if opt.flag_N=='2D':
                Astr = Astrbase/1.15
                fwhm = FWHMbase
            #3D
            elif opt.flag_N=='3D' or opt.flag_N=='MP':  
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

    # _U.save_tif(f'Atteg-{i,j}-{np.round(Astr,2),fwhm}.tif',Atteg)
    return Atteg
#分母部分
def GetMsum(opt,Irtest,cfai,PGxy,PGz,run_parameter=[],chushi=5,flagg=4,Debug=0,direction=[0,1,2],notch=0):
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
    nz = opt.nz_re          #(opt.nz+2*opt.Zpadding)*opt.kuodapz
    Norders=opt.Norders                #  7 5 3
    nc = opt.nc
    
    "1 OTF转psf"
    OTF3D = _U.image_reshape(opt.OTF,3) 
    OTF3D = _U.ToTensor(OTF3D).to(device=opt.device)
    PSF3D = tfftshift(_U.tifftn(OTF3D)).real
    print('PSF取消real 操作')
    "2 OTF2转psf2"
    OTFmask3D =  _U.ToTensor(opt.OTFmask3D).to(device=opt.device)
    PSFmask3D = tfftshift(_U.tifftn(OTFmask3D)).real   #取real时会将不对称的OTF变为对称的0.5

                                                        #无real时 会保留一侧的mask
                                                        #都比用生成方法好（因为生成方式会大于这个值）

    if torch.is_complex(OTF3D):
        dtype = opt.tdtype_com
    else:
        dtype = opt.tdtype_real
        
    FZotf = torch.zeros((nc,nz,nx,ny),dtype=dtype)     #
    Msum = torch.zeros((nz,nx,ny),dtype=dtype)         #
    # Msum2 = torch.zeros((nz,nx,ny),dtype=dtype)        #HIFI的第二个
    OTFnotchall = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)   #Compute sum of OTF*notch
    MsumMask = torch.zeros((nz,nx,ny),dtype=torch.bool)
    

    # '3、0阶1阶notch' #open-3DSIM用
    # if run_parameter['re_mode']==3:

    #     # _U.save_tif('Mask021.tif',Mask021)
    #     notch0 = _WF.Notch_fliter(opt,PGxy,PGz,0,0,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'] )
    #     notch0 = notch0/np.max(notch0)
    #     Mask0 =  opt.OTFmask3D
    #     OTFtemp = opt.OTF[0] #imread(r'E:\matlab2-code\open-3DSIMv2.2\Open_3DSIM_MATLAB_v2.2\OTFem.tif')# np.copy(OTF3D[0].cpu())  #0阶OTF
    #     OTFtemp[OTFtemp>0.5]=0.5
    #     notch0 = notch0*Mask0*OTFtemp
    #     notch0_ifft = fftshift(_U.ifftn(notch0)).real
    #     notch0_ifft = _U.ToTensor(notch0_ifft).to(device=opt.device)


    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)
    
    for i in direction:              #调试dirct 控制方向
        for j in (jiesu):  #调试
            # if i!=0 and j==0:
            #     continue

            #【移动矩阵】
            filename = str(i)+str(j)+'.tif'
            tIrtest =_U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)


            #移动-【OTF】 V4.53 带复数的OTF
            OTF_move = (_U.tfftn(tifftshift (PSF3D*tIrtest) )).cpu()           #GPU -> CPU
            OTF_move = _U.normalize(OTF_move) #归一化   分子没做归一化，分母也不做; 1.3改动 分子不敢做，但分母做 
                                                                            #;但从仿真结果来看 不做效果更好(感觉是OTF的问题)
            # _U.save_tif('')
            #移动-【mask】
            '生成式mask' 
            # OTFmask3D_move, _ = _DP.otf_mask_3d_2d(opt,PGxy =PGxy[i,j],PGz = [PGz[i,j]], Debug=0) #
            # OTFmask3D_move = torch.from_numpy(OTFmask3D_move).to(torch.bool)

            # '移动式mask'   
            '移动式mask'
            OTFmask3D_move = abs((_U.tfftn( tifftshift(PSFmask3D*tIrtest)) ).cpu())               #GPU->CPU
            # _U.save_tif_v21(fr'OTFmask3D_moveXYZ-{i,j}.tif',OTFmask3D_move,run_parameter=run_parameter)

            OTFmask3D_move[(OTFmask3D_move)>0.9]=1
            OTFmask3D_move[OTFmask3D_move!=1]=0
            OTFmask3D_move = OTFmask3D_move.to(torch.bool)


            # OTFmask3D_move = abs((_U.tfftn( tifftshift(PSFmask3D*tIrtest)) ).cpu())               #GPU->CPU
            # # _U.save_tif_v21(fr'OTFmask3D_moveXYZ-{i,j}.tif',OTFmask3D_move,run_parameter=run_parameter)

            # OTFmask3D_move[(OTFmask3D_move)>0.9]=1
            # OTFmask3D_move[OTFmask3D_move!=1]=0


            #移动截断，制作【FZotf】
            #腐蚀一圈
            # if j <1:
            #     OTFmask3D_move = torch.from_numpy(_U.Cir_Erode_3D(OTFmask3D_move.numpy(),size=5))

            FZotf[i*Norders+j] = (OTF_move)#*(OTFmask3D_move)
            # _U.save_tif_v21(fr'OTFmask3D_move-{i,j}.tif',OTFmask3D_move,run_parameter=run_parameter)

            if chushi==4:#启用方法2
                FZotf[i*Norders+j] = FZotf[i*Norders+j]*cfai[i,j,0]


            '制作【notch】' #mode=0分母不做notch，FZ才做
            if run_parameter['re_mode']==1 and notch!=0:  #HIFI
                Atteg1 = 1#choose_notch(opt,notch,i,j,PGxy,PGz,run_parameter['attStr'],run_parameter['attFWHM'])    #notch=1
                Atteg2 = 1#choose_notch(opt,notch+1,i,j,PGxy,PGz,run_parameter['attStr'],run_parameter['attFWHM'])  #notch=2

            elif run_parameter['re_mode']==0 and notch!=0 and j <=2: #2阶notch 
                Atteg1 = 1#choose_notch(opt,notch,i,j,PGxy,PGz,run_parameter['attStrFZ'],run_parameter['attFWHMFZ'])    
                Atteg2 = 1

            elif run_parameter['re_mode']==0 and notch!=0 and j >2: # 1阶notch 
                Atteg1 = 1#choose_notch(opt,notch,i,j,PGxy,PGz,run_parameter['attStr'],run_parameter['attFWHM'])  
                Atteg2 = 1

            elif run_parameter['re_mode']==2 and notch!=0: #open-3DSIM
                Atteg1 = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,PGz,i,j))
                # _U.save_tif(f'Atteg1-{i,j}.tif',Atteg1.numpy())
                Atteg1 = Atteg1*OTFmask3D_move
                # _U.save_tif(f'Atteg1-mask-{i,j}.tif',Atteg1.numpy())

                OTF_move[OTF_move>0.4]=0.4 
                OTF_move_notch = (OTF_move)*Atteg1
                # _U.save_tif(f'OTF_move_notch-{i,j}.tif',OTF_move_notch)
                #OTF_move_notch 
                if j==0 or j>=3:
                    OTFnotchall += abs(OTF_move_notch)/2;
                else:
                    OTFnotchall += abs(OTF_move_notch)
                Atteg2 = 1

            # elif run_parameter['re_mode']==3 and notch!=0: #notch、OTF、OTFmask一起移动
            #     if j == 0 :
            #         OTF_move_notch = notch0
            #     elif j <=2 : #0阶2阶
            #         OTF_move_notch = (_U.tfftn(tifftshift (notch0_ifft*tIrtest) )).cpu() 
                
            #     elif j>2 :#1阶
            #         #xy移动
            #         OTF_move_notch = 0.5*(_U.tfftn(tifftshift (notch0_ifft*tIrtest) )).cpu()
            #         # OTF_XYmove_ifft = fftshift(_U.ifftn(OTF_XYmove_))
            #         # #z移动
            #         # OTF_move_notch = (_U.fftn(ifftshift ( OTF_XYmove_ifft*Ir_z ) )) + (_U.fftn(ifftshift ( OTF_XYmove_ifft*Ir_z_ ) ))
            #     OTF_move_notch = _U.ToNumpy(OTF_move_notch)
            #     OTF_move_notch[OTF_move_notch<1e-9]=0 
            #     # _U.save_tif(f'OTF_move_notch-{i,j}.tif',abs(OTF_move_notch))
            #     #OTF_move_notch

            #     if j==0:
            #         OTFnotchall += abs(OTF_move_notch)/2;
            #     else:
            #         OTFnotchall += abs(OTF_move_notch)
            #     # _U.save_tif(f'OTFnotchall-{i,j}.tif',abs(OTFnotchall))
            #     Atteg1 = 1
            #     Atteg2 = 1

            else:
                Atteg1 = 1
                Atteg2 = 1

            '各阶相加'
            Msum  += (FZotf[i*Norders+j] * FZotf[i*Norders+j].conj ) * Atteg1            #  HIFI中的第一个 /wiener
            # Msum2 += (FZotf[i*Norders+j] * FZotf[i*Norders+j].conj ) * Atteg2            #  HIFI中的第二个
            MsumMask = MsumMask | OTFmask3D_move
            
            if Debug==1:  #isdebug[5]
                pass

        # pass
        # FZotf 不带notch， notch打在Msum上（0阶无，其他阶有） 
    return Msum.numpy(),FZotf.numpy(),MsumMask.numpy(),OTFnotchall.numpy()


def GetMsum_joint(opt,Irtest,cfai,PGxy,PGz,run_parameter=[],chushi=5,flagg=4,Debug=0,direction=[0,1,2],notch=0):
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
    nz = opt.nz_re          #(opt.nz+2*opt.Zpadding)*opt.kuodapz
    Norders=opt.Norders                #  7 5 3
    nc = opt.nc
    
    "1 OTF转psf"
    OTF3D = _U.image_reshape(opt.OTF,3) 
    OTF3D = _U.ToTensor(OTF3D).to(device=opt.device)
    PSF3D = tfftshift(_U.tifftn(OTF3D)).real
    print('PSF取消real 操作')
    "2 OTF2转psf2"
    OTFmask3D =  _U.ToTensor(opt.OTFmask3D).to(device=opt.device)
    PSFmask3D = tfftshift(_U.tifftn(OTFmask3D)).real   #取real时会将不对称的OTF变为对称的0.5

                                                        #无real时 会保留一侧的mask
                                                        #都比用生成方法好（因为生成方式会大于这个值）

    
    if torch.is_complex(OTF3D):
        dtype = opt.tdtype_com
    else:
        dtype = opt.tdtype_real
        
    FZotf = torch.zeros((nc,nz,nx,ny),dtype=dtype)     #
    Msum = torch.zeros((nz,nx,ny),dtype=dtype)         #
    # Msum2 = torch.zeros((nz,nx,ny),dtype=dtype)        #HIFI的第二个
    OTFnotchall = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)   #Compute sum of OTF*notch
    MsumMask = torch.zeros((nz,nx,ny),dtype=torch.bool)
    

    '3、0阶1阶notch' #open-3DSIM用
    # if run_parameter['re_mode']==3:

    #     # _U.save_tif('Mask021.tif',Mask021)
    #     notch0 = _WF.Notch_fliter(opt,PGxy,PGz,0,0,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'] )
    #     notch0 = notch0/np.max(notch0)
    #     Mask0 =  opt.OTFmask3D
    #     OTFtemp = opt.OTF[0] #imread(r'E:\matlab2-code\open-3DSIMv2.2\Open_3DSIM_MATLAB_v2.2\OTFem.tif')# np.copy(OTF3D[0].cpu())  #0阶OTF
    #     OTFtemp[OTFtemp>0.5]=0.5
    #     notch0 = notch0*Mask0*OTFtemp
    #     notch0_ifft = fftshift(_U.ifftn(notch0)).real
    #     notch0_ifft = _U.ToTensor(notch0_ifft).to(device=opt.device)


    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)
    
    for i in direction:              #调试dirct 控制方向
        for j in (jiesu):  #调试
            # if i!=0 and j==0:
            #     continue
            
            #【移动矩阵】
            filename = str(i)+str(j)+'.tif'
            tIrtest =_U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)

            #移动-【OTF】
            OTF_move = (_U.tfftn(tifftshift (PSF3D*tIrtest) )).cpu()           #GPU -> CPU
            # OTF_move = _U.normalize(OTF_move) #归一化   分子没做归一化，分母也不做; 1.3改动 分子不敢做，但分母做 
                                                                            #;但从仿真结果来看 不做效果更好(感觉是OTF的问题)
            # _U.save_tif('')
            #移动-【mask】
            '生成式mask' 
            # OTFmask3D_move, _ = _DP.otf_mask_3d_2d(opt,PGxy =PGxy[i,j],PGz = [PGz[i,j]], Debug=0) #
            # OTFmask3D_move = torch.from_numpy(OTFmask3D_move).to(torch.bool)

            # '移动式mask'   
            '移动式mask'
            OTFmask3D_move = abs((_U.tfftn( tifftshift(PSFmask3D*tIrtest)) ).cpu())               #GPU->CPU
            # _U.save_tif_v21(fr'OTFmask3D_moveXYZ-{i,j}.tif',OTFmask3D_move,run_parameter=run_parameter)

            OTFmask3D_move[(OTFmask3D_move)>0.9]=1
            OTFmask3D_move[OTFmask3D_move!=1]=0
            OTFmask3D_move = OTFmask3D_move.to(torch.bool)

            #移动截断，制作【FZotf】
            #腐蚀一圈
            # if j <1:
            #     OTFmask3D_move = torch.from_numpy(_U.Cir_Erode_3D(OTFmask3D_move.numpy(),size=5))

            FZotf[i*Norders+j] = (OTF_move)#*(OTFmask3D_move)
            # _U.save_tif_v21(fr'OTFmask3D_move-{i,j}.tif',OTFmask3D_move,run_parameter=run_parameter)

            if chushi==4:#启用方法2
                FZotf[i*Norders+j] = FZotf[i*Norders+j]*cfai[i,j,0]


            '制作【notch】' #mode=0分母不做notch，FZ才做
            if run_parameter['re_mode']==1 and notch!=0:  #HIFI
                Atteg1 = 1#choose_notch(opt,notch,i,j,PGxy,PGz,run_parameter['attStr'],run_parameter['attFWHM'])    #notch=1
                Atteg2 = 1#choose_notch(opt,notch+1,i,j,PGxy,PGz,run_parameter['attStr'],run_parameter['attFWHM'])  #notch=2

            elif run_parameter['re_mode']==0 and notch!=0 and j <=2: #2阶notch 
                Atteg1 = 1#choose_notch(opt,notch,i,j,PGxy,PGz,run_parameter['attStrFZ'],run_parameter['attFWHMFZ'])    
                Atteg2 = 1

            elif run_parameter['re_mode']==0 and notch!=0 and j >2: # 1阶notch 
                Atteg1 = 1#choose_notch(opt,notch,i,j,PGxy,PGz,run_parameter['attStr'],run_parameter['attFWHM'])  
                Atteg2 = 1

            elif run_parameter['re_mode']==2 and notch!=0: #open-3DSIM
                Atteg1 = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,PGz,i,j))
                # _U.save_tif(f'Atteg1-{i,j}.tif',Atteg1.numpy())
                Atteg1 = Atteg1*OTFmask3D_move
                # _U.save_tif(f'Atteg1-mask-{i,j}.tif',Atteg1.numpy())

                OTF_move[OTF_move>0.4]=0.4 
                OTF_move_notch = (OTF_move)*Atteg1
                # _U.save_tif(f'OTF_move_notch-{i,j}.tif',OTF_move_notch)
                #OTF_move_notch 
                if j==0 or j>=3:
                    OTFnotchall += abs(OTF_move_notch)/2;
                else:
                    OTFnotchall += abs(OTF_move_notch)
                Atteg2 = 1

            # elif run_parameter['re_mode']==3 and notch!=0: #notch、OTF、OTFmask一起移动
            #     if j == 0 :
            #         OTF_move_notch = notch0
            #     elif j <=2 : #0阶2阶
            #         OTF_move_notch = (_U.tfftn(tifftshift (notch0_ifft*tIrtest) )).cpu() 
                
            #     elif j>2 :#1阶
            #         #xy移动
            #         OTF_move_notch = 0.5*(_U.tfftn(tifftshift (notch0_ifft*tIrtest) )).cpu()
            #         # OTF_XYmove_ifft = fftshift(_U.ifftn(OTF_XYmove_))
            #         # #z移动
            #         # OTF_move_notch = (_U.fftn(ifftshift ( OTF_XYmove_ifft*Ir_z ) )) + (_U.fftn(ifftshift ( OTF_XYmove_ifft*Ir_z_ ) ))
            #     OTF_move_notch = _U.ToNumpy(OTF_move_notch)
            #     OTF_move_notch[OTF_move_notch<1e-9]=0 
            #     # _U.save_tif(f'OTF_move_notch-{i,j}.tif',abs(OTF_move_notch))
            #     #OTF_move_notch

            #     if j==0:
            #         OTFnotchall += abs(OTF_move_notch)/2;
            #     else:
            #         OTFnotchall += abs(OTF_move_notch)
            #     # _U.save_tif(f'OTFnotchall-{i,j}.tif',abs(OTFnotchall))
            #     Atteg1 = 1
            #     Atteg2 = 1

            else:
                Atteg1 = 1
                Atteg2 = 1

            '各阶相加'
            Msum  += (FZotf[i*Norders+j]*FZotf[i*Norders+j].conj())*Atteg1            #  HIFI中的第一个 /wiener
            # Msum2 += abs(FZotf[i*Norders+j])**2*Atteg2            #  HIFI中的第二个
            MsumMask = MsumMask | OTFmask3D_move
            
            
            #就多这一步
            if j>=run_parameter['joints'] and j<3:
                Msum += (FZotf[i*Norders+j]*FZotf[i*Norders+j].conj())* Atteg1   
                
            if Debug==1:  #isdebug[5]
                pass

        # pass
        # FZotf 不带notch， notch打在Msum上（0阶无，其他阶有） 
    return Msum.numpy(),FZotf.numpy(),MsumMask.numpy(),OTFnotchall.numpy()







def GetMsum_mask(opt,Irtest,run_parameter=[],flagg=4,Debug=0,direction=[0,1,2]):
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
    nz = opt.nz_re          #(opt.nz+2*opt.Zpadding)*opt.kuodapz
    Norders=opt.Norders                #  7 5 3
    nc = opt.nc
    
    "2 OTF2转psf2"
    OTFmask3D =  _U.ToTensor(opt.OTFmask3D).to(device=opt.device)
    PSFmask3D = tfftshift(_U.tifftn(OTFmask3D)).real   #取real时会将不对称的OTF变为对称的0.5

                                                        #无real时 会保留一侧的mask
                                                        #都比用生成方法好（因为生成方式会大于这个值）

    MsumMask = torch.zeros((nz,nx,ny),dtype=torch.bool)

    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)
    
    for i in direction:              #调试dirct 控制方向
        for j in (jiesu):  #调试
            #【移动矩阵】
            filename = str(i)+str(j)+'.tif'
            tIrtest =_U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)
            
            # _U.save_tif('')
            #移动-【mask】
            '生成式mask' 
            # OTFmask3D_move, _ = _DP.otf_mask_3d_2d(opt,PGxy =PGxy[i,j],PGz = [PGz[i,j]], Debug=0) #
            # OTFmask3D_move = torch.from_numpy(OTFmask3D_move).to(torch.bool)

            # '移动式mask'   
            '移动式mask'
            OTFmask3D_move = abs((_U.tfftn( tifftshift(PSFmask3D*tIrtest)) ).cpu())               #GPU->CPU
            # _U.save_tif_v21(fr'OTFmask3D_moveXYZ-{i,j}.tif',OTFmask3D_move,run_parameter=run_parameter)

            OTFmask3D_move[(OTFmask3D_move)>0.5]=1
            OTFmask3D_move[OTFmask3D_move!=1]=0
            OTFmask3D_move = OTFmask3D_move.to(torch.bool)

            MsumMask = MsumMask | OTFmask3D_move
            
        # pass
        # FZotf 不带notch， notch打在Msum上（0阶无，其他阶有） 
    return MsumMask.numpy()

def GetFZall(opt,sk,FZotf,Irtest,cfai,Msum,PGxy,PGz,run_parameter=[],chushi=5,flagg=3,Debug=1,direction=[0,1,2],notch=0):
    #opt,sk,FZotf,Irtest,kuodaxy=2,kuodaz=2,Norders=7,flagg=3
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
    sk = _U.ToTensor(sk)
    FZotf = _U.ToTensor(FZotf)
    FZnotchall = torch.zeros((nz,nx,ny),dtype=torch.float32)

   
    "OTF2转PSF2"
    # Om2 =  _U.ToTensor(opt.OTFmask3D).to(device=opt.device)                       #GPU
    # psf2 = tfftshift(_U.tifftn(Om2))
    "OTF2转psf2"
    OTFmask3D =  _U.ToTensor(opt.OTFmask3D).to(device=opt.device)
    PSFmask3D = tfftshift(_U.tifftn(OTFmask3D)).real

    'FM' 
    #用于查看各个分量的重建情况
    "SK横向补零"
    sr_pad = torch.zeros((nc,nz,nx,ny),dtype=opt.tdtype_com)#.to(device=opt.device)  #CPU
    sk_pad = _U.pad_or_cut(sk,x=nx,y=ny,z=nz)#.to(device=opt.device)                  #GPU
    
    for i in range(nc):
        sr_pad[i] = (_U.tifftn(sk_pad[i]) )#).cpu() )#.to(device=opt.device)             #GPU->CPU
        # if i ==3 or i==6:
        #     temp = _U.tfftn(sr_pad[i])
        #     _U.save_tif_v21(fr'sk_pad-{0,i}.tif',abs(temp),run_parameter)

    del sk_pad 

    'Apo'
    Zjust = opt.Apoz   # Zjust 大 轴向越衰减
    Apo = _WF.bhs_3D_Z(opt,opt.fc,opt.fc,kuodaz=opt.kuodapz, Zjust=Zjust )
    
    "移动sk——pad，OTF2"
    sr_pad = sr_pad.to(device=opt.device)                                          #CPU->GPU
    FZall =torch.zeros((nz,nx,ny),dtype=opt.tdtype_com)                            #CPU
    FZall_SD = torch.zeros((3,nz,nx,ny),dtype=opt.tdtype_com)

    if run_parameter['re_mode']==3:
        notch0 = (_WF.Notch_fliter(opt,PGxy,PGz,0,0,Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ']))
        notch0 = notch0/np.max(notch0)
        Mask0 =  opt.OTFmask3D
        OTFtemp = opt.OTF[0] #imread(r'E:\matlab2-code\open-3DSIMv2.2\Open_3DSIM_MATLAB_v2.2\OTFem.tif')# np.copy(OTF3D[0].cpu())  #0阶OTF
        OTFtemp[OTFtemp>0.5]=0.5
        notch0 = notch0*Mask0*(OTFtemp**0)
        notch0 = notch0/np.max(notch0)
        notch0_ifft = fftshift(_U.ifftn(notch0)).real
        notch0_ifft = _U.ToTensor(notch0_ifft).to(device=opt.device)


    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)
    print(jiesu)
    for i in direction:
        for j in jiesu:

            '【移动矩阵】'
            filename = str(i)+str(j)
            tIrtest = _U.ToTensor(Irtest[i*opt.Norders+j]).to(device=opt.device)   #GPU
            # if opt.flag_N=='MP':
            #     tIrtestxy = _U.ToTensor(Irtestxy[i*opt.Norders+j]).to(device=opt.device)   #GPU
            #     tIrtestz = _U.ToTensor(Irtestz[i*opt.Norders+j]).to(device=opt.device)   #GPU

            '移动-【mask】' #--生成mask
            '移动式mask'   
            
            OTFmask3D_move = abs((_U.tfftn( tifftshift(PSFmask3D*tIrtest)) ).cpu())               #GPU->CPU
            # _U.save_tif_v21(fr'OTFmask3D_moveXYZ-{i,j}.tif',OTFmask3D_move,run_parameter=run_parameter)

            OTFmask3D_move[(OTFmask3D_move)>0.9]=1
            OTFmask3D_move[OTFmask3D_move!=1]=0


            # OTFmask3D_move, _ = _DP.otf_mask_3d_2d(opt,PGxy =PGxy[i,j],PGz = [PGz[i,j]], Debug=0) #
            # OTFmask3D_move = torch.from_numpy(OTFmask3D_move).to(torch.bool)

            # if opt.flag_N!='MP':  
            #     OTFmask3D_move = abs((_U.tfftn( tifftshift(psf2*tIrtest)) ).cpu())               #GPU->CPU
            # if opt.flag_N=='MP':
            # OTFmask3D_move = abs(_U.tfftn( tifftshift(psf2*tIrtestxy)) )
            # OTFmask3D_move[(OTFmask3D_move)>0.90]=1
            # OTFmask3D_move[OTFmask3D_move!=1]=0
            # OTFmask3D_move = abs(_U.tfftn( tifftshift(tfftshift(_U.tifftn(OTFmask3D_move))*tIrtestz)) ).cpu()


            # OTFmask3D_move[(OTFmask3D_move)>0.90]=1
            # OTFmask3D_move[OTFmask3D_move!=1]=0

            '移动-【sk】'
            sk_shift = (_U.tfftn(sr_pad[i*opt.Norders+j]*tIrtest)).cpu()                 #GPU-CPU
            # if i==0 and (j == 0 or j==3 or j==6):
            #     _U.save_tif_v21(fr'sk_shift1-{i,j}.tif',abs(sk_shift),run_parameter)
            
            #对1阶打上切趾
            # if j >=3:
            #     sk_shift = sk_shift*Apo

            torch.cuda.empty_cache()
            '制作【notch】'
            if notch == 0:
                Atteg1 = 1  # 所有都不加
            
            #open3D-SIM的notch 方法
            elif notch==1 and run_parameter['re_mode']==2 : #open-3DSIM
                Atteg1 = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,PGz,i,j))
                OTFtemp = FZotf[i*opt.Norders+j]
                OTFtemp[OTFtemp>0.4]=0.4

                Atteg1 = abs(Atteg1*OTFmask3D_move*abs(OTFtemp)**0)  #
                Atteg1 = Atteg1/torch.max(Atteg1)

            elif notch==1 and run_parameter['re_mode']==3:
                if j == 0:
                    Atteg1 = (notch0)
                    
                elif j<=2: #2
                    Atteg1 = (_U.tfftn(tifftshift (notch0_ifft*tIrtest) )).cpu() 
                    # Atteg1 = torch.from_numpy(_WF.Notch_fliter(opt,PGxy,0,i,j))

                else:  #1
                    Atteg1 = 0.5*(_U.tfftn(tifftshift (notch0_ifft*tIrtest) )).cpu() 
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

            if j==0 and run_parameter['notch0th']!= 0: #0阶时 判断加不加（注意细胞重构时的控制）
                att = run_parameter['notch0th']
                print(f'注意这里施加了0频抑制-{i,j}-{att}')
                Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) #CPU
                # _U.save_tif_v21(f'notch0th-{run_parameter['notch0th']}.tif',Atteg1,run_parameter)

            #'对sk1阶旁边亮点的抑制'
            if opt.flag_N=='MP' and j>=3 and run_parameter['notch1th_pang']!=0:
                att = run_parameter['notch1th_pang']
                print(f'注意这里施加了1阶旁边的0频抑制-{i,j}-{att}')
                Atteg0 = _U.ToTensor(_WF.valAttenuation(opt,0,0,pz=PGz[i,j],Astr=run_parameter['notch1th_pang'],fwhm=1))
                sk_shift = (sk_shift*1)*Atteg0
                # _U.save_tif(f'Atteg0-{i,j}.tif',Atteg0)
        
            if run_parameter['re_mode']==2 or run_parameter['re_mode']==3:
                FZ = sk_shift*Atteg1

            else:  #普通wiener
                '合并【OTF,notch与mask】'
                if j >=1:
                    ################################################
                    if j == 1 or j==2:                          ##尝试抑制2阶的频谱部分
                        FZOTF_ = FZotf[i*opt.Norders+j]
                        # FZOTF_[FZOTF_>0.3] = 0.3
                        # print('OPEN3d')
                        pass
                    #################################################
                    else:
                        FZOTF_ = FZotf[i*opt.Norders+j]
                    FzOTF_notch = (FZOTF_)*Atteg1*OTFmask3D_move          #CPU
                else:
                    FzOTF_notch = (FZotf[i*opt.Norders+j])*Atteg1*(OTFmask3D.cpu())       #0阶
                FZ = sk_shift*FzOTF_notch 
                # _U.save_tif_v21(fr'sk_shift-{i,j}.tif',abs(sk_shift),run_parameter)

            '合并FZall'
            FZall += FZ
            FZall_SD[i] += FZ 
            # if i ==0:
            #     _U.save_tif_v21(fr'FZall-{i,j}.tif',abs(FZall),run_parameter)

            '合并fznotch'
            FZnotchall += Atteg1

    
    notch0th_ = run_parameter['notch0th'] #0
    attFWHM0_ = run_parameter['attFWHM0']
    attStrFZ_ = run_parameter['attStrFZ'] #2
    attFWHMFZ_ = run_parameter['attFWHMFZ']
    attStr_ = run_parameter['attStr']
    attFWHM_ = run_parameter['attFWHM'] #1
    notch1th_pang = run_parameter['notch1th_pang'] #1'

    if notch==0:
        _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang}).tif',abs(FZnotchall[0]),run_parameter)        
    else:
        _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang},{attStrFZ_}_{attFWHMFZ_}-{attStr_}-{attFWHM_}).tif',abs(FZnotchall[0]),run_parameter) 
    
    return _U.ToNumpy(FZall),_U.ToNumpy(FZall_SD)
 
def GetFZall_joint(opt,sk,sk2,FZotf,Irtest,cfai,Msum,PGxy,PGz,run_parameter=[],chushi=5,flagg=3,Debug=1,direction=[0,1,2],notch=0):
    #opt,sk,FZotf,Irtest,kuodaxy=2,kuodaz=2,Norders=7,flagg=3
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
    sk = _U.ToTensor(sk)
    sk2 = _U.ToTensor(sk2)
    FZotf = _U.ToTensor(FZotf)
    FZnotchall = torch.zeros((nz,nx,ny),dtype=torch.float32)

   
    "OTF2转PSF2"
    # Om2 =  _U.ToTensor(opt.OTFmask3D).to(device=opt.device)                       #GPU
    # psf2 = tfftshift(_U.tifftn(Om2))
    "OTF2转psf2"
    OTFmask3D =  _U.ToTensor(opt.OTFmask3D).to(device=opt.device)
    PSFmask3D = tfftshift(_U.tifftn(OTFmask3D)).real

    'FM' 
    #用于查看各个分量的重建情况
    "SK横向补零"
    sr_pad = torch.zeros((nc,nz,nx,ny),dtype=opt.tdtype_com)#.to(device=opt.device)  #CPU
    sr2_pad = torch.zeros((9,nz,nx,ny),dtype=opt.tdtype_com)
    sk_pad = _U.pad_or_cut(sk,x=nx,y=ny,z=nz)#.to(device=opt.device)                  #GPU
    sk2_pad = _U.pad_or_cut(sk2,x=nx,y=ny,z=nz)
    
    for i in range(nc):
        sr_pad[i] = (_U.tifftn(sk_pad[i]) )#).cpu() )#.to(device=opt.device)             #GPU->CPU
    for i in range(9):
        sr2_pad[i] = (_U.tifftn(sk2_pad[i]) )

    del sk_pad,sk2_pad

    'Apo'
    Zjust = opt.Apoz   # Zjust 大 轴向越衰减
    Apo = _WF.bhs_3D_Z(opt,opt.fc,opt.fc,kuodaz=opt.kuodapz, Zjust=Zjust )
    
    "移动sk——pad，OTF2"
    sr_pad = sr_pad.to(device=opt.device)                                          #CPU->GPU
    sr2_pad = sr2_pad.to(device=opt.device)      
    FZall =torch.zeros((nz,nx,ny),dtype=opt.tdtype_com)                            #CPU
    FZall_SD = torch.zeros((3,nz,nx,ny),dtype=opt.tdtype_com)

    if run_parameter['re_mode']==3:
        notch0 = (_WF.Notch_fliter(opt,PGxy,PGz,0,0,Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ']))
        notch0 = notch0/np.max(notch0)
        Mask0 =  opt.OTFmask3D
        OTFtemp = opt.OTF[0] #imread(r'E:\matlab2-code\open-3DSIMv2.2\Open_3DSIM_MATLAB_v2.2\OTFem.tif')# np.copy(OTF3D[0].cpu())  #0阶OTF
        OTFtemp[OTFtemp>0.5]=0.5
        notch0 = notch0*Mask0*(OTFtemp**0)
        notch0 = notch0/np.max(notch0)
        notch0_ifft = fftshift(_U.ifftn(notch0)).real
        notch0_ifft = _U.ToTensor(notch0_ifft).to(device=opt.device)


    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)
    print(jiesu)
    for i in direction:
        for j in jiesu:
            '【移动矩阵】'
            filename = str(i)+str(j)
            tIrtest = _U.ToTensor(Irtest[i*opt.Norders+j]).to(device=opt.device)   #GPU

            '移动式mask'   
            OTFmask3D_move = abs((_U.tfftn( tifftshift(PSFmask3D*tIrtest)) ).cpu())               #GPU->CPU
            OTFmask3D_move[(OTFmask3D_move)>0.9]=1
            OTFmask3D_move[OTFmask3D_move!=1]=0

            '移动-【sk】'
            sk_shift = (_U.tfftn(sr_pad[i*opt.Norders+j]*tIrtest)).cpu()                 #GPU-CPU
            torch.cuda.empty_cache()
            
            
            '制作【notch】'
            if notch == 0:
                Atteg1 = 1  # 所有都不加
            
            if notch ==1 and j>0 and j <=2: # 2阶的notch
                Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
            
            elif notch ==1 and j>2:         # 1阶的notch
                Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU
                
            else:
                Atteg1 = 1

            '对skshift的处理'

            if j==0 and run_parameter['notch0th']!= 0: #0阶时 判断加不加（注意细胞重构时的控制）
                att = run_parameter['notch0th']
                print(f'注意这里施加了0频抑制-{i,j}-{att}')
                Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) #CPU

            #'对sk1阶旁边亮点的抑制'
            if opt.flag_N=='MP' and j>=3 and run_parameter['notch1th_pang']!=0:
                att = run_parameter['notch1th_pang']
                print(f'注意这里施加了1阶旁边的0频抑制-{i,j}-{att}')
                Atteg0 = _U.ToTensor(_WF.valAttenuation(opt,0,0,pz=PGz[i,j],Astr=run_parameter['notch1th_pang'],fwhm=1))
                sk_shift = (sk_shift*1)*Atteg0
                # _U.save_tif(f'Atteg0-{i,j}.tif',Atteg0)
        

        
            '合并【OTF,notch与mask】'
            if j >=1:
                FZOTF_ = FZotf[i*opt.Norders+j]
                FzOTF_notch = (FZOTF_)*Atteg1*OTFmask3D_move          #CPU
            elif j==0:
                FzOTF_notch = (FZotf[i*opt.Norders+j])*Atteg1*(OTFmask3D.cpu())       #0阶
                
            FZ = sk_shift*FzOTF_notch 

            '合并FZall'
            FZall += FZ
            FZall_SD[i] += FZ 
            
            
            'joint'
            if j<3 and j>=run_parameter['joints']: #012
                sk2_shift = (_U.tfftn(sr2_pad[i*3+j]*tIrtest)).cpu()                 #GPU-CPU
                FZ = sk2_shift*FzOTF_notch 
                FZall += FZ     
                
            '合并fznotch'
            FZnotchall += Atteg1

    
    notch0th_ = run_parameter['notch0th'] #0
    attFWHM0_ = run_parameter['attFWHM0']
    attStrFZ_ = run_parameter['attStrFZ'] #2
    attFWHMFZ_ = run_parameter['attFWHMFZ']
    attStr_ = run_parameter['attStr']
    attFWHM_ = run_parameter['attFWHM'] #1
    notch1th_pang = run_parameter['notch1th_pang'] #1'

    if notch==0:
        _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang}).tif',abs(FZnotchall[0]),run_parameter)        
    else:
        _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang},{attStrFZ_}_{attFWHMFZ_}-{attStr_}-{attFWHM_}).tif',abs(FZnotchall[0]),run_parameter) 
    
    return _U.ToNumpy(FZall),_U.ToNumpy(FZall_SD)
  

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
    Norders=opt.Norders

    #Msum转空域
    Msumt = abs(_U.ToTensor(Msum).to(device=opt.device))  #取出0阶和1阶来
    # print(f'Msumt--{(Msumt)[8,255,256]}')
    # print(f'Msumt--{(Msumt)[8,257,256]}')
    Msumr = tfftshift(_U.tifftn(Msumt)).real    # GPU 需要取real

    del Msumt,Msum

    #"SK横向补零"
    sk =  _U.ToTensor(sk)
    sk_pad = _U.pad_or_cut(sk,x=nx,y=ny,z=nz)                  #CPU

    #FZOTF #这个OTF应该是没有删去的
    FZOTF = torch.clone(_U.ToTensor(opt.OTF) ).to(device=opt.device)            

    #OTFmask 
    OTFmask = torch.clone(_U.ToTensor(opt.OTFmask3D) ).to(device=opt.device)            

    #return : Wiener_sum
    Wiener_sum = torch.zeros((nz,nx,ny),dtype=opt.tdtype_com)  #CPU
    FZnotchall = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)

    #'mask'
    mask = torch.from_numpy(_WF.mask3(opt,sigma=0.25)).to(device=opt.device)      
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)

    for i in direction:              #调试dirct 控制方向
        for j in jiesu:              #调试
            "【移动矩阵】"
            filename = str(i)+str(j)+'.tif'
            tIrtest =_U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)
            tIrtest_1 =_U.ToTensor(Irtest_1[i*Norders+j]).to(device=opt.device)            #CPU-GPU

            '移动分母'
            Msum_r_move = (Msumr*tIrtest_1)*mask
            
            Msum_move =  abs(_U.tfftn( tifftshift(Msum_r_move)) )                 #GPU
            # Msum_move =  abs(_U.tfftn( tifftshift(Msum_3r[i]*tIrtest_1)) )                 #GPU

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
            '合并fznotch-无意义'
            # FZnotchall += Atteg1

            'sk/分母'
            FZ = (Atteg1 * sk_pad[i*Norders+j]).to(device=opt.device)
            FZ_ = (FZ*FZOTF*OTFmask)
            
            if j==1 or j==2:
                Wiener_sk =  FZ_ / (Msum_move + 1*w3)       #考虑这里要不要加一个Apo
            else:
                Wiener_sk =  FZ_ / (Msum_move + w3)
            
            del FZ

            '移动回来Wiener_sk>Wiener_sk_move'
            Wiener_sr = _U.tifftn(Wiener_sk)

            Wiener_sr_move = (Wiener_sr)*tIrtest
            Wiener_sum += Wiener_sr_move.cpu()#.real

            #检验频谱对称问题
            #保存临时文件
            # if i ==1 or i==2:
            Wiener_sum_k = _U.tfftn(Wiener_sum)

            # _U.save_tif_v21(f'Wiener_sr-{i,j}.tif',Wiener_sr.cpu().real,run_parameter)
            # _U.save_tif_v21(f'FZ-{i,j}.tif',abs(FZ).cpu(),run_parameter)
            # _U.save_tif_v21(f'sk_pad-{i,j}.tif',abs(sk_pad).cpu(),run_parameter)
            # _U.save_tif_v21(f'FZ_-{i,j}.tif',abs(FZ_).cpu(),run_parameter)
            
            # _U.save_tif_v21(f'Msum_move-{i,j}.tif',Msum_move.cpu(),run_parameter)
            
            # _U.save_tif_v21(f'Wiener_sk-{i,j}.tif',abs(Wiener_sk).cpu(),run_parameter)
            
            
            # _U.save_tif_v21(f'Wiener_sum_k-{i,j}.tif',abs(Wiener_sum_k),run_parameter)

            # _U.save_tif_v21(f'Wiener_sr_move-real-{i,j}.tif',Wiener_sr_move.real.cpu(),run_parameter)
            # _U.save_tif_v21(f'Wiener_sum-{i,j}.tif',Wiener_sum,run_parameter)
                


    Wienek_sum = _U.tfftn(Wiener_sum)
    Wiener_sum = Wiener_sum
    # Wiener_sum[Wiener_sum<0]=0
    _U.save_tif_v21(f'WieneR_sum-.tif',Wiener_sum,run_parameter)
    _U.save_tif_v21(f'WieneK_sum-.tif',abs(Wienek_sum),run_parameter)

    notch0th_ = run_parameter['notch0th'] #0
    attFWHM0_ = run_parameter['attFWHM0']
    attStrFZ_ = run_parameter['attStrFZ'] #2
    attFWHMFZ_ = run_parameter['attFWHMFZ']
    attStr_ = run_parameter['attStr']
    attFWHM_ = run_parameter['attFWHM'] #1
    notch1th_pang = run_parameter['notch1th_pang'] #1'

    
    if notch==0:
        _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang}).tif',abs(FZnotchall[0]),run_parameter)        
    else:
        Atteg2_ = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
        Atteg1_ = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU

        _U.save_tif_v21(r'Atteg2_.tif',Atteg2_,run_parameter)
        _U.save_tif_v21(r'Atteg1_.tif',Atteg1_,run_parameter)
        # _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang},{attStrFZ_}_{attFWHMFZ_}-{attStr_}-{attFWHM_}).tif',abs(FZnotchall[0]),run_parameter) 
    

    return  _U.ToNumpy(Wiener_sum),_U.ToNumpy(Wienek_sum)

def recon_konyu_joint(opt,sk,sk2,Msum,Msum_3,Irtest,Irtest_1,flagg,run_parameter,w3,PGxy,xiangwei,notch=0,direction=[0,1,2]):
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
    Norders=opt.Norders

    #Msum转空域
    Msumt = abs(_U.ToTensor(Msum).to(device=opt.device))  #取出0阶和1阶来
    Msumr = tfftshift(_U.tifftn(Msumt)).real    # GPU 需要取real

    del Msumt,Msum

    #"SK横向补零"
    sk =  _U.ToTensor(sk)
    sk2 =  _U.ToTensor(sk2)
    sk_pad = _U.pad_or_cut(sk,x=nx,y=ny,z=nz)                  #CPU
    sk2_pad = _U.pad_or_cut(sk2,x=nx,y=ny,z=nz)                  #CPU

    #FZOTF #这个OTF应该是没有删去的
    FZOTF = torch.clone(_U.ToTensor(opt.OTF*np.exp(1j*xiangwei*np.pi) ) ).to(device=opt.device)            

    #OTFmask 
    OTFmask = torch.clone(_U.ToTensor(opt.OTFmask3D) ).to(device=opt.device)            

    #return : Wiener_sum
    Wiener_sum = torch.zeros((nz,nx,ny),dtype=opt.tdtype_com)  #CPU
    FZnotchall = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)

    #'mask'
    mask = torch.from_numpy(_WF.mask3(opt,sigma=0.25)).to(device=opt.device)      
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)

    for i in direction:              #调试dirct 控制方向
        for j in jiesu:              #调试
            "【移动矩阵】"
            filename = str(i)+str(j)+'.tif'
            tIrtest =_U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)
            tIrtest_1 =_U.ToTensor(Irtest_1[i*Norders+j]).to(device=opt.device)            #CPU-GPU

            '移动分母'
            Msum_r_move = (Msumr*tIrtest_1)*mask
            Msum_move =  abs(_U.tfftn( tifftshift(Msum_r_move)) )                 #GPU
            # Msum_move =  abs(_U.tfftn( tifftshift(Msum_3r[i]*tIrtest_1)) )                 #GPU

            'notch'
            if j==0 : #0阶时 判断加不加（注意细胞重构时的控制）
                Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) #CPU
            elif j!=0:
                if notch == 0:
                    Atteg1 = torch.ones_like(sk_pad[0],dtype=opt.tdtype_real)  # 所有都不加
                elif notch ==1 and j>0 and j <=2: # 2阶的notch
                    Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
                elif notch ==1 and j>2:           # 1阶的notch
                    Atteg1 = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU
                    if run_parameter['notch1th_pang']!=0:  #移动回来的位置
                        Atteg0 = _U.ToTensor(_WF.valAttenuation(opt,-PGxy[i,j,0],-PGxy[i,j,1],Astr=run_parameter['notch1th_pang'],fwhm=1))
                        Atteg1 = Atteg1*Atteg0
                else:
                    Atteg1 = torch.ones_like(sk_pad[0],dtype=opt.tdtype_real)
            
            # #temp
            # if i == 0:
            #     _U.save_tif(rf'Atteg1-{j}.tif',Atteg1)
            #     if j ==3:
            #         _U.save_tif(rf'Atteg0-{j}.tif',Atteg0)
                
            
            'sk/分母'
            FZ = (Atteg1 * sk_pad[i*Norders+j]).to(device=opt.device)
            FZ_ = (FZ*FZOTF*OTFmask)
            Wiener_sk =  FZ_ / (Msum_move + w3)
            
            del FZ

            '移动回来Wiener_sk>Wiener_sk_move'
            Wiener_sr = _U.tifftn(Wiener_sk)

            Wiener_sr_move = (Wiener_sr)*tIrtest
            Wiener_sum += Wiener_sr_move.cpu()#.real

            Wiener_sum_k = _U.tfftn(Wiener_sum)
            
            if j>=run_parameter['joints'] and j<3:
                FZ2 = (Atteg1 * sk2_pad[i*3+j]).to(device=opt.device)
                FZ2_ = (FZ2*FZOTF*OTFmask)
                Wiener_sk =  FZ2_ / (Msum_move + w3)
                Wiener_sr = _U.tifftn(Wiener_sk)
                Wiener_sr_move = (Wiener_sr)*tIrtest
                Wiener_sum += Wiener_sr_move.cpu()#.real
                Wiener_sum_k = _U.tfftn(Wiener_sum)

    Wienek_sum = _U.tfftn(Wiener_sum)
    Wiener_sum = Wiener_sum
    # Wiener_sum[Wiener_sum<0]=0
    _U.save_tif_v21(f'WieneR_sum-.tif',Wiener_sum,run_parameter)
    _U.save_tif_v21(f'WieneK_sum-.tif',abs(Wienek_sum),run_parameter)

    notch0th_ = run_parameter['notch0th'] #0
    attFWHM0_ = run_parameter['attFWHM0']
    attStrFZ_ = run_parameter['attStrFZ'] #2
    attFWHMFZ_ = run_parameter['attFWHMFZ']
    attStr_ = run_parameter['attStr']
    attFWHM_ = run_parameter['attFWHM'] #1
    notch1th_pang = run_parameter['notch1th_pang'] #1'

    
    if notch==0:
        _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang}).tif',abs(FZnotchall[0]),run_parameter)        
    else:
        Atteg2_ = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
        Atteg1_ = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU

        _U.save_tif_v21(r'Atteg2_.tif',Atteg2_,run_parameter)
        _U.save_tif_v21(r'Atteg1_.tif',Atteg1_,run_parameter)
        # _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang},{attStrFZ_}_{attFWHMFZ_}-{attStr_}-{attFWHM_}).tif',abs(FZnotchall[0]),run_parameter) 
    

    return  _U.ToNumpy(Wiener_sum),_U.ToNumpy(Wienek_sum)


def recon_konyu_SD(opt,sk,Msum,Msum_3,PGxy,Irtest,Irtest_1,flagg,run_parameter,w3,notch=0,direction=[0,1,2]):
    '相比之前版本，这个是单独重建每个方向，然后再合起来'
    '需要前面函数提供每个方向的OTF'
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
    Norders=opt.Norders
    
    #"SK横向补零"
    sk =  _U.ToTensor(sk)
    sk_pad = _U.pad_or_cut(sk,x=nx,y=ny,z=nz)                  #CPU

    #FZOTF #这个OTF应该是没有删去的
    FZOTF = torch.clone(_U.ToTensor(opt.OTF) ).to(device=opt.device)            

    #OTFmask 
    OTFmask = torch.clone(_U.ToTensor(opt.OTFmask3D) ).to(device=opt.device)            

    #return : Wiener_sum
    Wiener_sum = torch.zeros((nz,nx,ny),dtype=opt.tdtype_com)  #CPU
    FZnotchall = torch.zeros((nz,nx,ny),dtype=opt.tdtype_real)

    #'mask'
    mask = torch.from_numpy(_WF.mask3(opt,sigma=0.25)).to(device=opt.device)      
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,flagg)
    
    for i in direction:              #调试dirct 控制方向
        
        #Msum转空域
        Msumt_3_i = abs(_U.ToTensor(Msum_3[i]).to(device=opt.device))  #取出0阶和1阶来
        Msumr_3_i = tfftshift(_U.tifftn(Msumt_3_i)).real    # GPU 需要取real
        del Msumt_3_i
        for j in jiesu:              #调试
            "【移动矩阵】"
            filename = str(i)+str(j)+'.tif'
            tIrtest =_U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)
            tIrtest_1 =_U.ToTensor(Irtest_1[i*Norders+j]).to(device=opt.device)            #CPU-GPU

            '移动分母'
            Msum_r_move = (Msumr_3_i*tIrtest_1)*mask
            
            Msum_move =  abs(_U.tfftn( tifftshift(Msum_r_move)) )                 #GPU
            # Msum_move =  abs(_U.tfftn( tifftshift(Msum_3r[i]*tIrtest_1)) )                 #GPU

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
            
            if j>=1 and j <3:
                Atteg3 = _U.ToTensor(_WF.valAttenuation(opt,PGxy[i,j,0]/2,PGxy[i,j,1]/2,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU
                Atteg4 = _U.ToTensor(_WF.valAttenuation(opt,-PGxy[i,j,0]/2,-PGxy[i,j,1]/2,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU
            
            elif j >=3:  #对1阶旁边点的抑制
                Atteg3 = _U.ToTensor(_WF.valAttenuation(opt,-PGxy[i,j,0],-PGxy[i,j,1],Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU
                Atteg4 = 1

            else:
                Atteg3 =  1
                Atteg4 =  1
            
            '合并fznotch-无意义'
            # FZnotchall += Atteg1
            
            'sk/分母'
            
            FZ = (Atteg4*Atteg3*Atteg1 * sk_pad[i*Norders+j]).to(device=opt.device)
            
            FZ_ = (FZ*FZOTF*OTFmask)
            
            if j==1 or j==2:
                Wiener_sk =  FZ_ / (Msum_move + 1*w3)       #考虑这里要不要加一个Apo
            else:
                Wiener_sk =  FZ_ / (Msum_move + w3)
            
            del FZ

            '移动回来Wiener_sk>Wiener_sk_move'
            Wiener_sr = _U.tifftn(Wiener_sk)

            Wiener_sr_move = (Wiener_sr)*tIrtest*mask
            
            Wiener_sum += Wiener_sr_move.cpu()#.real

            #检验频谱对称问题
            #保存临时文件
            # if i ==1 or i==2:
            if i==0  :
                Wiener_sum_k = _U.tfftn(Wiener_sum)

            # _U.save_tif_v21(f'Wiener_sr-{i,j}.tif',Wiener_sr.cpu().real,run_parameter)
            # _U.save_tif_v21(f'FZ-{i,j}.tif',abs(FZ).cpu(),run_parameter)
            # _U.save_tif_v21(f'sk_pad-{i,j}.tif',abs(sk_pad).cpu(),run_parameter)
            # _U.save_tif_v21(f'FZ_-{i,j}.tif',abs(FZ_).cpu(),run_parameter)
            
            # _U.save_tif_v21(f'Msum_move-{i,j}.tif',Msum_move.cpu(),run_parameter)
            
                if j>=1 and j <3:
                    _U.save_tif_v21(f'Atteg3-{i,j}.tif',abs(Atteg3),run_parameter)
                    _U.save_tif_v21(f'Atteg4-{i,j}.tif',abs(Atteg4),run_parameter)
                elif j >=3:
                    _U.save_tif_v21(f'Atteg3-{i,j}.tif',abs(Atteg3),run_parameter)

                _U.save_tif_v21(f'Wiener_sk-{i,j}.tif',abs(Wiener_sk).cpu(),run_parameter)
            
            
                _U.save_tif_v21(f'Wiener_sum_k-{i,j}.tif',abs(Wiener_sum_k),run_parameter)

            # _U.save_tif_v21(f'Wiener_sr_move-real-{i,j}.tif',Wiener_sr_move.real.cpu(),run_parameter)
            # _U.save_tif_v21(f'Wiener_sum-{i,j}.tif',Wiener_sum,run_parameter)
                
    Wienek_sum = _U.tfftn(Wiener_sum)
    Wiener_sum = Wiener_sum
    # Wiener_sum[Wiener_sum<0]=0
    _U.save_tif_v21(f'WieneR_sum-.tif',Wiener_sum,run_parameter)
    _U.save_tif_v21(f'WieneK_sum-.tif',abs(Wienek_sum),run_parameter)

    notch0th_ = run_parameter['notch0th'] #0
    attFWHM0_ = run_parameter['attFWHM0']
    attStrFZ_ = run_parameter['attStrFZ'] #2
    attFWHMFZ_ = run_parameter['attFWHMFZ']
    attStr_ = run_parameter['attStr']
    attFWHM_ = run_parameter['attFWHM'] #1
    notch1th_pang = run_parameter['notch1th_pang'] #1'

    
    if notch==0:
        _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang}).tif',abs(FZnotchall[0]),run_parameter)        
    else:
        Atteg2_ = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
        Atteg1_ = _U.ToTensor(_WF.valAttenuation(opt,Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU

        _U.save_tif_v21(r'Atteg2_.tif',Atteg2_,run_parameter)
        _U.save_tif_v21(r'Atteg1_.tif',Atteg1_,run_parameter)
        # _U.save_tif_v21(f'FZnotchall-({notch0th_}_{attFWHM0_}_{notch1th_pang},{attStrFZ_}_{attFWHMFZ_}-{attStr_}-{attFWHM_}).tif',abs(FZnotchall[0]),run_parameter) 
    

    return  _U.ToNumpy(Wiener_sum),_U.ToNumpy(Wienek_sum)
    

#sk修正 c fai 影响
def getsk(opt,ck,cfai,Norders=7,methods=0,Debug=0):
    #修正C 、fai系数
    sk = np.zeros_like(ck)
    for i in range(3):
        for j in range(Norders): #7,5,3

            if j == 0:
                assert cfai[i,j,0]==1 and cfai[i,j,1]==0 

            if methods == 4:  #方法2
                sk[i*Norders+j] = ck[i*Norders+j]/( np.exp(1j* cfai[i,j,1] ))
            elif methods == 5:  #方法1
                sk[i*Norders+j] = ck[i*Norders+j]/( cfai[i,j,0] * np.exp(1j* cfai[i,j,1] ))

    if Debug == 1:
        for i in range(7):
            _U.save_tif(f'⑥sk—all方向-{i}.tif',abs(sk[i]))
    return sk

def removefai(opt,ck,cfai):
    #消除ck的初相
    assert len(ck.shape)==4
    sk = np.zeros_like(ck)
    nc,nz,nx,ny = ck.shape
    Norders = nc//opt.Nangles
    for i in range(opt.Nangles):
        for j in range(Norders):
            if j ==0:
                assert cfai[i,j,0]==1 and cfai[i,j,1]==0 
                
            sk[i*Norders+j] = ck[i*Norders+j]/( np.exp(1j* cfai[i,j,1] ))  #仅消除fai
    return sk
            
            

# 产生Apo
def geneate_qiezhi(opt,kuodapz,plong,qiezhi=4,lenz=2):
    # 0 
    if qiezhi==0:  #无切趾
        Apo = 1   #FZall*FM
        qiezhiname='non'
    if qiezhi ==1: #1apod  突然截断的切趾
        Apo = _WF.apod(opt)
        _U.save_tif("⑧ apod.tif",Apo)
        qiezhiname='apod'
    elif qiezhi ==2:   #2高斯切趾
        Apo = _WF.build_gaussian_3D(opt,0,1,kuodaxy=2,kuodaz=opt.kuodapz)
        _U.save_tif('⑧ gaosi.tif',Apo)
        qiezhiname = 'gaosi'
    elif qiezhi ==3:   #3bhs无轴向
        Apo = _WF.bhs_3D(opt,plong,opt.fc,kuodaz=kuodapz)
        _U.save_tif("⑧ bhs.tif",Apo)
        qiezhiname = 'bhs'

    elif qiezhi ==4:   #4 bhs有轴向
        Apo = _WF.bhs_3D_Z(opt,1.5*plong,opt.fc,kuodaz=kuodapz,Zjust=lenz)
        _U.save_tif("⑧ bhs_Z.tif",Apo)
        qiezhiname = 'bhs_Z'
    return Apo,qiezhiname



##############2023-12-22-XYD-4方法#############
def GetFZotf(opt,Irtest,order_flag,run_parameter,PGxy,PGz,direction=[0,1,2]):
    """
    读取opt.OTF,根据Irtest移动OTF
    """
    
    nx = opt.nx*opt.kuodaxy
    ny = opt.ny*opt.kuodaxy
    nz = opt.nz_re          
    Norders=opt.Norders                #5
    nc = opt.nc
    # OTFmask_Meth = run_parameter['Msum_maskMove']


    "1 OTF转psf"
    if run_parameter['mear'] ==1:
        pass
    else:
        opt.OTF = opt.OTF * opt.OTFmask3D  #OTFmask
        
    OTF3D = _U.ToTensor(opt.OTF).to(device=opt.device)  #取出0阶和1阶来
    
    PSF3D = tfftshift(_U.tifftn(OTF3D)).real         
    OTF3D = OTF3D.cpu()  
    # PSF3D_1th = tfftshift(_U.tifftn(OTF3D[1])).real
    

    if torch.is_complex(OTF3D):
        dtype = opt.tdtype_com
    else:
        dtype = opt.tdtype_real
    
    # print("dtype:",dtype)        
    FZotf = torch.zeros((nc,nz,nx,ny),dtype=dtype)     #
    FZotfmask = np.zeros((nc,nz,nx,ny),dtype=np.bool_)
    
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,order_flag)

    for i in direction:              #调试dirct 控制方向
        for j in (jiesu):  #调试
            if j == 0:
                # OTFmask3D_move , _ = _DP.otf_mask_PE(opt,Debug=0,PGxy = PGxy[i,j],PGz=[PGz[i,j]],flag=1)
                # OTFmask3D_move = _U.pad_or_cut(OTFmask3D_move,x=nx,y=ny,z=nz)
                # _U.save_tif_v21(fr'OTFmask3D_move-{i,j}.tif',OTFmask3D_move,run_parameter=run_parameter)
                FZotf[i*Norders+j] = OTF3D#*OTFmask3D_move
            else:
                "【移动矩阵】"
                filename = str(i)+str(j)+'.tif'
                tIrtest =_U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)            #CPU-GPU
 
                '移动-【OTF】'
                OTF_move = abs(_U.tfftn(tifftshift (PSF3D*tIrtest) )).cpu()           #GPU -> CPU
                OTF_move = _U.normalize(OTF_move) #归一化   分子没做归一化，分母也不做; 1.3改动 分子不敢做，但分母做 
                                                                                #;但从仿真结果来看 不做效果更好(感觉是OTF的问题)
                # _U.save_tif(f'OTF_move-{filename}-.tif',abs(OTF_move).numpy())
                # _U.save_tif(f'OTFmask3D_move-{filename}-.tif',OTFmask3D_move.numpy())     
                # 
                # 
                'mask'
                '生成式mask' 
                # OTFmask3D_move , _ = _DP.otf_mask_PE(opt,Debug=0,PGxy = PGxy[i,j],PGz=[PGz[i,j]],flag=1)
                # OTFmask3D_move = _U.pad_or_cut(OTFmask3D_move,x=nx,y=ny,z=nz)
                # OTFmask3D_move = torch.from_numpy(OTFmask3D_move).to(torch.bool)       
                # _U.save_tif_v21(fr'OTFmask3D_move-{i,j}.tif',OTFmask3D_move,run_parameter=run_parameter)
                'FZotf'
                FZotf[i*Norders+j] = (OTF_move)#*(OTFmask3D_move)
            # FZotfmask[i*Norders+j] = OTFmask3D_move

    return FZotf,FZotfmask


def GetFZ(opt,ck,Irtest,order_flag,run_parameter,direction=[0,1,2]):


    if opt.device != 'cpu':
        torch.cuda.empty_cache()

    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Norders=opt.Norders


    ck = _U.ToTensor(ck)
    FZall =torch.zeros((nc,nz,nx,ny),dtype=torch.complex64)                            #CPU


    "CK横向补零"
    # cr_pad = torch.zeros((nc,nz,nx,ny),dtype=opt.tdtype_com)                         #
    # ck_pad = _U.pad_or_cut(ck,x=nx,y=ny,z=nz)                                        #padding
                         
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        for j in jiesu:
            if j==0:
              ck_pad = _U.pad_or_cut(ck[i*opt.Norders+j],x=nx,y=ny,z=nz)   
              FZall[i*opt.Norders+j] = ck_pad
            else:
                '【更大】'
                ck_pad = _U.pad_or_cut(ck[i*opt.Norders+j],x=nx,y=ny,z=nz)
                
                '【移动矩阵】'
                filename = str(i)+str(j)
                tIrtest = _U.ToTensor(Irtest[i*opt.Norders+j]).to(device=opt.device)   #GPU

                '移动-【ck_pad】'
                ck_pad_GPU = ck_pad.to(device=opt.device)                     #CPU->GPU
                ck_pad_move = _U.tfftn(_U.tifftn(ck_pad_GPU)*tIrtest).cpu()                    #IFFT-移动-FFT  、 GPU->CPU

                FZall[i*opt.Norders+j] = ck_pad_move
    
    return FZall


def GetFZ_Joint(opt,ck,ck_02,Irtest,order_flag,run_parameter,direction=[0,1,2]):
    if opt.device != 'cpu':
        torch.cuda.empty_cache()

    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc                  #额外的0  1 2
    Norders=opt.Norders            #7


    ck = _U.ToTensor(ck)
    ck_02 = _U.ToTensor(ck_02)
    
    FZall =torch.zeros((nc,nz,nx,ny),dtype=torch.complex64)                            #CPU 3+4 
    FZall_02 = torch.zeros((3*3,nz,nx,ny),dtype=torch.complex64)

    "CK横向补零"
    # cr_pad = torch.zeros((nc,nz,nx,ny),dtype=opt.tdtype_com)                         #
    # ck_pad = _U.pad_or_cut(ck,x=nx,y=ny,z=nz)                                        #padding
                         
    "控制要重建的阶数"
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        for j in jiesu:
            if j==0:
                ck_pad = _U.pad_or_cut(ck[i*Norders+j],x=nx,y=ny,z=nz)   
                FZall[i*Norders+j] = ck_pad 
                del ck_pad
                
                ck_02_pad = _U.pad_or_cut(ck_02[i*3+0],x=nx,y=ny,z=nz)   
                FZall_02[i*3+0] = ck_02_pad 
                del ck_02_pad
                
            elif j <3:   #额外joint
                '【更大】'
                ck_pad = _U.pad_or_cut(ck[i*Norders+j],x=nx,y=ny,z=nz)
                ck_02_pad = _U.pad_or_cut(ck_02[i*3+j],x=nx,y=ny,z=nz)  
                
                '【移动矩阵】'
                filename = str(i)+str(j)
                tIrtest = _U.ToTensor(Irtest[i*Norders+j]).to(device=opt.device)   #GPU

                '移动-【ck_pad】'
                ck_pad_GPU = ck_pad.to(device=opt.device)                     #CPU->GPU
                ck_pad_move = _U.tfftn(_U.tifftn(ck_pad_GPU)*tIrtest).cpu()                    #IFFT-移动-FFT  、 GPU->CPU
                FZall[i*Norders+j] = ck_pad_move
                
                del ck_pad_GPU,ck_pad_move
                
                
                ck_02_pad_GPU = ck_02_pad.to(device=opt.device)                     #CPU->GPU
                ck_02_pad_move = _U.tfftn(_U.tifftn(ck_02_pad_GPU)*tIrtest).cpu()                    #IFFT-移动-FFT  、 GPU->CPU
                FZall_02[i*3+j] = ck_02_pad_move
                
                del ck_02_pad_GPU,ck_02_pad_move
                
            else:
                '【更大】'
                ck_pad = _U.pad_or_cut(ck[i*opt.Norders+j],x=nx,y=ny,z=nz)
                
                '【移动矩阵】'
                filename = str(i)+str(j)
                tIrtest = _U.ToTensor(Irtest[i*opt.Norders+j]).to(device=opt.device)   #GPU

                '移动-【ck_pad】'
                ck_pad_GPU = ck_pad.to(device=opt.device)                     #CPU->GPU
                ck_pad_move = _U.tfftn(_U.tifftn(ck_pad_GPU)*tIrtest).cpu()                    #IFFT-移动-FFT  、 GPU->CPU

                FZall[i*opt.Norders+j] = ck_pad_move
                del ck_pad
    return FZall,FZall_02


def GetWiener1(opt,FZall,FZotf,PGxy,PGz,cfai,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    "按照方向相加,mask、notch"
    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Norders=opt.Norders
    
    out_K_all =  np.zeros((nz,nx,ny),dtype=np.complex64)
    
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        FZ_dir = np.zeros((nz,nx,ny),dtype=np.complex64)
        for j in jiesu:
            'make notch'
            if j ==0:
                Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) 
            elif j!=0:
                if notch == 0:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  # 不加
                elif notch ==1 and j>0 and j <=2: # 2 order notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
                elif notch ==1 and j>2:           # 1 order notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU
                else:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  # 不加
                
            'combine cfai'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            
            'Combine into a larger spectrum'
            scale = Atteg1*(FZotf[i*Norders+j].conj())*np.conj(amp)
            temp_FZ = FZall[i*Norders+j]*scale
            FZ_dir  += temp_FZ
            
        'Apo'
        Apo = _WF.bhs_3D_Z(opt,opt.plongxs*opt.plong,opt.fcxs*opt.fc,kuodaz=opt.kuodapz, Zjust=opt.Apoz )
        out_K_all +=  1*FZ_dir
        
    return out_K_all


def GetWiener1_Joint(opt,FZall,FZall_02,FZotf,PGxy,PGz,cfai,cfai_02,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    "组合分子，按照方向相加,mask、notch"
    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Norders=opt.Norders
    
    out_K_all =  np.zeros((nz,nx,ny),dtype=np.complex64)
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        FZ_dir = np.zeros((nz,nx,ny),dtype=np.complex64)
        for j in jiesu:
            'make notch'
            if j ==0:
                Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['notch0th'],fwhm=run_parameter['attFWHM0'])) 
                # 4B notch 
                # Atteg11 = _WF.getAttenuationMatrix3D(0,opt.nx,opt.ny,PGxy[i,j,0],PGxy[i,j,1],2*(int(opt.fc_z)+1)+1,opt.fc)
                # Atteg11 = _U.pad_or_cut(Atteg11,x = nx,y = ny,z = nz)
                Atteg1 = Atteg1*1
            elif j!=0:
                if notch == 0:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  # 不加
                elif notch ==1 and j>0 and j <=2: # 2阶的notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStrFZ'],fwhm=run_parameter['attFWHMFZ'])) #CPU
                elif notch ==1 and j>2:           # 1阶的notch
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=run_parameter['attStr'],fwhm=run_parameter['attFWHM'])) #CPU
                else:
                    Atteg1 = (_WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],pz=PGz[i,j],Astr=0,fwhm=1))  # 不加
                
                
            'combine cfai'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            if j<3:
                amp_02 = cfai_02[i,j,0]*np.exp(1j*cfai_02[i,j,1])
            
            'Combine into a larger spectrum'
            scale = Atteg1* (FZotf[i*Norders+j].conj()) *np.conj(amp)
            temp_FZ = FZall[i*Norders+j]*scale
            FZ_dir  += temp_FZ
            
            if j<3 and j>=run_parameter['joints']:
                scale_02 = Atteg1* (FZotf[i*Norders+j].conj()) * np.conj(amp_02)
                temp_FZ_02 = FZall_02[i*3+j]*scale_02
                FZ_dir  += temp_FZ_02                                             

        # 'Apo'
        Apo = _WF.bhs_3D_Z(opt,opt.plongxs*opt.plong,opt.fcxs*opt.fc,kuodaz=opt.kuodapz, Zjust=opt.Apoz )
        out_K_all += 1*FZ_dir
        
    return out_K_all


def GetWiener2(opt,Wiener1,FZotf,PGxy,PGz,cfai,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    
    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Norders=opt.Norders
    
    '构建分母'
    Msum_all = w1
    jiesu = choose_jiesu(opt,order_flag)
    for i in direction:
        for j in jiesu:
            'amp'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            Msum_all += FZotf[i*Norders+j]**2 * abs(amp)**2 
            
    # _U.save_tif_v21(f'Msum_all-{w1}.tif',abs(Msum_all),run_parameter)
    return Msum_all#finral_k


def GetWiener2_Joint(opt,Wiener1,FZotf,PGxy,PGz,cfai,cfai_02,w1,order_flag,run_parameter,notch,direction=[0,1,2]):
    'combine Msum'
    nx = opt.nx*opt.kuodaxy                 #（应该为最终重建的大小）
    ny = opt.ny*opt.kuodaxy                 #（应该为最终重建的大小）
    nz = opt.nz_re                 #(opt.nz+2*opt.Zpadding)*opt.kuodapz #（应该为最终重建的大小）
    nc = opt.nc
    Norders=opt.Norders
    
    '构建分母'
    Msum_all = w1
    # jiesu = choose_jiesu(opt,order_flag)
    jiesu=[0,1,2,3,4,5,6]
    for i in direction:
        for j in jiesu:
            'amp'
            amp = cfai[i,j,0]*np.exp(1j*cfai[i,j,1])
            Msum_all += (FZotf[i*Norders+j]* (FZotf[i*Norders+j].conj()) ) * abs(amp)**2 
            
            if j>=run_parameter['joints'] and j<3:
                amp_02 = cfai_02[i,j,0]*np.exp(1j*cfai_02[i,j,1])
                Msum_all += (FZotf[i*Norders+j]* (FZotf[i*Norders+j].conj()) ) * abs(amp_02)**2

            
    '构建APO'

    # finral_k = Apo*Wiener1/Msum_all
    
    # filename = run_parameter['filename']
    # # if "001" in filename:
    # _U.save_tif_v21(f'Apo-GetWiener2-JOi-{opt.Apoz}-{filename}.tif',abs(Apo),run_parameter)

    # _U.save_tif_v21(f'Msum_all-{w1}.tif',abs(Msum_all),run_parameter)
    
    return Msum_all#finral_k








#save file
def Final_ifft(opt,fimage_k,chushi,w1,notch,flagg,direction,run_parameter,filename,p1_i_):
    """
    传入频域，逆变换
    """
    
    #当前阶数
    if flagg == 0:
        strt2 = '(0)'
    elif flagg == 1:
        strt2 = '(±1)'
    elif flagg == 2:
        strt2 = '(±2)'
    elif flagg == 3:
        strt2 = '(0±1)'
    elif flagg == 4:
        strt2 = '(0±2)'
    elif flagg == 5:
        strt2 = '(±1±2)'
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
    elif isinstance(w1, float) and w1 > 1e-3:     
        w2 = (np.round(w1,5))
    elif isinstance(w1, float):
        w2 = f"{w1:.0e}"
    else:
        w2 = w1
        
    #re_mode
    re_mode='Wie'
    
    #notch
    notch0th = _U.fill0(run_parameter['notch0th'])
    attFWHM0 = _U.fill0( run_parameter['attFWHM0'])
    notch1th_pang = _U.fill0(run_parameter['notch1st_pang'])
    
    Apo_Z = opt.Apoz  #记录Apo_Z宽度
    Apo_XY = opt.fcxs  
    PZ = run_parameter['pzzz'] #记录pz的使用

    ##############################文件处理##########################################
    temp_r = _U.ifftn((fimage_k))
    
    #crop the padding
    if opt.Zpadding!=0:
        temp_r_cai = temp_r[2*opt.Zpadding:-2*opt.Zpadding].real
    else:
        temp_r_cai = temp_r.real

    #Truncate negative values.
    temp_r_cai_j = np.copy(temp_r_cai)
    temp_r_cai_j[temp_r_cai_j<0] = 0  

    # #normal
    temp_r_cai_j_n = np.copy(temp_r_cai_j)
    temp_r_cai_j_n = temp_r_cai_j_n/np.max(temp_r_cai_j_n)
    
    # #转16bit
    # temp_r_cai_j_16 = temp_r_cai_j_n*65535

    ##################################################################
    #FFT
    fimage_k_cai = _U.fftn(temp_r_cai)

    version = opt.version
    data_nameall = run_parameter['data_name'] #2024.2.26
    # data_name = _U.find_dates(data_nameall)[0]
    data_name = data_nameall[-8:]

    dirname = run_parameter['dir_name']    #01
    movex = run_parameter['movex']
    movey = run_parameter['movey']
    # OTFname = run_parameter['OTFname_']
    note = run_parameter['note']
    
    #set name
    if notch==0:
        filenamerreal_cai = f'{dirname}_{version}-MP-W{w2}+({notch}-{notch0th}_{attFWHM0}-{notch1th_pang})-{dire}_{strt2}-AZ{Apo_Z}-PZ{PZ}-{re_mode}-{opt.Zpadding}【{filename}】.tif'
        filenamerreal_cai2 = f'MP-{data_nameall}-{dirname}-W{w2}+({notch0th})-AZ{Apo_Z}-{Apo_XY}-PZ{PZ}-{version}【{filename}】-{note}.tif'
    else:#notch参数
        attStrFZ = run_parameter['attStrFZ']  #2
        attFWHMFZ =  run_parameter['attFWHMFZ']
        attStr = run_parameter['attStr']      #1
        attFWHM =  run_parameter['attFWHM']

        filenamerreal_cai = f'{dirname}_{version}-MP-W{w2}+({notch0th}_{attFWHM0}-{notch1th_pang},{attStrFZ}_{attFWHMFZ}-{attStr}_{attFWHM})-{dire}_{strt2}-AZ{Apo_Z}-PZ{PZ}-{re_mode}-{opt.Zpadding}【{filename}】.tif'

        filenamerreal_cai2 = f'MP-{data_nameall}-{dirname}-W{w2}+({notch0th})-AZ{Apo_Z}-{Apo_XY}-PZ{PZ}-{version}【{filename}】{note}.tif'
        
    # filenamerreal_real = filenamerreal_cai[:-4]+'-real'+'.tif'
    filenamerreal_cai_j = filenamerreal_cai[:-4]+''+'.tif'
    # filenamerreal_cai_j_n = filenamerreal_cai_j[:-4]+'-nor'+'.tif'
    # filenamerreal_cai_j_n_XZ = filenamerreal_cai_j[:-4]+'_XZ'+'.tif'
    filenamek = '_Klog-'+filenamerreal_cai[6:]
    filenamek2_cai = '_Kcai-'+filenamerreal_cai[6:]
    filenamek2 = '_KV-'+filenamerreal_cai[6:]

    PGz_avg = run_parameter['PGz_avg']
    Dr_sub = run_parameter['Dr_sub'] 
    # _U.save_tif_v21(filenamerreal_cai_j_n,temp_r_cai_j_n,run_parameter,sub_filename=f'n-{PGz_avg}-'+Dr_sub,datatype=np.float32)
    
    
    if run_parameter['mear']==1:
        _U.save_tif_v21('留负'+filenamerreal_cai2,temp_r_cai,run_parameter,sub_filename=f'n-{PGz_avg}-'+Dr_sub)
    else:
        _U.save_tif_v21(filenamerreal_cai2,temp_r_cai_j,run_parameter,sub_filename=f'n-{PGz_avg}-'+Dr_sub)
    

    _U.save_tif_v22(filenamerreal_cai2,temp_r_cai_j,run_parameter,sub_filename=f'{PGz_avg}-'+Dr_sub,subsub_filename=f"{data_name}"+"_"+f"{dirname}"+"_"+f"{w2}"+f"({notch0th})",datatype=np.float32)
    

    # _U.save_tif_self3((run_parameter['data_path']), temp_r_cai_j, sub_filename='Result' , filename='Wiener_results.tif' ,min=0,max=65535)
    
    

    #保存插值后的结果
    import _4_space_inter as _4

    #先插值,再保存完整结果、XZ、YZ
    if run_parameter.get('chazhi', 0)==1 or run_parameter.get('XZ_save', 0)==1 or run_parameter.get('YZ_save', 0)==1:
        dx = opt.dx / 2
        dx2 = dx
        dz = opt.dz / 2
        dz2 = dz / 3
        before_shape = temp_r_cai_j.shape
        temp_r_cai_j_inter = _4.main(temp_r_cai_j, dx=dx, dx2=dx2, dz=dz, dz2=dz2)
        after_shape = temp_r_cai_j_inter.shape
        print("【插值信息】")
        print(f"shape: {before_shape} -> {after_shape}")
        print(f"XY pixel size: {dx:g} -> {dx2:g}")
        print(f"Z pixel size: {dz:g} -> {dz2:g}")
        
        if run_parameter.get('chazhi', 0)==1:
            # _U.save_tif_v21('chazhi-'+filenamerreal_cai2,temp_r_cai_j_inter,run_parameter,sub_filename=f'{PGz_avg}-'+Dr_sub,subsub_filename=f"{data_name}"+"_"+f"{dirname}"+"_"+f"{w2}"+f"({notch0th})",datatype=np.float32)
            _U.save_tif_v21("chazhi-"+filenamerreal_cai2,temp_r_cai_j_inter,run_parameter,sub_filename=f'n-{PGz_avg}-'+Dr_sub)

        
        if run_parameter.get('XZ_save', 0)==1:
            dataxz = np.swapaxes(temp_r_cai_j_inter,0,1)
            _U.save_tif_v21('xz-'+filenamerreal_cai,dataxz,run_parameter,sub_filename=f'n-{PGz_avg}-'+Dr_sub,datatype=np.uint16)
        
        if run_parameter.get('YZ_save', 0)==1:
            datazy = np.swapaxes(temp_r_cai_j_inter,0,2)
            _U.save_tif_v21('yz-'+filenamerreal_cai,datazy,run_parameter,sub_filename=f'n-{PGz_avg}-'+Dr_sub,datatype=np.uint16)
            

    #保存频域
    if p1_i_==0 or p1_i_==1: #多个时间点，只保存一个
        if opt.Zpadding!=0:
            _U.save_tif_v21(filenamek2,abs(fimage_k),run_parameter,sub_filename=f'K--{PGz_avg}-'+Dr_sub ,datatype=opt.dtype_real)
            _U.save_tif_v21(filenamek2_cai,abs(fimage_k_cai),run_parameter,sub_filename=f'K--{PGz_avg}-'+Dr_sub,datatype=opt.dtype_real)
            _U.save_tif_v22(filenamek2,abs(fimage_k),run_parameter,sub_filename=f'K--{PGz_avg}-'+Dr_sub ,datatype=opt.dtype_real)
            # fimage_k_AVG = _U.Pingjun_CUDA(abs(fimage_k))
            # _U.save_tif_v22('AVG_'+filenamek2,abs(fimage_k_AVG).astype(np.float32),run_parameter,sub_filename='K-'+ Dr_sub)  
            
        else:
            _U.save_tif_v21(filenamek2,abs(fimage_k),run_parameter,sub_filename=f'K--{PGz_avg}-'+Dr_sub,datatype=opt.dtype_real)
            _U.save_tif_v21(filenamek2_cai,abs(fimage_k_cai),run_parameter,sub_filename=f'K--{PGz_avg}-'+Dr_sub,datatype=opt.dtype_real)
            # fimage_k_AVG = _U.Pingjun_CUDA(abs(fimage_k))
            # _U.save_tif_v22('AVG_'+filenamek2,abs(fimage_k_AVG).astype(np.float32),run_parameter,sub_filename='K-'+ Dr_sub)  

    if run_parameter['save']==1:
        _U.save_tif_v21(filenamek2,abs(fimage_k),run_parameter,sub_filename=f'K--{PGz_avg}-'+Dr_sub,datatype=opt.dtype_real)


    #写入字典
    # _U.save_txt(f"{filenamerreal_cai[:-4]}.txt",run_parameter)
    return temp_r_cai,fimage_k_cai
