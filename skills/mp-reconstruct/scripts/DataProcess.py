import numpy as np
import torch
from tifffile import imread, imwrite
from pathlib import Path
from pyotf.otf import HanserPSF


import Utils as _U
import Para_Estimate as _PE
import WindowF as _WF
'''
raw data read
parameter setting
raw data preprocess
    deconvolution
raw data seperate
'''

def GetParams(opt,Dr_read,Debug=False): # uniform randomisation
    '''
    Optical-related parameters
    '''
    # opt = argparse.Namespace()
    
    #dtype
    opt.dtype_real = np.float32  #实数用32
    opt.dtype_com = np.complex64 #复数用64
    
    opt.tdtype_real = torch.float32
    opt.tdtype_com = torch.complex64
    
    #device
    opt.gpu_ids = 1 #
    if opt.gpu_ids and torch.cuda.is_available():
        opt.device = torch.device("cuda:0")
    else:
        opt.device = torch.device("cpu")

    ###外面设置##############
        #opt.flag_N  = 7       # imaging setting parameter
        #opt.Nshifts = 7       # phase shifts for each stripe
        # opt.Nangles = 3       # number of orientations of stripes

        # 60nm 30nmOTF仿真参数
        # opt.dx = 60#72    #  应该等于拍图时的piexlsize = 仿真OTF时的piexlsize
        # opt.dz = 200#200#240#133#120#130#65#130#32.5#35#38#90#120#100    # 拍图时的piexlsize = 仿真时间隔8层的piexlsize
        
        # opt.na = 1.27# 1.5      # NA
        # opt.n = 1.33 #1.518       # n
        # opt.namin = min(opt.na,opt.n)
        
        # opt.na_mask = 1.27 #1.5#1.27#2#1.7    #可以让maskna 稍微大一点
        # opt.n_mask =  1.33 #.518#1.33#2#1.718
        
        # # n*sinθ
        # opt.waterN = 1.33
        # opt.theta = 60* np.pi / 180.     
        
        # opt.lamdex = 488        #lamdex 入射光波长
        # opt.lamdem = 525       #lamdem 出射光波长
        # opt.Zpadding =  0    #轴向的Zpadding
    ######################
    
    #V3.7 版本 以下代码不再适用，放到外面手动给
    # opt.nc = opt.Nshifts*opt.Nangles    #这两个应该试一样的
    # opt.nz = Dr_read.shape[0]//opt.nc
    # opt.nx = Dr_read.shape[1]
    # opt.ny = Dr_read.shape[2]
    ############################
    
    opt.kuodaxy = 2       #SIM XY上一般都扩大两倍
    if opt.flag_N=="2D":     #
        opt.kuodapz = 1
        opt.Zpadding =  0 
        opt.nz_re = 1  
    elif opt.flag_N=="3D":
        opt.kuodapz = 1
        opt.nz_re = opt.kuodapz*(opt.nz+2*opt.Zpadding)  
        
    elif opt.flag_N=="MP":            
        opt.kuodapz = 2                                 #轴向有扩大
        opt.nz_re = opt.kuodapz*(opt.nz+2*opt.Zpadding)  
        

    opt.eps = np.finfo(float).eps
    opt.cyclesPerMicron = 1/((opt.nx*opt.dx)*1e-3)     #单位微米
    
    
    'OTF截止频率fc(每nm长度)'
    opt.fcXYnm = 2*opt.NAjs / opt.Wjs
    opt.fcZnm = opt.NAjs**2 / (2*opt.Wjs)
    #像素离散化
    opt.fc = opt.dx*opt.nx * opt.fcXYnm            #OTF横向半径
    opt.fc_z = opt.dz*(opt.nz+2*opt.Zpadding) * opt.fcZnm        #OTF轴向半径


    'Ir波矢量K （每nm长度）'
    opt.kxynm = 2*opt.NAzm/opt.Wzm
    opt.kznm  = opt.n*(1-np.cos(opt.theta)) / opt.Wzm
    #像素离散化
    opt.pxy = opt.dx*(opt.nx)* opt.kxynm     # Ir横向移动半径 #fc 与pxy 应该与生成psf时对标，
    opt.pz  = opt.dz*(opt.nz+2*opt.Zpadding)* opt.kznm      # 结构光深度由它自己范围决定


    #计算接收光路otf的截止频率 namin lamdem
    # opt.fc = opt.dx*(opt.nx)*2*opt.namin/opt.lamdem       #OTF横向半径
    # opt.fc_z = opt.dz*(opt.nz++2*opt.Zpadding)*opt.namin**2/(2*opt.lamdem)    #OTF轴向半径

    #计算照明光路的移动率 lamdex waterN theta
    # opt.pxy = opt.dx*(opt.nx)* 2*opt.waterN*np.sin(opt.theta)/opt.lamdex     # Ir横向移动半径 pxy = r/2 
    # d=opt.waterN-np.sqrt(opt.waterN**2 - (opt.waterN*np.sin(opt.theta))**2)    # Ir轴向移动半径 pz = rz/2
    # opt.pz = opt.dz*(opt.nz+2*opt.Zpadding)*d/opt.lamdex          # 为什么前面要乘2？因为我piexlsize没算对 应该给raw data数据中的，240
    
    #手动重新赋值fc，pxy大小
    opt.fc =  opt.fc #影响bhs 切趾函数
    opt.pxy = opt.pxy #(opt.nx/340) *92.8#opt.pxy
    opt.pz = opt.pz #2.62#3.25#((2*opt.Zpadding+opt.nz) /opt.nz  )*2.62
    
    opt.EEdge = opt.fc+opt.pxy 
    if Debug ==True:
        print('GetParams:opt.fc={},\n opt.fc_z={},\n opt.pxy={},\n opt.pz={}'.format(np.round(opt.fc,2),np.round(opt.fc_z,2),np.round(opt.pxy,2),np.round(opt.pz,2)))
    return opt



#load raw data

def load_RAW_DATA(pathfiname, Debug = False ):
    '''
    load raw data, adjust 方形, 
    '''

    '1、load raw data'
    Dr_read = imread(pathfiname)
    
    '2、adjust 为方形,'
    Dr_read = _U.image_reshape(Dr_read,dim=3)
    nz,nx,ny = Dr_read.shape
    if nx != ny:             #零填充
        maxx= max(nx,ny)
        print(f'load_RAW_DATA：0填充->{nz,nx,ny}->{nz,maxx,maxx}')
        Dr_read_F = _U.pad_or_cut(Dr_read,x=maxx,y=maxx,z=nz)
    else:
        print(f'load_RAW_DATA：no 填充->{nz,nx,ny}')
        Dr_read_F = Dr_read
    
    assert len(Dr_read_F.shape)==3                    #(nc ,nz,nx,ny)
    
    return Dr_read_F,nx,ny



def Transfer2NPD(opt,Dr_read):
    #transfer (phase,direction,nz)-> (nz,phase,direction)
    nc,nz,nx,ny = opt.Nangles*opt.Nshifts, opt.nz, opt.nx, opt.ny 
    assert len(Dr_read.shape)==3  #3D
    Dr_PAZ = Dr_read.reshape(nz,nc,nx,ny)  #PAZ  == （nz,np,na,...） = (nz,nc,...)
    Dr_ZPA =np.swapaxes( Dr_PAZ,0,1)       
    return Dr_ZPA

def Image_zpadding(opt,Dr_zpd,xishu=1):
    #2023年7月28日15:51:29 Z轴首尾补0
    nc,nz,nx,ny = opt.Nangles*opt.Nshifts, opt.nz+2*opt.Zpadding, opt.nx, opt.ny #注意这里的nz前后补了0层
    Zp = opt.Zpadding

    Dr_zpadding = np.zeros( (nc,nz,nx,ny),dtype=opt.dtype_real )
    print(f'Dr_zpadding={Dr_zpadding.shape},Dr_zpd={Dr_zpd.shape}')
    for i in range(nc): 
        Dr_zpadding[i,0:Zp]= xishu*Dr_zpd[i,0]  #0.77            or 这里用blur 模糊一下！ 
        Dr_zpadding[i,Zp:nz-Zp] = Dr_zpd[i]
        Dr_zpadding[i,nz-Zp:] = xishu*Dr_zpd[i,-1:None] 
    return Dr_zpadding


def Image_zpadding_WF(opt,Dr_zpd,xishu=1):
    nc,nz,nx,ny = opt.Nangles*opt.Nshifts, opt.nz+2*opt.Zpadding, opt.nx, opt.ny #注意这里的nz前后补了0层
    Zp = opt.Zpadding
    # WF = np.mean(Dr_zpd,axis=0)
    
    # Sxishu_s = np.array([0,0,0,1])
    # Sxishu_s = np.array([0.1,0.5,0.8,1])
    # Sxishu_s = np.array([0,0.5,0.87,1])
    # Sxishu_s = np.array([2,1.5,1.2,1])
    
    # Sxishu_s = np.repeat(np.array([1]),Zp)
    # Sxishu_e = np.flip(Sxishu_s)

    # S_image = Sxishu_s.reshape(opt.Zpadding,1,1) * np.repeat( WF[0].reshape(1,nx,ny),opt.Zpadding,axis=0 )
    # E_image = Sxishu_e.reshape(opt.Zpadding,1,1) * np.repeat( WF[-1].reshape(1,nx,ny),opt.Zpadding,axis=0)
    
    Dr_zpadding = np.zeros( (nc,nz,nx,ny),dtype=opt.dtype_real )
    print(f'Dr_zpadding={Dr_zpadding.shape},Dr_zpd={Dr_zpd.shape}')
    for i in range(nc): 
        Dr_zpadding[i,0:Zp]= 0#xishu*S_image   #
        Dr_zpadding[i,Zp:nz-Zp] = Dr_zpd[i]
        Dr_zpadding[i,nz-Zp:] =0# xishu*E_image#
        
    return Dr_zpadding



def Image_zpadding_replease(opt,Dr_zpd,xishu=1):
    nc,nz,nx,ny = opt.Nangles*opt.Nshifts, opt.nz+2*opt.Zpadding, opt.nx, opt.ny #注意这里的nz前后补了0层
    Zp = opt.Zpadding
    WF = np.mean(Dr_zpd,axis=0)
    
    Sxishu_s = np.array([0,0])
    # Sxishu_s = np.array([0.1,0.5,0.8,1])
    # Sxishu_s = np.array([0,0.5,0.87,1])
    # Sxishu_s = np.array([2,1.5,1.2,1])
    
    # Sxishu_s = np.repeat(np.array([1]),Zp)
    Sxishu_e = np.flip(Sxishu_s)

    # S_image = Sxishu_s.reshape(opt.Zpadding,1,1) * np.repeat( WF[0].reshape(1,nx,ny),opt.Zpadding,axis=0 )
    # E_image = Sxishu_e.reshape(opt.Zpadding,1,1) * np.repeat( WF[-1].reshape(1,nx,ny),opt.Zpadding,axis=0)
    
    Dr_zpadding = np.zeros( (nc,nz,nx,ny),dtype=opt.dtype_real )
    print(f'Dr_zpadding={Dr_zpadding.shape},Dr_zpd={Dr_zpd.shape}')
    for i in range(nc): 
        Dr_zpadding[i,0:3]= 0#xishu*S_image   #
        Dr_zpadding[i,3:7] = Dr_zpd[i,3:3+4]
        Dr_zpadding[i,7:] = 0#xishu*E_image#
        
    return Dr_zpadding


def Conv3d_2d(data3d,psf3d):
    '''
    input:  data3d、psf3d : (nz,nx,ny)  
    outpu： out2d_          (nx,ny) numpy
    '''
    from fft_conv_pytorch import FFTConv2d, fft_conv
    assert len(data3d.shape)==3
    assert len(psf3d.shape)==3
    
    
    '1.转为torch'
    tdata4d=_U.ToTensor(data3d).view(1,*data3d.shape).to(torch.float64)
    tpsf4d=_U.ToTensor(psf3d).view(1,*psf3d.shape).to(torch.float64)
    
    '2.fft-conv'
    _,nz,nx,ny = tpsf4d.shape
    with torch.no_grad():
        fft_conv = FFTConv2d(1, 1, nx,(nx//2,nx//2), bias=False)
        fft_conv.weight = torch.nn.Parameter(tpsf4d)
        out = fft_conv(tdata4d)

    
    #4D--2D
    out2d = out[0,0]
    
    '裁剪大小'
    nxx,nyy = out2d.shape
    # print(f'{out2d.shape}  and {out.shape}')
    # print(f'nxx={nxx}  and nx={nx}')
    if nxx>nx and nxx-nx==1:
        out2d_ = out2d[:-1,:-1]
    elif nxx==nx:
        out2d_=out2d
    else:
        
        raise(f'输出尺寸不一样,{nx}!={nxx}')
    
    try:
        out2d_ =out2d_.cpu()
    except:
        pass
    
    return _U.ToNumpy(out2d_)

def Image_zpadding_PSF(opt,Dr_zpd,xishu=0.5):
    #2023年7月28日15:51:29 Z轴首尾补0
    from pyotf.otf import HanserPSF

    psfs = HanserPSF( opt.Wjs, opt.NAjs, opt.n, opt.dx, int(opt.nx), opt.dz, opt.nz)  # 这里生成理想的OTF
    # Om = abs(psfs.OTFi)
    psf = abs(psfs.PSFi)
    
    nc,nz,nx,ny = opt.Nangles*opt.Nshifts, opt.nz+2*opt.Zpadding, opt.nx, opt.ny #注意这里的nz前后补了0层
    Zp = opt.Zpadding

    Dr_zpadding = np.zeros( (nc,nz,nx,ny),dtype=opt.dtype_real )
    # print(f'Dr_zpadding={Dr_zpadding.shape},Dr_zpd={Dr_zpd.shape}')
    
    import cv2
    kernel_size = (9,9)

    _U.save_tif('zpaddiing_psf.tif',psf)
    for i in range(nc): 
        # S_image = cv2.GaussianBlur(Dr_zpd[i,0],kernel_size,0)
        # E_image = cv2.GaussianBlur(Dr_zpd[i,-1:None],kernel_size,0)
        
        S_image = Dr_zpd[i,0]
        E_image = Dr_zpd[i,-1:None]
        
        Dr_zpadding[i,0:Zp]= xishu*S_image#Dr_zpd[i,0]#S_image  #0.77            or 这里用blur 模糊一下！ 
        Dr_zpadding[i,Zp:nz-Zp] = Dr_zpd[i]
        Dr_zpadding[i,nz-Zp:] = xishu*E_image#Dr_zpd[i,-1]#E_image#xishu*Dr_zpd[i,-1:None] 
    return Dr_zpadding


def Image_zpadding_PSF_flip(opt,Dr_zpd,xishu=0.5):
    #2023年7月28日15:51:29 Z轴首尾补0
    from pyotf.otf import HanserPSF

    psfs = HanserPSF( opt.Wjs, opt.NAjs, opt.n, opt.dx, int(opt.nx), opt.dz, opt.nz)  # 这里生成理想的OTF
    # Om = abs(psfs.OTFi)
    psf = abs(psfs.PSFi)
    
    nc,nz,nx,ny = opt.Nangles*opt.Nshifts, opt.nz+2*opt.Zpadding, opt.nx, opt.ny #注意这里的nz前后补了0层
    Zp = opt.Zpadding

    Dr_zpadding = np.zeros( (nc,nz,nx,ny),dtype=opt.dtype_real )
    # print(f'Dr_zpadding={Dr_zpadding.shape},Dr_zpd={Dr_zpd.shape}')
    
    import cv2
    kernel_size = (9,9)

    _U.save_tif('zpaddiing_psf.tif',psf)
    for i in range(nc): 
        # S_image = cv2.GaussianBlur(Dr_zpd[i,0],kernel_size,0)
        # E_image = cv2.GaussianBlur(Dr_zpd[i,-1:None],kernel_size,0)
        
        # S_image = Dr_zpd[i,0:]
        # E_image = Dr_zpd[i,-1:None]
        
        S_image = Dr_zpd[i,0:Zp]
        E_image = Dr_zpd[i,-Zp:None]
        
        S_image = np.flip(S_image,axis=0)
        E_image = np.flip(E_image,axis=0)
        
        Dr_zpadding[i,0:Zp]= xishu*S_image#Dr_zpd[i,0]#S_image  #0.77            or 这里用blur 模糊一下！ 
        Dr_zpadding[i,Zp:nz-Zp] = Dr_zpd[i]
        Dr_zpadding[i,nz-Zp:] = xishu*E_image#Dr_zpd[i,-1]#E_image#xishu*Dr_zpd[i,-1:None] 
    return Dr_zpadding


def windows(opt,data,nx_raw,ny_raw):
    # mask = _WF.mask2(opt,sigma=0.25)
    mask = _WF.mask2_self(nx_raw,ny_raw,sigma=0.25)
    nc,nz,nx,ny = data.shape
    if nx!=nx_raw or ny!=ny_raw:
        mask = _U.pad_or_cut(mask,x=nx,y=ny)
    print(data.shape,mask.shape)
    data = data*(mask**9)
    _U.save_tif('mask.tif',mask)
    return data

def bleach_adjust(data,zoffset=0,Nshift = 5):
    """
    data assumed taken with z changing fast, direction changing slowly
    V1.1 传入为4D
    """
    
    assert len(data.shape)==4 
    nc,nz,nx,ny = data.shape
    data5D = data.reshape(3,Nshift,nz,nx,ny)
    
    _,_,nz,nx,ny = data5D.shape
    sum_dir0_phase0 = np.zeros((nz))
    equalizez = 1
    
    for direction in range(3):  # 遍历三方向
        raw_images = data5D[direction]  # (5,nz,nx, ny)

        for z in range(nz):  #  nz=1  (5,nx, ny)
            sum_buffer = np.zeros(Nshift, dtype=np.float32)
            
            for pi in range(Nshift):  # (nx, ny, 1, phase=1)
                # Get sum value of current z image for individual 5 phases
                sum_buffer[pi] = np.sum(raw_images[pi,z + zoffset, :, :])

            #让后面值都以第一个方向第一个相位的，第一层 or 不同层为base （反正5个相位是一定相同的）
            if direction == 0:
                # Saving the sum value for direction=1, phase=1, when t=0
                sum_dir0_phase0[0*nz+ z] =  sum_buffer[0]
                
            if equalizez:
                # Sum value of image intensity under direction=1, phase=1, z=1
                ref = sum_dir0_phase0[0*nz]
            else:
                # Sum value of image intensity under direction=1, phase=1
                ref = sum_dir0_phase0[0*nz+z]

            for pi in range(5):
                ratio = ref / sum_buffer[pi]
                ratio = ratio/1.
                data5D[direction,pi,z+zoffset,:,:] *=  ratio

    data4D = data5D.reshape(3*Nshift,nz,nx,ny) 
    return data4D


#############################loadANDprocess_Dr################################################
def loadANDprocess_Dr(opt,p_i,run_parameter):
    "load Dr4D_ZPA"
    # 1、load raw data, 、adjust ,  #return 3D(nz,nx,ny)
    Dr3D_square,nx_raw,ny_raw = load_RAW_DATA(p_i)   #原始的nx_raw 与ny_raw 用于后面边缘加窗控制
    
    # 2 get opt
    opt = GetParams(opt,Dr3D_square)
    
    #'3、PAZ - > ZPA'           # 3 transfer (phase,direction,nz)-> (nz,phase,direction) # return 4D(nc,nz,nx,ny)
    Dr4D_ZPA = Transfer2NPD(opt,Dr3D_square) #这个只要交换两个轴就行 将Z放到前面就好             
    
    #'4、边缘加窗'
    Dr4D_ZPA = windows(opt,Dr4D_ZPA,nx_raw,ny_raw)
    
    # #5、漂白校正
    # if run_parameter['bleach']==1:
    #     print('进行光漂白校正')
    #     Dr4D_ZPA = bleach_adjust(Dr4D_ZPA,Nshift=opt.Nshifts)
    #     _U.save_tif_v21('Dr-mask-bleach.tif',Dr4D_ZPA,run_parameter)
    
    #crop 
    
    X_stare = run_parameter['movex']
    X_long = run_parameter['movex_long']
    
    Y_stare = run_parameter['movey']
    Y_long = run_parameter['movex_long']
    
    Dr4D_ZPA_crop =  (Dr4D_ZPA[:,:,X_stare:X_stare+X_long,Y_stare:Y_stare+Y_long])
    
    Dr4D_ZPA_crop_p = _U.pad_or_cut(Dr4D_ZPA_crop,opt.nx,opt.ny,opt.nz)
    
    
    Dr4D_ZPA = Dr4D_ZPA.astype(opt.dtype_real)
    Dr4D_ZPA_crop_p = Dr4D_ZPA_crop_p.astype(opt.dtype_real)
    assert len(Dr4D_ZPA.shape)==4
    return Dr4D_ZPA_crop_p,opt

    # return Dr4D_ZPA,opt



###############OTF/OTFmask###########################
# OTF 处理
# def load_OTF(opt,pathfiname,Debug=False,PSF=False):
#     '''
#     opt.OTF  这是一个4维的数据
#     '''
#     Om = imread(pathfiname) #这里是加载的OTF
    
#     if PSF:
#         psf = Om
#         Om = abs(_U.fftn(Om))
#         _U.save_tif('PSF2OTF.tif',Om)
#     else:
#         psf = np.real(np.fft.fftshift(_U.ifftn(Om)))
#         psf[psf<0]=0
#         _U.save_tif('opt.PSF.tif',psf)

#     #轴向补零
#     nz = opt.nz*opt.kuodapz   
    
#     #横向扩大两倍，轴向看情况
#     Om = _U.image_reshape(Om,dim=4)
#     Om = _U.pad_or_cut(Om,x=opt.nx,y=opt.ny,z=opt.nz)
#     Om_pad = _U.pad_or_cut(Om,x=2*opt.nx ,y=2*opt.ny,z = nz )



#     #2D OTF
#     assert len(Om_pad.shape) == 4
#     if nz!=1:
#         Om_pad_2D = Om_pad[0,nz//2] #np.sum(Om_pad,axis=1)#
#         Om_pad_2D = _U.image_reshape(Om_pad_2D,dim=4) 
#     else:
#         Om_pad_2D = np.copy(Om_pad)


#     #归一化 abs
#     Om_pad = Om_pad/np.max(Om_pad)  
    
#     if Debug==True:
#         print("DP-load_OTF、Om_pad.shape",Om_pad.shape)
#         _U.save_tif('DP-load_OTF、Om_pad.tif',abs(Om_pad))
#     return Om_pad,Om_pad_2D,Om,_U.image_reshape(psf,dim=3)

def load_OTF(opt,pathfiname,Debug=False):
    '''
    1、load OTF
    2、normalize
    3、get 2D
    '''
    
    '1、读入OTF文件'
    OTFload = imread(pathfiname)
    nz,nx,ny = OTFload.shape
    # assert nx==opt.nx and ny == opt.ny    #asure shape 
    
    
    PSF = (np.fft.fftshift(_U.ifftn(OTFload)))
    PSF[PSF<0]=0
    # # PSF = _U.pad_or_cut(PSF,opt.nx,opt.ny,opt.nz) #在PSF中补零or 裁剪 先取消
    # OTF = abs(_U.fftn(np.fft.fftshift(PSF))) #
    OTF = OTFload 
    
    '2、normalize'
    OTF_n = _U.normalize(OTF)
    # OTF_n = np.copy(OTF)
    
    # _U.save_tif(r'OTF.tif',OTF_n)
    
    '3、get 2D'
    OTF_n_3D = _U.image_reshape(OTF_n,dim=3)  #
    nz,nx,ny = OTF_n_3D.shape
    OTF_n_2D = OTF_n_3D[nz//2]
    
    return OTF_n_3D,OTF_n_2D,PSF      #3D/2D 实数/复数 




def load_OTF_3DSIM(opt,pathfiname,Debug=False):

    Om = imread(pathfiname) #这里是加载的OTF
    Om = _U.image_reshape(Om,dim=4)

    #横向扩大两倍
    Om = _U.pad_or_cut(Om,x=opt.nx,y=opt.ny,z=opt.nz)
    Om_pad = _U.pad_or_cut(Om,x=2*opt.nx ,y=2*opt.ny,z = opt.nz )

    #归一化
    if opt.nz!=1:
        Om_pads = []
        Om_pads.append( Om_pad[0]/np.max(Om_pad[0]) ) # 0阶
        Om_pads.append( Om_pad[0]/np.max(Om_pad[0]) )# 在真实OTF中，0阶采用2阶的OTF
        Om_pads.append( Om_pad[2]/np.max(Om_pad[2]) )# 2阶
    elif opt.nz==1:  #slice 情况
        Om_pads = []
        Om_pads.append( Om_pad[0]/np.max(Om_pad[0]) )
        Om_pads.append( Om_pad[0]/np.max(Om_pad[0]) )
        Om_pads.append( Om_pad[0]/np.max(Om_pad[0]) )

    Om_pads= np.array(Om_pads)

    assert len(Om_pads.shape) == 4
    
    #2DOTF
    if opt.nz!=1:
        Om_pad_2D = np.sum(Om_pads[0],axis=0)#轴向相加得到2DOTF      Om_pad[0,nz//2]
        Om_pad_2D = _U.image_reshape(Om_pad_2D,dim=4) 
    else:
        Om_pad_2D = np.copy(Om_pads[0])

    if Debug==True:
        _U.save_tif('①load_OTF(all)_norm.tif',abs(Om_pads))
        _U.save_tif('①load_OTF(0)_norm.tif',abs(Om_pads[0]))
        _U.save_tif('①load_OTF(2)_norm.tif',abs(Om_pads[2]))   
    
    return Om_pads,Om_pad_2D,Om


def create_OTF(opt,Debug=False):
    '''
    1、生成OTF文件
    2、归一化
    3、取中间层，得到2D OTF
    
    opt.OTF  这是一个4维的数据
    '''
    '调用函数得到原始psf otf（8，nx,ny）'
    psfs = HanserPSF(opt.lamdem,opt.na,opt.n,opt.dx,opt.nx,opt.dz,opt.nz) #这里生成理想的OTF
    Om = abs(psfs.OTFi)
    psf = abs(psfs.PSFi)
    
    '归一化 abs'
    Om_n = Om/np.max(Om)  
    Om_n = _U.image_reshape(Om_n,dim=3)  #
    

    '取中间层，得到2D OTF' 
    nz,nx,ny = Om_n.shape
    Om_n_2D = Om_n[nz//2]
    
    return Om_n, Om_n_2D,psf



#OTF mask -- 公式生成OTF的mask
# def otf_mask_formula(opt,nx,ny,nz,PGxy=[0,0],PGz=[0],Debug=0,kuodaxy=1):
#     """
#     7.2日调节，变为可以得到不扩大的OTF
#     只生成0阶
#     """
#     #增加GPU加速
#     #
#     try:
#         NA = opt.na_mask
#         refmed = opt.n_mask
#         NAmin = min(NA,refmed)
#         # print('用otf自己的na值')
#     except:
#         NA = opt.na
#         refmed = opt.n
#         NAmin = opt.namin
    
#     lamdex = opt.lamdex #激发波1 488
#     lamdem = opt.lamdem #发射波2 525
#     theta = opt.theta
#     dx = opt.dx          
#     dy = opt.dx
    
#     # padd_scale = 2        # 相当于外面扩大
#     # piexl_scale = 0.5     #

#     padd_scale = 1
#     piexl_scale=1       # 像素没扩大，不用减半
#     # print('初次生成WF大小的OTF')

#     #设置三维网格
#     if opt.nz==1:   #应对2D的情况
#         nz = 2
#     DqxSupport = 1/(nx*padd_scale)/(dx*piexl_scale)
#     DqySupport = 1/(ny*padd_scale)/(dx*piexl_scale)
#     DqzSupport = 1/nz/opt.dz
#     QXSupport = np.arange(np.ceil(-nx),np.ceil(nx))*DqxSupport;    #这里的网格没有扩大，保存原来的大小
#     QYSupport = np.arange(np.ceil(-ny),np.ceil(ny))*DqySupport;
#     # QZSupport = np.arange(np.ceil(-nz//2),np.ceil(nz//2))*DqzSupport;
#     QZSupport = np.arange(np.ceil(-nz),np.ceil(nz))*DqzSupport;
#     [qz,qx,qy] = np.meshgrid(QZSupport,QXSupport,QYSupport,indexing='ij')
    
    
#     #计算OTF fc fc_z------算OTF的大小
#     q0 = refmed/lamdem    #
#     NA1 = (NAmin/lamdem)   #
#     NB1 =  np.sqrt(q0**2-NA1**2)

#     qpar = np.sqrt( (qx-PGxy[0]*DqxSupport)**2+(qy-PGxy[1]*DqxSupport)**2)  #
#     axialcutoff = np.sqrt(q0**2-(qpar-NA1)**2)-NB1;
#     axialcutoff = (qpar<=2*NA1)*axialcutoff;

#     otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz-PGz[0]*DqzSupport))
#     otf_mask = (qpar<=2*NA1)*otf_mask
#     otf_mask = otf_mask.astype(np.uint8)

#     # if opt.flag_N==7: #3DSIM   ￥￥￥￥￥￥￥这种移动的方法不好￥￥￥￥￥￥￥￥
#     #     nz2,nx2,ny2  = otf_mask.shape
#     #     zv,xv,yv = _U.meshgridxy_self(nx2,ny2,nz=nz2)
#     #     pzz=2/nz
#     #     shiftmat = np.exp(0*xv+0*yv)*np.exp(2j*np.pi*pzz*zv) + np.exp(0*xv+0*yv)*np.exp(-2j*np.pi*pzz*zv)
#     #     otf_mask_three = np.zeros((3,nz2,nx2,ny2),dtype=np.float16)
#     #     otf_mask_three[0] = otf_mask
#     #     otf_mask_three[1] = otf_mask
#     #     otf_mask_three[2] = abs( _U.ifftn( _U.fftn(otf_mask)*shiftmat ))  
        
#     #     otf_mask_three[2][otf_mask_three[2]>0.7]=1 
#     #     _U.save_tif('otf_mask_three.tif',otf_mask_three.astype(bool))

    
#     if opt.nz==1:   #应对2D的情况
#         nz = 1
#         otf_mask = otf_mask[-1]
#     if Debug == 1:   
#         _U.save_tif('④otf_mask--mask0.tif',otf_mask.astype(bool))
#         pass
#     return _U.image_reshape(otf_mask,dim=3)  #返回一个3D的OTF


def otf_mask_formula(opt,nx,ny,nz,PGxy=[0,0],PGz=[0],Debug=0,kuodaxy=1):
    """
    7.2日调节，变为可以得到不扩大的OTF
    只生成0阶,nz=8层
    """
    #增加GPU加速
    #
    try:
        NA = opt.na_mask
        refmed = opt.n_mask
        NAmin = min(NA,refmed)
        # print('用otf自己的na值')
    except:
        NA = opt.na
        refmed = opt.n
        NAmin = opt.namin
    
    lamdex = opt.Wzm#opt.lamdex #激发波1 488
    lamdem = opt.Wjs#opt.lamdem #发射波2 525
    theta = opt.theta
    dx = opt.dx          
    dy = opt.dx
    
    # padd_scale = 2        # 相当于外面扩大
    # piexl_scale = 0.5     #

    padd_scale = 1
    piexl_scale=1       # 像素没扩大，不用减半
    # print('初次生成WF大小的OTF')

    #设置三维网格
    if opt.nz==1:   #应对2D的情况
        nz = 2
    DqxSupport = 1 / ((nx*padd_scale)*(dx*piexl_scale))        #每像素所占的长度
    DqySupport = 1 / ((ny*padd_scale)*(dx*piexl_scale))
    DqzSupport = 1 / (nz*opt.dz)                               #没有padding，实际的pz长度                              
    QXSupport = torch.arange(np.ceil(-nx),np.ceil(nx))*DqxSupport;    #这里的网格没有扩大，保存原来的大小
    QYSupport = torch.arange(np.ceil(-ny),np.ceil(ny))*DqySupport;    
    QZSupport = torch.arange(np.ceil(-nz/2),np.ceil(nz/2))*DqzSupport; # nz生成8层
    # QZSupport = torch.arange(np.ceil(-nz),np.ceil(nz))*DqzSupport;     #nz直接生成16层
    [qz,qx,qy] = torch.meshgrid(QZSupport,QXSupport,QYSupport)
    
    
    #计算OTF fc fc_z------算OTF的大小
    q0 = refmed/lamdem    #
    NA1 = (NAmin/lamdem)   #
    NB1 =  np.sqrt(q0**2-NA1**2)

    qpar = torch.sqrt( (qx-PGxy[0]*DqxSupport)**2+(qy-PGxy[1]*DqxSupport)**2)  #
    axialcutoff = torch.sqrt(q0**2-(qpar-NA1)**2)-NB1;
    axialcutoff = (qpar<=2*NA1)*axialcutoff;

    otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz-PGz[0]*DqzSupport))
    otf_mask = (qpar<=2*NA1)*otf_mask
    otf_mask = otf_mask.to(torch.uint8)
    otf_mask = otf_mask.numpy()

    # if opt.flag_N==7: #3DSIM   ￥￥￥￥￥￥￥这种移动的方法不好￥￥￥￥￥￥￥￥
    #     nz2,nx2,ny2  = otf_mask.shape
    #     zv,xv,yv = _U.meshgridxy_self(nx2,ny2,nz=nz2)
    #     pzz=2/nz
    #     shiftmat = np.exp(0*xv+0*yv)*np.exp(2j*np.pi*pzz*zv) + np.exp(0*xv+0*yv)*np.exp(-2j*np.pi*pzz*zv)
    #     otf_mask_three = np.zeros((3,nz2,nx2,ny2),dtype=np.float16)
    #     otf_mask_three[0] = otf_mask
    #     otf_mask_three[1] = otf_mask
    #     otf_mask_three[2] = abs( _U.ifftn( _U.fftn(otf_mask)*shiftmat ))  
        
    #     otf_mask_three[2][otf_mask_three[2]>0.7]=1 
    #     _U.save_tif('otf_mask_three.tif',otf_mask_three.astype(bool))
    # _U.save_tif('④otf_mask--mask1.tif',otf_mask.astype(bool))
    
    if opt.nz==1:   #应对2D的情况
        nz,nx,ny = otf_mask.shape
        maxindex=0
        maxpp = np.mean(otf_mask[maxindex])
        for i in range(nz):
            tempp = np.mean(otf_mask[i])
            if maxpp<tempp:
                maxindex = i
                maxpp = np.mean(otf_mask[maxindex])
        otf_mask = otf_mask[maxindex]

    if Debug == 1:   
        _U.save_tif('④otf_mask--mask0.tif',otf_mask.astype(bool))
        pass
    return _U.image_reshape(otf_mask,dim=3)  #返回一个3D的OTF

def otf_mask_formula2(opt,nx,ny,nz,PGxy=[0,0],PGz=[0],Debug=0,kuodaxy=1):
    """
    7.2日调节，变为可以得到不扩大的OTF
    只生成0阶
    nz=16层，用于OTFmask移动那里
    """
    #增加GPU加速
    #
    try:
        NA = opt.na_mask
        refmed = opt.n_mask
        NAmin = min(NA,refmed)
        # print('用otf自己的na值')
    except:
        NA = opt.na
        refmed = opt.n
        NAmin = opt.namin
    
    lamdex = opt.Wzm#opt.lamdex #激发波1 488
    lamdem = opt.Wjs#opt.lamdem #发射波2 525
    theta = opt.theta
    dx = opt.dx          
    dy = opt.dx
    
    # padd_scale = 2        # 相当于外面扩大
    # piexl_scale = 0.5     #

    padd_scale = 1
    piexl_scale=1       # 像素没扩大，不用减半
    # print('初次生成WF大小的OTF')

    #设置三维网格
    if opt.nz==1:   #应对2D的情况
        nz = 2
    DqxSupport = 1 / ((nx*padd_scale)*(dx*piexl_scale))        #每像素所占的长度
    DqySupport = 1 / ((ny*padd_scale)*(dx*piexl_scale))
    DqzSupport = 1 / (nz*opt.dz)                               #没有padding，实际的pz长度                              
    QXSupport = torch.arange(np.ceil(-nx),np.ceil(nx))*DqxSupport;    #这里的网格没有扩大，保存原来的大小
    QYSupport = torch.arange(np.ceil(-ny),np.ceil(ny))*DqySupport;    
    # QZSupport = torch.arange(np.ceil(-nz//2),np.ceil(nz//2))*DqzSupport; # nz生成8层
    QZSupport = torch.arange(np.ceil(-nz),np.ceil(nz))*DqzSupport;     #nz直接生成16层
    [qz,qx,qy] = torch.meshgrid(QZSupport,QXSupport,QYSupport)
    
    
    #计算OTF fc fc_z------算OTF的大小
    q0 = refmed/lamdem    #
    NA1 = (NAmin/lamdem)   #
    NB1 =  np.sqrt(q0**2-NA1**2)

    qpar = torch.sqrt( (qx-PGxy[0]*DqxSupport)**2+(qy-PGxy[1]*DqxSupport)**2)  #
    axialcutoff = torch.sqrt(q0**2-(qpar-NA1)**2)-NB1;
    axialcutoff = (qpar<=2*NA1)*axialcutoff;

    otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz-PGz[0]*DqzSupport))
    otf_mask = (qpar<=2*NA1)*otf_mask
    otf_mask = otf_mask.to(torch.uint8)
    otf_mask = otf_mask.numpy()

    # if opt.flag_N==7: #3DSIM   ￥￥￥￥￥￥￥这种移动的方法不好￥￥￥￥￥￥￥￥
    #     nz2,nx2,ny2  = otf_mask.shape
    #     zv,xv,yv = _U.meshgridxy_self(nx2,ny2,nz=nz2)
    #     pzz=2/nz
    #     shiftmat = np.exp(0*xv+0*yv)*np.exp(2j*np.pi*pzz*zv) + np.exp(0*xv+0*yv)*np.exp(-2j*np.pi*pzz*zv)
    #     otf_mask_three = np.zeros((3,nz2,nx2,ny2),dtype=np.float16)
    #     otf_mask_three[0] = otf_mask
    #     otf_mask_three[1] = otf_mask
    #     otf_mask_three[2] = abs( _U.ifftn( _U.fftn(otf_mask)*shiftmat ))  
        
    #     otf_mask_three[2][otf_mask_three[2]>0.7]=1 
    #     _U.save_tif('otf_mask_three.tif',otf_mask_three.astype(bool))
    # _U.save_tif('④otf_mask--mask1.tif',otf_mask.astype(bool))
    
    if opt.nz==1:   #应对2D的情况
        nz,nx,ny = otf_mask.shape
        maxindex=0
        maxpp = np.mean(otf_mask[maxindex])
        for i in range(nz):
            tempp = np.mean(otf_mask[i])
            if maxpp<tempp:
                maxindex = i
                maxpp = np.mean(otf_mask[maxindex])
        otf_mask = otf_mask[maxindex]

    if Debug == 1:   
        _U.save_tif('④otf_mask--mask0.tif',otf_mask.astype(bool))
        pass
    return _U.image_reshape(otf_mask,dim=3)  #返回一个3D的OTF


def otf_mask_formula3(opt,nx,ny,nz,PGxy=[0,0],PGz=[0],piexl_scale=1,piexl_scalez=1,Debug=0):
    """
    7.2日调节，变为可以得到不扩大的OTF
    只生成0阶,
    piexl_scale 控制
    """
    #增加GPU加速
    #
    try:
        NA = opt.na_mask
        refmed = opt.n_mask
        NAmin = min(NA,refmed)
        # print('用otf自己的na值')
    except:
        NA = opt.na
        refmed = opt.n
        NAmin = opt.namin
    
    lamdex = opt.Wzm  #激发波1 488
    lamdem = opt.Wjs  #发射波2 525
    theta = opt.theta
    dx = opt.dx          
    dy = opt.dx
        
    # print('初次生成WF大小的OTF')

    #设置三维网格
    DqxSupport = 1 / ((nx)*(dx*piexl_scale))        #每像素所占的长度
    DqySupport = 1 / ((ny)*(dx*piexl_scale))
    DqzSupport = 1 / ( nz*opt.dz*piexl_scalez )                               #没有padding，实际的pz长度                              
    QXSupport = torch.arange(np.ceil(-nx/2),np.ceil(nx/2))*DqxSupport;    #这里的网格没有扩大，保存原来的大小
    QYSupport = torch.arange(np.ceil(-ny/2),np.ceil(ny/2))*DqySupport;  
    QZSupport = torch.arange(np.ceil(-nz/2),np.ceil(nz/2))*DqzSupport; # nz生成8、9层

    # QZSupport = torch.arange(np.ceil(-nz),np.ceil(nz))*DqzSupport;     #nz直接生成16层
    [qz,qx,qy] = torch.meshgrid(QZSupport,QXSupport,QYSupport)
    
    
    #计算OTF fc fc_z------算OTF的大小
    q0 = refmed/lamdem    #
    NA1 = (NAmin/lamdem)   #
    NB1 =  np.sqrt(q0**2-NA1**2)

    qpar = torch.sqrt( (qx-PGxy[0]*DqxSupport)**2+(qy-PGxy[1]*DqxSupport)**2)  #
    axialcutoff = torch.sqrt(q0**2-(qpar-NA1)**2)-NB1;
    axialcutoff = (qpar<=2*NA1)*axialcutoff;

    otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz-PGz[0]*DqzSupport))
    otf_mask = (qpar<=2*NA1)*otf_mask
    otf_mask = otf_mask.to(torch.uint8)
    otf_mask = otf_mask.numpy()
    
    if opt.nz==1:   #应对2D的情况
        nz,nx,ny = otf_mask.shape
        maxindex=0
        maxpp = np.mean(otf_mask[maxindex])
        for i in range(nz):
            tempp = np.mean(otf_mask[i])
            if maxpp<tempp:
                maxindex = i
                maxpp = np.mean(otf_mask[maxindex])
        otf_mask = otf_mask[maxindex]

    return _U.image_reshape(otf_mask,dim=3)  #返回一个3D的OTF



def otf_mask_3d_2d(opt,Debug,kuodapz=2,PGxy=[0,0],PGz=[0]):
    if kuodapz==1:
        #对OTF进行mask
        OTFmask3D_raw = otf_mask_formula(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz, PGxy=PGxy,PGz=PGz,Debug=Debug,kuodaxy=1) #这里
    elif kuodapz ==2:
        #移动不同位置的mask
        OTFmask3D_raw = otf_mask_formula2(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz, PGxy=PGxy,PGz=PGz,Debug=Debug,kuodaxy=1) #这里
    # _U.save_tif('OTFmask3D原始大小.tif',OTFmask3D_raw.astype(bool))
    
    #让OTFmask对称(删除掉了左边那个)
    OTFmask3D = np.zeros_like(OTFmask3D_raw)

    nz,nx,ny = OTFmask3D_raw.shape
    nz_med = nz//2
    OTFmask3D[nz_med:] = OTFmask3D_raw[nz_med:]
    for ii in range(nz_med):
        OTFmask3D[ii+1] = OTFmask3D_raw[-(ii+1)]  #1 -1;   2 -2  ;3  -3;  
     
    #      ！！！！ 把OTF的扩大放到外面来
    # _U.save_tif('OTFmask3D.tif',OTFmask3D)
    OTFmask3D = _U.pad_or_cut(OTFmask3D,x=2*opt.nx,y=2*opt.ny,z = opt.kuodapz*opt.nz) #

    #插值OTFmask3D
    if opt.Zpadding!=0:
        #轴向z插值
        nx,ny,nz =  opt.kuodaxy*opt.nx, opt.kuodaxy*opt.ny, opt.nz_re
        OTFmask3D = _U.Interpolate(OTFmask3D.astype(np.float32),nx,ny,nz)
        OTFmask3D[OTFmask3D < 0.1] = 0
        OTFmask3D[OTFmask3D != 0 ] = 1
        OTFmask3D = OTFmask3D.astype(np.uint8)

    #让OTFMASK更小一点，腐蚀一圈
    # OTFmask3D = _PE.Cir_Erode_3D(OTFmask3D,size=3 )
    
    #OTFmask2D
    OTFmask2D = OTFmask3D[opt.nz_re//2]
    
    # if Debug==1:
    #     _U.save_tif('optOTFmask3D.tif',OTFmask3D.astype(bool))
    #     _U.save_tif('optOTFmask2D.tif',OTFmask2D.astype(bool))
    return OTFmask3D,OTFmask2D,OTFmask3D_raw


def otf_mask_3d_2d_3DSIM(opt,Debug,PGxy=[0,0],PGz=[0]):
    
    OTFmask3D = np.zeros((3, opt.nz, 2*opt.nx, 2*opt.ny))
    OTFmask3D[0] = otf_mask_formula(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz//2,Debug=Debug,kuodaxy=1)
    temp1  = otf_mask_formula(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz//2, PGxy=PGxy,PGz=PGz,Debug=Debug,kuodaxy=1)
    temp2  = otf_mask_formula(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz//2, PGxy=PGxy,PGz=[-PGz[0]],Debug=Debug,kuodaxy=1)
    OTFmask3D[1] = OTFmask3D[0]
    OTFmask3D[2] = temp1 + temp2

    OTFmask2D = OTFmask3D[0,opt.nz_re//2]

    if opt.nz==1:
        OTFmask3D[0,0] = OTFmask2D
        OTFmask3D[1,0] = OTFmask2D
        OTFmask3D[2,0] = OTFmask2D


    if Debug==1:
        _U.save_tif('optOTFmask3D.tif',OTFmask3D.astype(bool))
        _U.save_tif('optOTFmask2D.tif',OTFmask2D.astype(bool))
    return OTFmask3D,OTFmask2D


def otf_mask_PE(opt,Debug,PGxy=[0,0],PGz=[0],flag=0):
    #OTFmask 大小为 单层 xy 没扩大，nz的大小
    # OTFmask3D_raw = otf_mask_formula3(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz, PGxy=PGxy,PGz=PGz,piexl_scale=1,piexl_scalez= 1 ,Debug=Debug) #这里
    
    if flag ==0: #没移动
        OTFmask3D_raw = otf_mask_formula3(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz+2*opt.Zpadding, PGxy=PGxy,PGz=PGz,piexl_scale=1,piexl_scalez= 1 ,Debug=Debug) #这里
    else:
        OTFmask3D_raw = otf_mask_formula3(opt,nx=2*opt.nx, ny=2*opt.ny, nz=opt.nz_re, PGxy=PGxy,PGz=PGz,piexl_scale=0.5,piexl_scalez= 0.5 ,Debug=Debug) #这里


    #前后平均一下
    if PGz[0]==0:
        OTFmask3D_raw = duicehng(OTFmask3D_raw)
    
    #OTFmask2D
    nz,_,_ = OTFmask3D_raw.shape
    OTFmask2D = OTFmask3D_raw[nz//2]

    #膨胀一下
    

    return OTFmask3D_raw,OTFmask2D

def duicehng(data):
    #从后往前对称
    assert(len(data.shape)==3)
    nz,_,_ = data.shape
    nz_med = nz//2
    data_out = np.zeros_like(data)
    for i in range(nz-nz_med): #6,12
        data_out[nz_med-i] = np.copy(data[nz_med+i]) 
        data_out[nz_med+i] = np.copy(data[nz_med+i]) 
    return data_out




def get_pzp1(opt,PGxy):
    #计算照明光路，通过估计的1阶PGxy，计算theta，计算pz
    #返回PGzz (3,7)
    
    n = opt.n
    Wzm = opt.Wzm    #照明光路
    dx = opt.dx/2      #SIM piexlsize
    nx = opt.nx*2      #SIM nx
    dz = opt.dz/2
    
    P_long = np.sqrt( np.sum(PGxy**2,axis=2) )  #(3,7)计算每方向每阶的p长度
    PGzz = np.zeros((3,7))
    PGxy1 = np.zeros((3,1))
    for i in range(3):                          #遍历3个方向
        p1 =np.mean(P_long[i,3:7])              #每个方向1阶 平均长度    
        
        theta = np.arcsin(p1/dx/nx *Wzm/n)*180/np.pi #每个方向的theta角度
        
        kznm =  n*(1-np.cos(theta*np.pi/180)) / Wzm  #通过theta 计算实际pz
        pz = dz*2*(opt.nz+2*opt.Zpadding)*kznm

        '保存记录'
        # print(f'第{i+1}方向theta = {theta},pz={pz}')
        PGzz[i,[3,5]] = -pz
        PGzz[i,[4,6]] = pz
        PGxy1[i]   = p1 
        
    return PGzz,PGxy1


def drawDr(Dr,Nshift=8,number=[0,1,2,3,4]):
    #此时的Dr为（24，nz,nx,ny）
    
    Drout = np.zeros_like(Dr[:3*len(number)])
    for i in range(3):
        ii = (i)*Nshift + np.array(number) #[0,1,2,3,4]
        # print('ii',ii)
        Drout[i*len(number):(i+1)*len(number)] = np.copy(Dr[[ii]]) #连续的间隔number 
    return Drout




# def generatemask(opt,isdebug):
#     import DataProcess as _DP
#     # # OTFmask3D/2D
#     #衰减形态
#     opt.na_mask = 1.21
#     OTFmask3D_121,OTFmask2D = otf_mask_PE(opt,Debug=isdebug['loa']) #
    
#     opt.na_mask = 1.24
#     OTFmask3D_124,OTFmask2D = otf_mask_PE(opt,Debug=isdebug['loa']) #

#     opt.na_mask = 1.27
#     OTFmask3D_127,OTFmask2D = otf_mask_PE(opt,Debug=isdebug['loa']) #


#     opt.na_mask = 1.3
#     OTFmask3D_13,_ = otf_mask_PE(opt,Debug=isdebug['loa']) #
    
#     opt.na_mask = 1.35
#     OTFmask3D_135,_ = otf_mask_PE(opt,Debug=isdebug['loa']) #
    
    
#     _U.save_tif(r'OTFmask3D_13.tif',OTFmask3D_13)
#     _U.save_tif(r'OTFmask3D_124.tif',OTFmask3D_124)
#     _U.save_tif(r'OTFmask3D_127.tif',OTFmask3D_127)
#     _U.save_tif(r'OTFmask3D_121.tif',OTFmask3D_121)
#     maskotf1 = OTFmask3D_13 -OTFmask3D_121
#     maskotf11 = OTFmask3D_13 -OTFmask3D_121
    
#     maskotf13_124 = OTFmask3D_13-OTFmask3D_124
#     maskotf13_1241 = OTFmask3D_13-OTFmask3D_124
    
#     maskotf13_127 = OTFmask3D_13-OTFmask3D_127
#     maskotf13_1271 = OTFmask3D_13-OTFmask3D_127
    
#     maskotf135_1271 = OTFmask3D_135-OTFmask3D_13
    
#     _U.save_tif(r'maskotf11.tif',maskotf11)
#     _U.save_tif(r'maskotf13_1241.tif',maskotf13_1241)
#     _U.save_tif(r'maskotf13_1271.tif',maskotf13_1271)
#     _U.save_tif(r'maskotf13_all.tif',maskotf11+maskotf13_1241+maskotf13_1271+maskotf135_1271)
    
#     center_x, center_y = OTFmask3D_13.shape[1] // 2, OTFmask3D_13.shape[2] // 2
#     y, x = np.ogrid[:OTFmask3D_13.shape[1], :OTFmask3D_13.shape[2]]
#     distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
#     maskotf11[int(maskotf11.shape[0]/2-opt.pz/2):int(maskotf11.shape[0]/2+opt.pz/2)+2,distance_from_center<10] = 0
#     maskotf1 = maskotf1-maskotf11
    
#     maskotf13_1241[int(maskotf11.shape[0]/2-opt.pz/2):int(maskotf11.shape[0]/2+opt.pz/2)+2,distance_from_center<10] = 0
#     maskotf2 = maskotf13_124-maskotf13_1241
    
#     maskotf13_1271[int(maskotf11.shape[0]/2-opt.pz/2):int(maskotf11.shape[0]/2+opt.pz/2)+2,distance_from_center<10] = 0
#     maskotf3 = maskotf13_127-maskotf13_1271
    
#     maskotf1+= maskotf2+maskotf3
#     maskotf1[maskotf1>1] = 1
    
#     _U.save_tif(r'maskotf1-删除中间10半径.tif',maskotf11)
#     _U.save_tif(r'maskotf11-保留里面一圈.tif',maskotf1)
    
#     # _U.save_tif(r'maskotf11-保留里面一圈.tif',maskotf1)

#     maskotf1 = maskotf1.astype(np.float32)
#     for i in range(maskotf1.shape[0]//2):
#         maskotf1[i] = maskotf1[i]*(0.5**i)
#     _U.save_tif(r'maskotf111.tif',maskotf1)
    
#     OTFmask3D_121 = OTFmask3D_121/1.0
#     OTFmask3D = OTFmask3D_121 + maskotf1 
    
#     maskotf44 = OTFmask3D_13-OTFmask3D_121
#     maskotf44_all = np.copy(maskotf44)
#     _U.save_tif(r'maskotf44.tif',maskotf44)
    
#     maskotf44[int(maskotf44.shape[0]/2-opt.pz/1):int(maskotf44.shape[0]/2+opt.pz/1)+2,distance_from_center<30] = 0
#     _U.save_tif(r'maskotf44-删除中间30半径.tif',maskotf44)
    
#     maskotf44_all += maskotf44
#     maskotf44_all[maskotf44_all<=1] = 0
#     _U.save_tif(r'maskotf1-.tif',maskotf1)

#     _U.save_tif(r'maskotf44_all.tif',maskotf44_all)
#     OTFmask3D += maskotf44_all
    
#     OTFmask3D[OTFmask3D.shape[0]//2+1:] = np.flip(OTFmask3D[1:OTFmask3D.shape[0]//2], axis=0)
#     OTFmask3D[OTFmask3D>1] =1

#     return OTFmask3D, OTFmask2D


#原始mask 衰减
# def generatemask(opt,isdebug):
#     import DataProcess as _DP
#     # # OTFmask3D/2D
#     #衰减形态
#     opt.na_mask = 1.21
#     OTFmask3D_121,OTFmask2D = otf_mask_PE(opt,Debug=isdebug['loa']) #

#     opt.na_mask = 1.3
#     OTFmask3D_13,_ = otf_mask_PE(opt,Debug=isdebug['loa']) #
    
#     maskotf1 = OTFmask3D_13 -OTFmask3D_121
#     maskotf11 = OTFmask3D_13 -OTFmask3D_121

    
#     center_x, center_y = OTFmask3D_13.shape[1] // 2, OTFmask3D_13.shape[2] // 2
#     y, x = np.ogrid[:OTFmask3D_13.shape[1], :OTFmask3D_13.shape[2]]
#     distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
#     maskotf11[int(maskotf11.shape[0]/2-opt.pz/2):int(maskotf11.shape[0]/2+opt.pz/2)+2,distance_from_center<10] = 0
#     maskotf1 = maskotf1-maskotf11

#     maskotf1 = maskotf1.astype(np.float32)
#     for i in range(maskotf1.shape[0]//2):
#         maskotf1[i] = maskotf1[i]*(0.5**i)
#     _U.save_tif(r'maskotf111.tif',maskotf1)
    
#     OTFmask3D_121 = OTFmask3D_121/1.0
#     OTFmask3D = OTFmask3D_121 + maskotf1 
    
#     maskotf44 = OTFmask3D_13-OTFmask3D_121
#     maskotf44_all = np.copy(maskotf44)
#     _U.save_tif(r'maskotf44.tif',maskotf44)


#     maskotf44[int(maskotf44.shape[0]/2-opt.pz/1):int(maskotf44.shape[0]/2+opt.pz/1)+2,distance_from_center<30] = 0
    
#     maskotf44_all += maskotf44
#     maskotf44_all[maskotf44_all<=1] = 0
#     _U.save_tif(r'maskotf1-.tif',maskotf1)

#     _U.save_tif(r'maskotf44_all.tif',maskotf44_all)
#     OTFmask3D += maskotf44_all
    
#     OTFmask3D[OTFmask3D.shape[0]//2+1:] = np.flip(OTFmask3D[1:OTFmask3D.shape[0]//2], axis=0)
#     OTFmask3D[OTFmask3D>1] =1

#     return OTFmask3D, OTFmask2D


def generatemask(opt,isdebug):
    import DataProcess as _DP
    # # OTFmask3D/2D
    #衰减形态
    opt.na_mask = 1.21
    OTFmask3D_121,OTFmask2D = otf_mask_PE(opt,Debug=isdebug['loa']) #

    opt.na_mask = 1.3
    OTFmask3D_13,_ = otf_mask_PE(opt,Debug=isdebug['loa']) #
    
    # _U.save_tif(r'OTFmask3D_13.tif',OTFmask3D_13)
    _U.save_tif(r'OTFmask3D_121.tif',OTFmask3D_121)
    
    # maskotf1 = OTFmask3D_13 -OTFmask3D_121
    # maskotf11 = OTFmask3D_13 -OTFmask3D_121
    
    # _U.save_tif(r'maskotf11.tif',maskotf11)
    # _U.save_tif(r'maskotf13_all.tif',maskotf11)
    
    center_x, center_y = OTFmask3D_13.shape[1] // 2, OTFmask3D_13.shape[2] // 2
    y, x = np.ogrid[:OTFmask3D_13.shape[1], :OTFmask3D_13.shape[2]]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    maskotf44 = OTFmask3D_13-OTFmask3D_121
    # maskotf44_all = np.copy(maskotf44)
    # _U.save_tif(r'maskotf44.tif',maskotf44)
    
    maskotf44[int(maskotf44.shape[0]/2-opt.pz/1):int(maskotf44.shape[0]/2+opt.pz/1)+2,distance_from_center<30] = 0
    # _U.save_tif(r'maskotf44-删除中间30半径.tif',maskotf44)
    
    # maskotf44_all += maskotf44
    # maskotf44_all[maskotf44_all<=1] = 0
    # _U.save_tif(r'maskotf1-.tif',maskotf1)

    # _U.save_tif(r'maskotf44_all.tif',maskotf44_all)
    OTFmask3D = OTFmask3D_121+maskotf44
    # _U.save_tif(r'OTFmask3D.tif',OTFmask3D)
    
    OTFmask3D[OTFmask3D.shape[0]//2+1:] = np.flip(OTFmask3D[1:OTFmask3D.shape[0]//2], axis=0)
    OTFmask3D[OTFmask3D>1] =1

    return OTFmask3D, OTFmask2D