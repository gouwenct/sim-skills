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
    Optical-related parameters----3D-SIM
    '''
    #opt = argparse.Namespace()
    
    #dtype
    opt.dtype_real = np.float32
    opt.dtype_com = np.complex64
    
    opt.tdtype_real = torch.float32
    opt.tdtype_com = torch.complex64
    
    #device
    opt.gpu_ids = 1 #
    if opt.gpu_ids and torch.cuda.is_available():
        opt.device = torch.device("cuda:0")
        print('use GPU')
    else:
        opt.device = torch.device("cpu")
        print('warning,no GPU,only CPU')
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
        
        # opt.opt.Wzm = 488        #lamdex 入射光波长
        # opt.lamdem = 525       #lamdem 出射光波长
        # opt.Zpadding =  0    #轴向的Zpadding
    ######################
    

    opt.nc = opt.Nshifts*opt.Nangles    #这两个应该试一样的
    opt.nz = Dr_read.shape[0]//opt.nc
    opt.nx = Dr_read.shape[1]
    opt.ny = Dr_read.shape[2]
    
    opt.kuodaxy = 2       #之所以要超分，就是扩大了两倍
    if opt.flag_N!=7 or opt.Nshifts!=7:     #2D、传统
        opt.kuodapz = 1
        # opt.Zpadding =  0                  #轴向的Zpadding
        opt.nz_re = opt.kuodapz*(opt.nz+2*opt.Zpadding)  #用于记录重建过程中的pz
            
    elif opt.flag_N==7 and opt.Nshifts==7:              #多层
        opt.kuodapz = 2 
        opt.nz_re = opt.kuodapz*(opt.nz+2*opt.Zpadding)  #用于记录重建过程中的pz
        print('opt.nz_re:',opt.nz_re)



    opt.eps = np.finfo(float).eps
    opt.cyclesPerMicron = 1/((opt.nx*opt.dx)*1e-3)     #单位微米
    
    
    #计算接收光路otf的截止频率 namin lamdem
    opt.fc = opt.dx*(opt.nx)*2*opt.namin/opt.lamdem       #OTF横向半径
    opt.fc_z = opt.dz*(opt.nz+2*opt.Zpadding)*opt.namin**2/(2*opt.lamdem)    #OTF轴向半径

    #计算照明光路的移动率 opt.Wzm  waterN theta
    opt.pxy = opt.dx*(opt.nx)* 2*opt.waterN*np.sin(opt.theta)/opt.Wzm     # Ir横向移动半径 pxy = r/2 
    d=opt.waterN-np.sqrt(opt.waterN**2 - (opt.waterN*np.sin(opt.theta))**2)    # Ir轴向移动半径 pz = rz/2
    opt.pz = opt.dz*(opt.nz+2*opt.Zpadding)*d/opt.Wzm          # 为什么前面要乘2？因为我piexlsize没算对 应该给raw data数据中的，240
    
    #手动重新赋值fc，pxy大小
    opt.fc =  opt.fc #影响bhs 切趾函数
    opt.pxy = opt.pxy #(opt.nx/340) *92.8#opt.pxy
    opt.pz = opt.pz #2.62#3.25#((2*opt.Zpadding+opt.nz) /opt.nz  )*2.62
    
    opt.EEdge = opt.fc+opt.pxy 

    if Debug ==True:
        print('opt.fc={},\n opt.fc_z={},\n opt.pxy={},\n opt.pz={}'.format(np.round(opt.fc,2),np.round(opt.fc_z,2),np.round(opt.pxy,2),np.round(opt.pz,2)))
    return opt



#load raw data
def load_RAW_DATA(opt,pathfiname, run_parameter,Debug = False ):
    '''
    load raw data,
    transfer (phase,direction,nz)-> (nz,phase,direction)'''

    #1 load raw data
    Dr_read = imread(pathfiname)
    Dr_read = Dr_read.astype(np.float32)  #因为3D的之前没处理过 所以这里需要处理

    #自从加了数据预处理 10.16日之后的不在这里减背景了
    
    # bg = imread(r'E:\4TB图片\python 实验数据\仪器配准实验数据\3.beads球差\BG\AVG_cam1_background_MMStack_Pos0.ome-0.tif')
    print(Dr_read.dtype)
    # bg = bg.astype(float32)
    Dr_read = Dr_read

    #2 adjust opt
    Dr_read = _U.image_reshape(Dr_read,dim=3)
    nz,_,_ = Dr_read.shape

    _U.save_tif(f'Dr_read-cut-{Dr_read.shape}.tif',Dr_read)

    #2.2 变矩形为方形
    nz,nx,ny = Dr_read.shape
    print(nz,nx,ny)
    if nx != ny:             #零填充
        maxx= max(nx,ny)
        print(f'load_RAW_DATA：0填充->{nz,nx,ny}->{nz,maxx,maxx}')
        Dr_read_F = _U.pad_or_cut(Dr_read,x=maxx,y=maxx,z=nz)
    else:
        print(f'load_RAW_DATA：no 填充->{nz,nx,ny}')
        Dr_read_F = Dr_read
    opt = GetParams(opt,Dr_read_F)
    # opt.nz = Dr_read.shape[0]//opt.nc
    # opt.nx = Dr_read.shape[1]
    # opt.ny = Dr_read.shape[2]
    # print(f'opt.nc= {opt.nc},opt.nz={opt.nz},opt.nx={opt.nx}')

        
    #3 transfer (phase,direction,nz)-> (nz,phase,direction)
    Dr_zpd = Transfer2NPD(opt,Dr_read_F)
    assert len(Dr_zpd.shape)==4                    #(nc ,nz,nx,ny)
    
    # 4 增加这一个Zpadding ，当Zpadding=0时是一样的操作  8月新增 #放到外面去
    if opt.Zpadding!=0:
        Dr_zpd = Image_zpadding(opt,Dr_zpd)
   
   
    #5 裁剪
    X_stare = run_parameter['movex']
    X_long = run_parameter['movex_long']
    Y_stare = run_parameter['movey']
    Y_long = run_parameter['movex_long']
    
    Dr4D_ZPA_crop =  (Dr_zpd[:,:,X_stare:X_stare+X_long,Y_stare:Y_stare+Y_long])
    # Dr4D_ZPA_crop_p = _U.pad_or_cut(Dr4D_ZPA_crop,462,462,1)
    Dr4D_ZPA_crop_p = _U.pad_or_cut(Dr4D_ZPA_crop,opt.nx,opt.ny,opt.nz)
    
    # return Dr_zpd.astype(opt.dtype_real),opt
    return Dr4D_ZPA_crop_p.astype(opt.dtype_real),opt

def windows(opt,data):
    mask = _WF.mask2(opt,sigma=0.25)
    data = data*(mask**9)
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


def Transfer2NPD(opt,Dr_read):
    #transfer (phase,direction,nz)-> (nz,phase,direction)
    Dr_zpd = np.zeros_like(Dr_read).reshape(opt.nc,opt.nz,opt.nx,opt.ny) #4D!!!
    for i in range(opt.nc):
        Dr_zpd[i] = Dr_read[i:(opt.nc*opt.nz):opt.nc,:,:]  #最终Dr 变量为我们所要的raw data
    Dr_zpd = _U.image_reshape(Dr_zpd,dim=4).reshape(opt.nc,opt.nz,opt.nx,opt.ny)
    return Dr_zpd


def Image_zpadding(opt,Dr_zpd):
    nc,nz,nx,ny = opt.nc,opt.nz+2*opt.Zpadding,opt.nx,opt.ny 
    Zp = opt.Zpadding

    Dr_zpadding = np.zeros( (nc,nz,nx,ny),dtype=opt.dtype_real )
    print(f'Dr_zpadding={Dr_zpadding.shape},Dr_zpd={Dr_zpd.shape}')
    for i in range(nc): 
        Dr_zpadding[i,0:Zp]= 0.77*Dr_zpd[i,0]  #0.77            or 这里用blur 模糊一下！ 
        Dr_zpadding[i,Zp:nz-Zp] = Dr_zpd[i]
        Dr_zpadding[i,nz-Zp:] = 0.77*Dr_zpd[i,-1:None] 
    return Dr_zpadding

def Image_zpadding3(opt,Dr_zpd):
    nc,nz,nx,ny = opt.nc,opt.nz+2*opt.Zpadding,opt.nx,opt.ny #nz = 8+2x2 =12
    Zp = opt.Zpadding    #2

    Dr_zpadding = np.zeros( (nc,nz,nx,ny),dtype=opt.dtype_real )
    print(f'Dr_zpadding={Dr_zpadding.shape},Dr_zpd={Dr_zpd.shape}')
    for i in range(nc): 
        Dr_zpadding[i,Zp-1] = 0.6*Dr_zpd[i,0]
        Dr_zpadding[i,Zp:nz-Zp] = Dr_zpd[i]
        Dr_zpadding[i,nz-Zp] = 0.6*Dr_zpd[i,-1:None]
        for j in range(1,opt.Zpadding): #1:0        2:0,1   3:0,1,2
            Dr_zpadding[i,Zp-1-j] = 0.6*Dr_zpadding[i,Zp-j]   
            Dr_zpadding[i,nz-Zp+j] = 0.6*Dr_zpadding[i,nz-Zp+j-1] # 14  -3  -2  -1 
    return Dr_zpadding


def Image_zpadding_blur(opt,Dr_zpd):
    nc,nz,nx,ny = opt.nc,opt.nz+2*opt.Zpadding,opt.nx,opt.ny 
    Zp = opt.Zpadding
    OTF2D = _U.pad_or_cut(opt.OTF2D, x=opt.nx,y =opt.ny)
    Dr_zpadding = np.zeros( (nc,nz,nx,ny),dtype=opt.dtype_real )
    print(f'Dr_zpadding={Dr_zpadding.shape},Dr_zpd={Dr_zpd.shape}')
    for i in range(nc): 
        Dr_zpadding[i,0:Zp]= abs(_U.ifftn(_U.fftn(Dr_zpd[i,0])*OTF2D))  #0.77            or 这里用blur 模糊一下！ 
        # Dr_zpadding[i,Zp:nz-Zp] = Dr_zpd[i,Zp:nz-Zp]   #结合版
        Dr_zpadding[i,Zp:nz-Zp] = Dr_zpd[i,:]          #单独使用
        Dr_zpadding[i,nz-Zp:] = abs(_U.ifftn(_U.fftn(Dr_zpd[i,-1:None])*OTF2D)) 
    return Dr_zpadding


###############OTF/OTFmask###########################
# OTF 处理
def load_OTF(opt,pathfiname,Debug=False,PSF=False):
    '''
    opt.OTF  这是一个4维的数据 #传入的应该是一个nz,nx,ny 数据,   需要变为3,nz,nx,ny
    '''
    Om = imread(pathfiname) #这里是加载的OTF
    
    
        #维度调整为（3，nz，nx，ny）
    # if len(Om.shape)==4:
    #     nc,nz,nx,ny = Om.shape
    # elif len(Om.shape) ==3:
    #     ncnz,nx,ny = Om.shape
    #     nc = 3
    #     nz = ncnz//nc
    # elif len(Om.shape) ==2: #2D slice
    #     nx,ny = Om.shape
    #     Om = _U.image_reshape(Om,dim=4)
    #     Om = np.repeat(Om,3,axis=0)
    #     nz = 1
        
    if PSF:
        psf = Om
        Om = abs(_U.fftn(Om))
        _U.save_tif('PSF2OTF.tif',Om)
    else:
        psf = np.real(np.fft.fftshift(_U.ifftn(Om)))
        psf[psf<0]=0
        _U.save_tif('opt.PSF.tif',psf)

    #轴向补零
    nz = opt.nz*opt.kuodapz   
    

    #横向扩大两倍，轴向看情况
    Om = _U.image_reshape(Om,dim=4)
    Om = _U.pad_or_cut(Om,x=opt.nx,y=opt.ny,z=opt.nz)
    Om_pad = _U.pad_or_cut(Om,x=2*opt.nx ,y=2*opt.ny,z = nz )

    #归一化 abs
    Om_pad = Om_pad/np.max(Om_pad)  

    #2D OTF
    assert len(Om_pad.shape) == 4
    if nz!=1:
        Om_pad_2D = Om_pad[0,nz//2] #np.sum(Om_pad,axis=1)#
        # Om_pad_2D = _U.image_reshape(Om_pad_2D,dim=4) 
    else:
        Om_pad_2D = np.copy(Om_pad)
        Om_pad_2D = _U.image_reshape(Om_pad_2D,dim=3)

    #有zpadding的话 要插值
    # if opt.Zpadding!=0 and opt.flag_N==7 and opt.Nshifts==7:       #轴向插值
    #     _,nzz,nxx,nyy = Om_pad.shape
    #     Om_pad = _U.Interpolate(opt,Om_pad,nx=nxx,ny=nyy,nz=opt.nz_re)
    
    #nc 循环三次
    
    Om_pad = np.repeat(Om_pad,3,axis=0)
    Om_pad_2D = np.repeat(Om_pad_2D,3,axis=0)
    
    if Debug==True:
        print("DP-load_OTF、Om_pad.shape",Om_pad.shape)
        _U.save_tif('DP-load_OTF、Om_pad.tif',abs(Om_pad))
        
    return Om_pad,Om_pad_2D,Om,_U.image_reshape(psf,dim=3)

def load_OTF_3DSIM(opt,pathfiname,Debug=False):
    '''
    主要是这里传入的为3*nz的 OTF
    '''
    Om = imread(pathfiname) #这里是加载的OTF
    
    #维度调整为（3，nz，nx，ny）
    if len(Om.shape)==4:
        nc,nz,nx,ny = Om.shape
    elif len(Om.shape) ==3:
        ncnz,nx,ny = Om.shape
        nc = 3
        nz = ncnz//nc
    elif len(Om.shape) ==2: #2D slice
        nx,ny = Om.shape
        Om = _U.image_reshape(Om,dim=4)
        Om = np.repeat(Om,3,axis=0)
        nz = 1
    Om = Om.reshape(3,-1,nx,ny)

    #调整OTF深度
    # Om_just = []
    # if opt.nz!=nz: #DR与OTF深度不统一,将4D OTF变到
    #     print(fr'修改OTF，nz={nz}->{opt.nz}')
    #     for i in range(3):
    #         PSF_temp = np.fft.fftshift(_U.ifftn(Om[i])).real
    #         PSF_temp = _U.pad_or_cut(PSF_temp,x=nx,y=ny,z=opt.nz) #裁剪为Dr深度
    #         OTF_temp = abs(_U.fftn(np.fft.ifftshift(PSF_temp)))
    #         Om_just.append(OTF_temp)
    #     Om_just = np.array(Om_just)
        
    #     Om = np.copy(Om_just)
    #     del Om_just
    
    #轴向补零 3D 一般不需要
    nz = opt.nz*1#  opt.kuodapz

    #横向扩大两倍
    Om = _U.pad_or_cut(Om,x=opt.nx,y=opt.ny,z=opt.nz)  #
    Om_pad = _U.pad_or_cut(Om,x=2*opt.nx ,y=2*opt.ny,z = nz )

    #2DOTF
    Om_pad_2D = np.zeros((3,2*opt.nx,2*opt.ny),dtype = opt.dtype_real)    #3D数据
    med = opt.nz//2
    Om_pad_2D[0] = Om_pad[0,nz//2]  #取中间层进行参数估计 不是进行重建（所以不是求和）
    Om_pad_2D[1] = Om_pad[0,nz//2]  #np.sum(Om_pads[0],axis=0)#轴向相加得到2DOTF      Om_pad[0,nz//2]
    Om_pad_2D[2] = Om_pad[2,nz//2]

    # #有zpadding的话 要插值
    # if opt.Zpadding!=0 and opt.nz!=0:       #轴向插值
    #     #变化到空域补0
    #     _,nzz,nxx,nyy = Om_pad.shape
    #     Om_pad = _U.Interpolate(Om_pad,nx=nxx,ny=nyy,nz=opt.nz_re)

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
    assert len(Om_pads.shape) == 4  # 
    
    if Debug==True:
        _U.save_tif('①load_OTF(all)_norm.tif',abs(Om_pads))
        _U.save_tif('①load_OTF(0)_norm.tif',abs(Om_pads[0]))
        _U.save_tif('①load_OTF(2)_norm.tif',abs(Om_pads[2]))   
    
    return Om_pads,Om_pad_2D,Om

#调用pyOTF产生OTF
def create_OTF(opt,Debug=False):
    '''
    opt.OTF  这是一个4维的数据
    '''
    psfs = HanserPSF(opt.lamdem,opt.na,opt.n,opt.dx,opt.nx,opt.dz,opt.nz) #这里生成理想的OTF
    Om = abs(psfs.OTFi)
    psf = abs(psfs.PSFi)
    _U.save_tif('opt.PSF.tif',psf)
                  #HanserPSF(525, 0.85, 1.0, 140, 64)

    #轴向补零
    nz = opt.nz*opt.kuodapz

    #归一化 abs
    Om = Om/np.max(Om)  
    
    #横向扩大两倍，轴向看情况
    Om = _U.image_reshape(Om,dim=4) 
    Om = _U.pad_or_cut(Om,x=opt.nx,y=opt.ny,z=opt.nz)
    Om_pad = _U.pad_or_cut(Om,x=2*opt.nx ,y=2*opt.ny,z = nz )

    #2D OTF
    assert len(Om_pad.shape) == 4
    if nz!=1:
        Om_pad_2D = np.sum(Om_pad,axis=1)  #m_pad[0,nz//2]
        Om_pad_2D = _U.image_reshape(Om_pad_2D,dim=4) 
    else:
        Om_pad_2D = np.copy(Om_pad)

    #有zpadding的话 要插值
    if opt.Zpadding!=0 and opt.flag_N==7 and opt.Nshifts==7:       #轴向插值
        _,nzz,nxx,nyy = Om_pad.shape
        Om_pad = _U.Interpolate(Om_pad,nx=nxx,ny=nyy,nz=opt.nz_re)
    
    
    if Debug==True:
        print("DP-load_OTF、Om_pad.shape",Om_pad.shape)
        _U.save_tif('DP-load_OTF、Om_pad.tif',abs(Om_pad))
    return Om_pad,Om_pad_2D,Om,_U.image_reshape(psf,dim=3) 



# def otf_mask5(opt,otf,PGxy,mask_move =False,paddingxy=1):
#     ''' 
#     仅仅只需要获得OTFmask_cha 就可以吧
#     input：nz，nx，ny OTF大小
#     output： nz+2*Zpadding，2nx，2ny
#     '''
#     opt.Wzm = opt.opt.Wzm #激发波1 488
#     lamdem = opt.lamdem #发射波2 512
#     try:
#         NA = opt.na_mask
#         refmed = opt.n_mask
#         NAmin = min(NA,refmed)
#         print('用otf自己的na值')
#     except:
#         NA = opt.na
#         refmed = opt.n
#         NAmin = opt.namin


#     theta = opt.theta
#     #大小
#     nz,nx,ny = otf.shape  #这个OTF还没有扩大
#     padd_scale = 2        #扩大否
#     #间距
#     dx = opt.dx   #因为像素扩大，所以减半
#     dy = opt.dx

#     piexl_scale = 0.5        #间隔扩大倍数

#     if kuodaxy == 0:
#         nx = nx//2
#         ny = ny//2
#         padd_scale=2   
#         piexl_scale=1       #这样生成的OTF才不需要扩大
#         print('xy 不扩大')

#     #设置三维网格
#     DqxSupport = 1/(nx*padd_scale)/(dx*piexl_scale)
#     DqySupport = 1/(ny*padd_scale)/(dx*piexl_scale)
#     DqzSupport = 1/nz/opt.dz
#     QXSupport = np.arange(np.ceil(-nx),np.ceil(nx))*DqxSupport;    #这里的网格没有扩大，保存原来的大小
#     QYSupport = np.arange(np.ceil(-ny),np.ceil(ny))*DqySupport;
#     QZSupport = np.arange(np.ceil(-nz//2),np.ceil(nz//2))*DqzSupport;
#     [qz,qx,qy] = np.meshgrid(QZSupport,QXSupport,QYSupport,indexing='ij')

#     #计算PZ----axialshift
#     pixelsizexy = piexl_scale*dx
#     p11=np.zeros((3))
#     for i in range(3):
#         qpixelsize = 1/(pixelsizexy*nx*padd_scale)     #这个nx是移频的像素长度
#         qvector_x = PGxy[i,1,0] * qpixelsize 
#         qvector_y = PGxy[i,1,1] * qpixelsize 
        
#         p11[i] = 1/(np.sqrt(qvector_x**2+qvector_y**2))
#     p11_mean = np.mean(p11) #横向上的pxy   ##############!要除2 才对

#     q0ex = refmed/opt.Wzm
#     axialshift = q0ex-np.sqrt(q0ex**2-1/p11_mean**2);

#     #计算OTF fc fc_z------算OTF的大小
#     q0 = refmed/lamdem    #
#     NA1 = (NAmin/lamdem)   #
#     NB1 =  np.sqrt(q0**2-NA1**2)

#     qpar = np.sqrt(qx**2+qy**2)  #这里没有搞
#     axialcutoff = np.sqrt(q0**2-(qpar-NA1)**2)-NB1;
#     axialcutoff = (qpar<=2*NA1)*axialcutoff;

#     qvector = PGxy * qpixelsize
#     OTFmask_cha = np.zeros((opt.nc, nz+2*opt.Zpadding,2*nx,2*ny),np.float)
#     for i in range(3):
#         for j in range(5):
#             qpar = np.sqrt( (qx-qvector[i,j,0])**2 + (qy-qvector[i,j,1])**2 )
#             axialcutoff = np.sqrt(q0**2-(qpar-NA1)**2)-NB1
#             axialcutoff =  (qpar<=2*NA1)*axialcutoff
#             if j==1 or j==2:  #±1阶
#                 otf_mask1 = axialcutoff+DqzSupport/2 >= abs(qz-axialshift)
#                 otf_mask2 = axialcutoff+DqzSupport/2 >= abs(qz+axialshift)
#                 otf_mask = otf_mask1|otf_mask2   #取或的操纵
#             else:
#                 otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz))

#             otf_mask = (qpar<=2*NA1)*otf_mask
#             otf_mask = otf_mask.astype(np.float32)
#             otf_mask[otf_mask<0.1] = 0
#             otf_mask[otf_mask!=0] = 1
            
#             #插值插到2n 大小
#             # otf_mask_temp = _U.Interpolate(opt,otf_mask.astype(np.float32),nx,ny,nz)
#             OTFmask_cha[i*opt.Nshifts+j] =  otf_mask#otf_mask_temp
#     OTFmask_cha[OTFmask_cha<0.1]=0
#     OTFmask_cha[OTFmask_cha !=0] =1
#     OTFmask_cha = OTFmask_cha.astype(np.bool)
#     _U.save_tif('OTFmask_cha.tif',OTFmask_cha)

#     return OTFmask_cha.astype(np.bool)  



#3D-SIM
def otf_mask3D_SIM(opt,nx,ny,nz,PGxy,paddingxy=2):
    ''' 
    仅仅只需要获得OTFmask_cha 就可以吧
    input：nz，nx，ny OTF大小 
           kuodaxy =1 保持原来大小， =2 扩大2倍
    output： 根据参数kuodaxy 来决定， OTF形状不变，仅仅是padding不padding的问题
    '''

    try:
        NA = opt.na_mask
        refmed = opt.n_mask
        NAmin = min(NA,refmed)
        print('用otf自己的na值')
    except:
        NA = opt.na
        refmed = opt.n
        NAmin = opt.namin

    Wzm = opt.Wzm #激发波1 488
    lamdem = opt.lamdem #发射波2 525
    theta = opt.theta
    dx = opt.dx   #因为像素扩大，所以减半
    dy = opt.dx

    padd_scale = 1       #这个值改变会影响OTF的形状
    piexl_scale=1
    if paddingxy ==1 :   #保存原来形状

        nxx = nx/2  #保持原来1倍
        nyy = ny/2
    else :
        nxx = nx   # 扩大2倍
        nyy = ny
    nzz = nz/2

    #设置三维网格
    DqxSupport = 1/(nx*padd_scale)/(dx*piexl_scale)
    DqySupport = 1/(ny*padd_scale)/(dx*piexl_scale)
    DqzSupport = 1/nz/opt.dz
    QXSupport = np.arange(np.ceil(-nxx),np.ceil(nxx))*DqxSupport;    #这里的网格没有扩大，保存原来的大小
    QYSupport = np.arange(np.ceil(-nyy),np.ceil(nyy))*DqySupport;
    QZSupport = np.arange(np.ceil(-nzz),np.ceil(nzz))*DqzSupport;
    [qz,qx,qy] = np.meshgrid(QZSupport,QXSupport,QYSupport,indexing='ij')

    #计算PZ----axialshift
    pixelsizexy = piexl_scale*dx
    p11=np.zeros((3))
    for i in range(3):
        qpixelsize = 1/(pixelsizexy*nx*padd_scale)     #这个nx是移频的像素长度
        qvector_x = PGxy[i,3,0] * qpixelsize 
        qvector_y = PGxy[i,3,1] * qpixelsize 
        
        p11[i] = 1/(np.sqrt(qvector_x**2+qvector_y**2))
    p11_mean = np.mean(p11) #给1阶的pxy

    q0ex = refmed/Wzm
    axialshift = q0ex-np.sqrt(q0ex**2-1/p11_mean**2);

    #计算OTF fc fc_z------算OTF的大小
    q0 = refmed/lamdem    #
    NA1 = (NAmin/lamdem)   #
    NB1 =  np.sqrt(q0**2-NA1**2)

    qpar = np.sqrt(qx**2+qy**2)  #这里没有搞
    axialcutoff = np.sqrt(q0**2-(qpar-NA1)**2)-NB1;
    axialcutoff = (qpar<=2*NA1)*axialcutoff;

    qvector = PGxy * qpixelsize
    OTFmask_cha = []#np.zeros((opt.nc, 2*nzz,2*nxx,2*nyy),dtype=np.float16)
    for i in range(3):
        for j in range(5):
            qpar = np.sqrt( (qx-qvector[i,j,0])**2 + (qy-qvector[i,j,1])**2 )
            axialcutoff = np.sqrt(q0**2-(qpar-NA1)**2)-NB1
            axialcutoff =  (qpar<=2*NA1)*axialcutoff
            if j==3 or j==4:  #±1阶
                otf_mask1 = axialcutoff+DqzSupport/2 >= abs(qz-axialshift)
                otf_mask2 = axialcutoff+DqzSupport/2 >= abs(qz+axialshift)
                otf_mask = otf_mask1|otf_mask2   #取或的操纵
            else:
                otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz))

            otf_mask = (qpar<=2*NA1)*otf_mask
            otf_mask = otf_mask.astype(np.float32)
            otf_mask[otf_mask<0.1] = 0
            otf_mask[otf_mask!=0] = 1
            
            #插值插到2n 大小
            # otf_mask_temp = _U.Interpolate(opt,otf_mask.astype(np.float32),nx,ny,nz)
            OTFmask_cha.append(otf_mask)
            # OTFmask_cha[i*opt.Nshifts+j] =  otf_mask#otf_mask_temp
    OTFmask_cha = np.array(OTFmask_cha)
    OTFmask_cha[OTFmask_cha<0.1]=0
    OTFmask_cha[OTFmask_cha !=0] =1
    OTFmask_cha = OTFmask_cha.astype(np.bool_)
    _U.save_tif('OTFmask_cha.tif',OTFmask_cha)
    # print(OTFmask_cha.shape)
    return OTFmask_cha

def otf_mask2D_SIM_PE(OTFmask3D,piancha=0):

    nc,nz,nx,ny = OTFmask3D.shape
    medz = nz//2+piancha
    OTFmask2D = np.zeros((3,nx,ny))
    OTFmask2D[0] = OTFmask3D[0,nz//2]
    OTFmask2D[1] = OTFmask3D[1,nz//2]
    OTFmask2D[2] = OTFmask3D[2,nz//2]

    return OTFmask2D

def otf_mask3D_SIM_PE(opt,nx,ny,nz,PGxy=np.zeros((1,4,2)),paddingxy=2):
    ''' 
    仅仅只需要获得OTFmask_cha 就可以吧
    input：nz，nx，ny OTF大小 
           kuodaxy =1 保持原来大小， =2 扩大2倍
    output： 根据参数kuodaxy 来决定， OTF形状不变，仅仅是padding不padding的问题
    #note 仅仅第一个方向的0 2 1 阶mask
    '''

    try:
        NA = opt.na_mask
        refmed = opt.n_mask
        NAmin = min(NA,refmed)
        print('用otf自己的na值')
    except:
        NA = opt.na
        refmed = opt.n
        NAmin = opt.namin

    Wzm = opt.Wzm #激发波1 488
    lamdem = opt.lamdem #发射波2 525
    theta = opt.theta
    dx = opt.dx   #因为像素扩大，所以减半
    dy = opt.dx

    padd_scale = 1       #这个值改变会影响OTF的形状
    piexl_scale=1
    if paddingxy ==1 :   #保存原来形状
        nxx = nx/2  #保持原来1倍
        nyy = ny/2
    else :
        nxx = nx   # 扩大2倍
        nyy = ny
    nzz = nz/2

    #设置三维网格
    DqxSupport = 1/(nx*padd_scale)/(dx*piexl_scale)
    DqySupport = 1/(ny*padd_scale)/(dx*piexl_scale)
    DqzSupport = 1/nz/opt.dz
    QXSupport = np.arange(np.ceil(-nxx),np.ceil(nxx))*DqxSupport;    #这里的网格没有扩大，保存原来的大小
    QYSupport = np.arange(np.ceil(-nyy),np.ceil(nyy))*DqySupport;
    QZSupport = np.arange(np.ceil(-nzz),np.ceil(nzz))*DqzSupport;
    [qz,qx,qy] = np.meshgrid(QZSupport,QXSupport,QYSupport,indexing='ij')

    #计算PZ----axialshift
    pixelsizexy = piexl_scale*dx
    p11=np.zeros((1))
    qpixelsize = 1/(pixelsizexy*nx*padd_scale)     #这个nx是移频的像素长度
    # for i in range(1):
        
    #     qvector_x = PGxy[i,3,0] * qpixelsize 
    #     qvector_y = PGxy[i,3,1] * qpixelsize 
        
    #     p11[i] = 1/(np.sqrt(qvector_x**2+qvector_y**2))
    # p11_mean = np.mean(p11) #给1阶的pxy

    # q0ex = refmed/Wzm
    # axialshift = q0ex-np.sqrt(q0ex**2-1/p11_mean**2);

    axialshift = opt.pz*DqzSupport    #不通过PGxy来算，直接用参数中pz来
    # print('axialshift = ',axialshift/DqzSupport )

    #计算OTF fc fc_z------算OTF的大小
    q0 = refmed/lamdem    #
    NA1 = (NAmin/lamdem)   #
    NB1 =  np.sqrt(q0**2-NA1**2)

    qpar = np.sqrt(qx**2+qy**2)  #这里没有搞
    axialcutoff = np.sqrt(q0**2-(qpar-NA1)**2)-NB1;
    axialcutoff = (qpar<=2*NA1)*axialcutoff;

    qvector = PGxy * qpixelsize
    OTFmask_cha = []
    for i in range(1):
        for j in [0,1,3]:  #0 2 1 阶
            qpar = np.sqrt( (qx-qvector[i,j,0])**2 + (qy-qvector[i,j,1])**2 )
            axialcutoff = np.sqrt(q0**2-(qpar-NA1)**2)-NB1
            axialcutoff =  (qpar<=2*NA1)*axialcutoff
            if j==3 or j==4:  #±1阶
                otf_mask1 = axialcutoff+DqzSupport/2 >= abs(qz-axialshift)
                otf_mask2 = axialcutoff+DqzSupport/2 >= abs(qz+axialshift)
                otf_mask = otf_mask1|otf_mask2   #取或的操纵
            else:
                otf_mask = (axialcutoff+DqzSupport/2 >= abs(qz))

            otf_mask = (qpar<=2*NA1)*otf_mask
            otf_mask = otf_mask.astype(np.float32)
            otf_mask[otf_mask<0.1] = 0
            otf_mask[otf_mask!=0] = 1
            
            #插值插到2n 大小
            # otf_mask_temp = _U.Interpolate(opt,otf_mask.astype(np.float32),nx,ny,nz)
            OTFmask_cha.append(otf_mask)
    OTFmask_cha = np.array(OTFmask_cha)
    OTFmask_cha[OTFmask_cha<0.1]=0
    OTFmask_cha[OTFmask_cha !=0] =1
    OTFmask_cha = OTFmask_cha.astype(np.bool_)
    
    if nz ==1:  #slice
        OTFmask_cha[1] = OTFmask_cha[0]
        OTFmask_cha[2] = OTFmask_cha[0]

    _U.save_tif('OTFmask_021.tif',OTFmask_cha)
    return OTFmask_cha.astype(np.bool_)  


#MP
def otf_mask_formula(opt,nx,ny,nz,PGxy=[0,0],PGz=[0],Debug=0,kuodaxy=1):
    """
    7.2日调节，变为可以得到不扩大的OTF
    只生成0阶
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
    
    Wzm = opt.Wzm #激发波1 488
    lamdem = opt.lamdem #发射波2 525
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
    DqxSupport = 1/(nx*padd_scale)/(dx*piexl_scale)
    DqySupport = 1/(ny*padd_scale)/(dx*piexl_scale)
    DqzSupport = 1/nz/opt.dz
    QXSupport = torch.arange(np.ceil(-nx),np.ceil(nx))*DqxSupport;    #这里的网格没有扩大，保存原来的大小
    QYSupport = torch.arange(np.ceil(-ny),np.ceil(ny))*DqySupport;
    # QZSupport = np.arange(np.ceil(-nz//2),np.ceil(nz//2))*DqzSupport;
    QZSupport = torch.arange(np.ceil(-nz),np.ceil(nz))*DqzSupport;
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

def otf_mask_3d_2d(opt,Debug,PGxy=[0,0],PGz=[0]):

    OTFmask3D = otf_mask_formula(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz, PGxy=PGxy,PGz=PGz,Debug=Debug,kuodaxy=1) #这里
    OTFmask3D = _U.pad_or_cut(OTFmask3D,x=2*opt.nx,y=2*opt.ny,z = opt.kuodapz*opt.nz)
    #      ！！！！ 把OTF的扩大放到外面来
    # _u.save_tif('OTFmask3D.tif',OTFmask3D)
    #插值OTFmask3D
    if opt.Zpadding!=0:
        #轴向z插值
        nx,ny,nz =  opt.kuodaxy*opt.nx, opt.kuodaxy*opt.ny, opt.nz_re
        OTFmask3D = _U.Interpolate(OTFmask3D.astype(np.float32),nx,ny,nz)
        OTFmask3D[OTFmask3D < 0.1] = 0
        OTFmask3D[OTFmask3D != 0 ] = 1
        OTFmask3D = OTFmask3D.astype(np.uint8)

    #让OTFMASK更小一点，腐蚀一圈
    OTFmask3D = _PE.Cir_Erode_3D(OTFmask3D,size=8 )
    #OTFmask2D
    OTFmask2D = OTFmask3D[opt.nz_re//2]
    # _u.save_tif('OTFmask3D.tif',OTFmask3D)
    if Debug==1:
        _U.save_tif('optOTFmask3D.tif',OTFmask3D.astype(bool))
        _U.save_tif('optOTFmask2D.tif',OTFmask2D.astype(bool))
    return OTFmask3D,OTFmask2D



# def otf_mask_3d_2d_3DSIM(opt,Debug,PGxy=[0,0],PGz=[0],paddingxy=2):
    
#     # OTFmask3D = np.zeros((3, opt.nz, 2*opt.nx, 2*opt.ny))
#     OTFmask3D = otf_mask3D_SIM_PE(opt,opt.nx,opt.ny,opt.nz_re,PGxy,paddingxy=2)
#     # OTFmask3D[0] = otf_mask_formula(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz//2,Debug=Debug,kuodaxy=1)
#     # temp1  = otf_mask_formula(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz//2, PGxy=PGxy,PGz=PGz,Debug=Debug,kuodaxy=1)
#     # temp2  = otf_mask_formula(opt,nx=opt.nx, ny=opt.ny, nz=opt.nz//2, PGxy=PGxy,PGz=[-PGz[0]],Debug=Debug,kuodaxy=1)
#     # OTFmask3D[1] = OTFmask3D[0]
#     # OTFmask3D[2] = temp1 + temp2

#     # OTFmask2D = OTFmask3D[0,opt.nz_re//2]
#     nc,nz,nx,ny = OTFmask3D.shape
#     # if opt.nz==1:
#     OTFmask2D = np.zeros((3,nx,ny))
#     OTFmask2D[0] = OTFmask3D[0,nz//2]
#     OTFmask2D[1] = OTFmask3D[1,nz//2]
#     OTFmask2D[2] = OTFmask3D[2,nz//2]


#     if Debug==1:
#         _U.save_tif('optOTFmask3D.tif',OTFmask3D.astype(bool))
#         _U.save_tif('optOTFmask2D.tif',OTFmask2D.astype(bool))
#     return OTFmask3D,OTFmask2D



def get_pzp1(opt,PGxy):
    #计算照明光路，通过估计的1阶PGxy，计算theta，计算pz
    #返回PGzz (3,7)
    
    n = opt.n
    Wzm = opt.Wzm    #照明光路
    dx = opt.dx/2      #SIM piexlsize
    nx = opt.nx*2      #SIM nx
    dz = opt.dz        #SIM dz
    
    P_long = np.sqrt( np.sum(PGxy**2,axis=2) )  #(3,5)计算每方向每阶的p长度
    PGzz = np.zeros((3,5))
    PGxy1 = np.zeros((3,1))
    for i in range(3):                          #遍历3个方向
        p1 =np.mean(P_long[i,3:])              #每个方向1阶 平均长度    
        theta = np.arcsin(p1/dx/nx *Wzm/n)*180/np.pi #每个方向的theta角度
        
        kznm =  n*(1-np.cos(theta*np.pi/180)) / Wzm  #通过theta 计算实际pz
        pz = dz*(opt.nz+2*opt.Zpadding)*kznm

        '保存记录'
        # print(f'第{i+1}方向theta = {theta},pz={pz}')
        PGzz[i,[3]] = pz
        PGzz[i,[4]] = pz
        PGxy1[i]   = p1 
        
    return PGzz,PGxy1



