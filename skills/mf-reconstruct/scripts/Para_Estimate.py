import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import cv2
# import BaseFunctions_tt_7_1 as t
# import BaseFunctions_np_7_1 as n
import Utils as _U
import WindowF as _WF
import DataProcess as _DP
import scipy
# import tifffile as tf
from PyEMD import EMD

import cv2
import math
import datetime  #计时

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


def flip180(arr):#2D 3D都可以翻转
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr.copy()


def dft_corr2d(opt,h,g):# tt  h,g求的互相关(频域实现)
    """进行二维互相关"""
    g = torch.conj(g)   #先求共轭
    try:
        g = (flip180(g.numpy()))#再翻转180°
    except:
        g = (flip180(g.resolve_conj().numpy()))#再翻转180°

    k= np.array(h.shape)
    L = np.array(g.shape)

    g = _U.ToTensor(g)
    h = _U.ToTensor(h)
    f = torch.zeros_like(_U.ToTensor(k+L),dtype = opt.tdtype_com)
    f = tifftn( tfftn( h,s=tuple(k+L-1) )*tfftn( (g),s=tuple(k+L-1) ) )
    nx,ny = f.shape

    if nx!= (k+L)[0]: #就补一个零
        f = _U.pad_or_cut(f,x=(k+L)[0],y=(k+L)[1])
    return _U.ToNumpy(f)

def dft_corr3d(opt,h,g):# tt  h,g求的互相关(频域实现)
    """进行3维互相关
    将互相关变为卷积，再变到频域的乘积
    """
    g = torch.conj(g)   #先求共轭
    try:
        g = (flip180(g.numpy()))#再翻转180°
    except:
        g = (flip180(g.resolve_conj().numpy()))#再翻转180°

    k= np.array(h.shape)
    L = np.array(g.shape)

    g = _U.ToTensor(g)
    h = _U.ToTensor(h)
    f = torch.zeros_like(_U.ToTensor(k+L),dtype = opt.tdtype_com)
    f = tifftn(tfftn(h,s=tuple(k+L-1) )*tfftn((g),s=tuple(k+L-1)))
    nz,nx,ny = f.shape
    # print('dft_corr3d-f:',f.shape)
    
    if nx!= (k+L)[0]: #就补一个零
        f = _U.pad_or_cut(f,x=(k+L)[1],y=(k+L)[2],z=(k+L)[0])
    return _U.ToNumpy(f)


def Cir_Erode(data,size=3):
    """
    input: np
    output: np
    """
    if size == 0:
        return data
    kernel = np.ones((size, size), np.uint8)
    kernel_re = []
    rows, cols = kernel.shape
    for ii in range(rows):
        result = [0 if math.sqrt((ii-size//2)**2+(jj-size//2)**2) > size//2 else 1 for jj in range(cols)]
        kernel_re.append(result)
    kernel_re = np.array(kernel_re, np.uint8)

    data_erode =cv2.erode(data,kernel_re)


    if np.mean(data)!=0 and np.mean(data_erode) == 0:
        _U.save_tif(r'查看data.tif',data)
        _U.save_tif(r'查看data_erode.tif',data_erode)
        raise '3D 腐蚀太严重 ,需减少size'
    return data_erode

def Cir_dilate(data,size=3):
    """
    input: np
    output: np
    """
    if size == 0:
        return data
    kernel = np.ones((size, size), np.uint8)
    kernel_re = []
    rows, cols = kernel.shape
    for ii in range(rows):
        result = [0 if math.sqrt((ii-size//2)**2+(jj-size//2)**2) > size//2 else 1 for jj in range(cols)]
        kernel_re.append(result)
    kernel_re = np.array(kernel_re, np.uint8)

    data_erode =cv2.dilate(data,kernel_re)
    # if np.mean(data)!=0 and np.mean(data_erode) == 0:
    #     _U.save_tif(r'查看data.tif',data)
    #     _U.save_tif(r'查看data_erode.tif',data_erode)
    #     raise '3D 腐蚀太严重 ,需减少size'
    return data_erode


def Cir_Erode_3D(data,size=3):
    #没有轴向腐蚀，只是分成slice这样腐蚀
    data2=_U.image_reshape(data,dim=3)
    nz,nx,ny = data2.shape
    for i in range(nz):
        data2[i]=Cir_Erode(data[i],size=size)
    return data2

def Cir_dilate_3D(data,size=3):
    #没有轴向腐蚀，只是分成slice这样腐蚀
    data2=_U.image_reshape(data,dim=3)
    nz,nx,ny = data2.shape
    for i in range(nz):
        data2[i]=Cir_dilate(data[i],size=size)
    return data2


def saveparameter(path,PGxy,PGz,cfai,run_parameter):
    '''
    用于保存估计的参数
    '''
    path = str(path)
    filename = run_parameter['filename']
     
    if PGxy.any():
        filename1 = path+'\\'+'PGxy'+filename+'.npy'
        np.save(filename1,PGxy)
        print(filename1)
        
        filename1 = _U.save_path('PGxy'+filename+'.npy',run_parameter)#保存在dataOut中
        np.save(filename1,PGxy)

    if PGz.any():
        filename2 = path+'\\'+'PGz'+filename+'.npy'
        np.save(filename2,PGz)

        filename2 = _U.save_path('PGz'+filename+'.npy',run_parameter)#保存在dataOut中
        np.save(filename2,PGz)

    if cfai.any():
        filename3 = path+'\\'+'cfai'+filename+'.npy'
        np.save(filename3,cfai)

        filename3 = _U.save_path('cfai'+filename+'.npy',run_parameter)#保存在dataOut中
        np.save(filename3,cfai)

# def saveparameter2(path,PGz,cfai,filename=''):
#     '''
#     用于保存估计的参数，不保存PGxy
#     '''

#     # if PGxy.any():
#     # filename1 = path+'PGxy'+filename+'.npy'
#     # np.save(filename1,PGxy)
#     # print(filename1)
# # if PGz.any():
#     filename2 = path+'PGz'+filename+'.npy'
#     np.save(filename2,PGz)
# # if cfai.any():
#     filename3 = path+'cfai'+filename+'.npy'
#     np.save(filename3,cfai)


def loadparameter(opt,path):
    '''
    用于加载估计的参数
    '''
    from pathlib import Path
    print(f'{path}')
    PGzF=0
    PGxyF=0
    cfaiF=0
    loop_file = Path(str(path)).iterdir()
    for ldr_i in loop_file:
        if ldr_i.is_file() == 1 :  #是文件
            if   ('PGxy' in ldr_i.name):
                PGxy = np.load(ldr_i)
                PGxyF=1
            elif ('cfai' in ldr_i.name):
                cfai = np.load(ldr_i)
                cfaiF=1
            # elif ('PGz' in ldr_i.name):
            #     PGz = np.load(ldr_i)
            #     PGzF=1


    if PGxyF == 1:
        plong=0
        for i in range(opt.Nangles):
            for j in range(1,3):   #±2
                plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
            for j in range(3,opt.Nshifts):   #±1
                plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
        plong = plong/(opt.nc-3)


    if PGzF==0:
        PGz = np.zeros((3,5))
        print('加载pz参数失败！！！！')
    return cfai,PGxy,PGz,plong


def loadparameter2(opt,path):
    '''
    只加载了P值
    '''
    from pathlib import Path
    print(f'loadparameter2：{path}')
    loop_file = Path(str(path)).iterdir()
    for ldr_i in loop_file:
        if ldr_i.is_file() == 1 :  #是文件
            if   ('PGxy' in ldr_i.name):
                PGxy = np.load(ldr_i)
                PGxyF=1
            # elif ('PGz' in ldr_i.name):
            #     PGz = np.load(ldr_i)
            #     PGzF=1


    # PGxy = np.array(PGxy)
    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Nshifts):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)
    return PGxy,plong

def loadparameter3(opt,path):
    '''
    只加载了cfai值
    '''
    try:
        filename3 = path+'cfai.npy'
        cfai = np.load(filename3)
    except:
        print('加载参数失败！！！！')
        pass

    return cfai


#############重新整理参数估计部分##########################
#int px计算
def get_intpxy(opt,ck0,ck1,i,j,methods='FAN'):
    '''
    input 3D
    '''
    assert len(ck0.shape) == 3
    assert len(ck1.shape) == 3
    nz,nx,ny = ck0.shape
    print(nz,nx,ny)
    # 范老师的方法
    if methods=='FAN':
        Hzuan = _U.ToTensor(_U.pad_or_cut(opt.OTFmask2D,x=nx,y=ny))
        Hzuan = _U.image_reshape(Hzuan,dim=2)
        assert len(Hzuan.shape)==2
        
        cishu = abs(dft_corr2d(opt,Hzuan,Hzuan))     #消除面积效应
        _U.save_tif('cishu.tif',cishu.real)

        
        imagf0AC = ck0[nz//2]                         #取出中间一层/ projetction
        imagf0_ = imagf0AC/(np.abs(imagf0AC)+opt.eps) #取出 相位角
        imagf1AC = ck1[nz//2]                         
        imagf1_ = imagf1AC/(np.abs(imagf1AC)+opt.eps) 
        _U.save_tif('imagf0AC.tif',imagf0AC.real)
        imagf0_ = _U.ToTensor(imagf0_)#*Hzuan        #消
        imagf1_ = _U.ToTensor(imagf1_)#*Hzuan

        jieguo = dft_corr2d(opt,(imagf0_),(imagf1_));     #用m=0阶 和m=2阶来求互相关
        _U.save_tif('jieguo.tif',jieguo.real)
        lihe = abs(jieguo/(cishu+opt.eps));           #消除面积影响

    elif methods=='SHAO':
        imagf0AC = ck0*Hzuan                          #全部用上 
        imagf1AC = ck1*Hzuan
        
        #变换到空域，先对imagf0AC横向增大
        imagf0AC_pad = _U.pad_or_cut(imagf0AC,x=2*nx,y=2*ny,z=1*nz)
        imagf1AC_pad = _U.pad_or_cut(imagf1AC,x=2*nx,y=2*ny,z=1*nz)
        # print(imagf0AC_pad.shape)
        imagf0AC_kong = _U.ifftn(imagf0AC_pad)      #变换到空域 
        imagf1AC_kong = _U.ifftn(imagf1AC_pad)      #
        # _U.save_tif('imagf0AC_kong.tif',np.real(imagf0AC_kong))

        lihe = (np.max( abs(_U.fftn(imagf0AC_kong*imagf1AC_kong.conj() ))/1000,axis=0))   #轴向投影

    #通用部分，得到lihe后，确定最亮位置
    XV,YV=_U.meshgridxy(opt,kuodaxy=2)
    kr = np.sqrt(XV**2+YV**2)
    lihe[kr>(opt.pg+opt.fanwei)]=0
    lihe[kr<(opt.pg-opt.fanwei)]=0

    lihe = abs(lihe)

    # print((abs(lihe)==np.max(abs(lihe))).nonzero()) #这个为最大的值，坐标
    # print((lihe==np.max(lihe)).nonzero())    #找到最大值的坐标
    _U.save_tif(f"4 lihe-pxy-{i}-{j}.tif",lihe)
    opt.lihe2d.append(lihe)
    maxx = np.where(lihe==np.max(lihe))[0][0] # 168。8
    maxy = np.where(lihe==np.max(lihe))[1][0] # 127.5

    return maxx-opt.nx,maxy-opt.ny

#整数pz
def get_intpz(opt,ck0,ck1,i,j,methods='SHAO'):
    pass

def get_subpxy(opt,i,j,ck_0,ck_p,pgx,pgy,func):
    # 迭代互相关 COR
    #Lihe函数，作为更新依据
    xstep = 1
    ystep = 1
    buchangy = ystep
    buchangx = xstep
    jindu = 0.05
    # n_256 = opt.nx   #256
    PXY =np.array([pgx,pgy])/(2*opt.nx)
    he_0 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
    while( (buchangx>jindu) or (buchangy>jindu)):

        test = np.zeros(4)

        """max x方向"""
        maxx_tmp1 = pgx-1e-5; #上
        maxx_tmp2 = pgx+1e-5; #下
        
        for i in range(2):  #% 向上or 向下 （X轴往哪里移动）
            if i==0:
                kxtest=maxx_tmp1;
            else:
                kxtest=maxx_tmp2;
            PXY =np.array([kxtest,pgy])/(2*opt.nx)   #这里PXY的计算与图像大小有光，所以需要实时操作
        
            he1 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
            test[i] = he1
        
        if((test[0]>test[1])):  #移动更小P 得到的结果更大
            flag_maxx=-1;       #recoder
        elif(test[0]<test[1]):
            flag_maxx=1;        #需要加大P
        else:
            try:
                flag_maxx=-flag_maxx
            except:
                flag_maxx=1

        while(buchangx>(jindu)):   #根据上面的上下指示，开始修正maxx_temp
            kxtest = pgx+flag_maxx*buchangx;      #不断的修正maxx_TMP

            PXY = np.array([kxtest,pgy])/(2*opt.nx)   #每次仅改变一个方向
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1]) 
            if(he_tmp<=he_0):
                buchangx=buchangx*0.5
            else:
                print("updata pgx =",kxtest,'hetemp = ',he_tmp)
                pgx=kxtest           #更新！
                he_0=he_tmp
                #重置
                buchangy = ystep
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环，开始找maxy
            

        """max y方向"""
        maxy_tmp1 = pgy-1e-5; #左
        maxy_tmp2 = pgy+1e-5; #右
        for i in range(2,4):  #% 不同方向找最优
            if i==2:
                kytest=maxy_tmp1;
            else:
                kytest=maxy_tmp2;
            PXY = np.array([pgx,kytest])/(2*opt.ny)   #每次仅改变一个方向
        
            he1 = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
            test[i] = he1
        
        if((test[2]>test[3])):  #移动更小P 得到的结果更大
            flag_maxy=-1;       #recoder
        elif(test[2]<test[3]):
            flag_maxy=1;        #需要加大P
        else:
            try:
                flag_maxy=-flag_maxy
            except:
                flag_maxy=1
            
        
        while(buchangy>(jindu)):
            kytest = pgy+flag_maxy*buchangy;      #不断的修正maxx_TMP
            PXY = np.array([pgx,kytest])/(2*opt.ny)  #每次仅改变一个方向
            he_tmp = func(opt,i,j,ck_0,ck_p,PXY[0],PXY[1])
            if(he_tmp<=he_0):
                buchangy=buchangy*0.5
            else:
                print("updata pgy =",kytest,'hetemp = ',he_tmp)
                pgy=kytest
                he_0=he_tmp
                #重置
                buchangx = xstep
                break            #通过几次迭代得到更优的maxx 了，就退出这轮循环，开始找maxy
    return pgx,pgy





def get_subpzxy_PCA(opt,ckp,j,pxint,pyint,pzint=0,flag='zy'):
    '''用PCA方式得到亚像素'''
    from numpy import linalg as la
    assert len(ckp.shape) == 3 
    #step 准备工作：
    
    #step2--int移频 先对ck_P移动
    ##法1
    # PXY = np.array([maxx-n_256,maxy-n_256])/(2*n_256)
    # ZV,XV,YV = _U.meshgrid(opt,kuodaxy=2,kuodaz=opt.kuodapz)
    # ysh =(np.exp(2j*np.pi*(kx*XV+ky*YV+kz*ZV)))   #单边移动 to beginning

    ##法2
    ckp_intmove = np.roll(ckp, (pzint,pxint,pyint) ,axis=(0,1,2) )


    #取mask
    if j<=2: #2th
        mask_size = 3
        padding_size= 5
        ROI = _U.pad_or_cut(ckp_intmove,x=mask_size, y =mask_size,z=mask_size)
        ROI = _U.pad_or_cut(ROI,x=mask_size+padding_size+1, y =mask_size+padding_size+1,z=opt.nz_re)
        #3D降2D
        ROI_XY = np.sum(ROI,axis=0)  #按Z进行累加,结果显示这个更加鲁棒
        Space_ROI = _U.ifftn(ROI_XY)

    #3D将2D 截取一面的方式
    if j>2: #1th
        mask_size = 3
        padding_size= 5
        ROI = _U.pad_or_cut(ckp_intmove,x=mask_size, y =mask_size,z=mask_size)
        ROI = _U.pad_or_cut(ROI,x=mask_size+padding_size+1, y =mask_size+padding_size+1,z=mask_size+padding_size+1)
        #3D降2D
        nzz,nxx,nyy = ROI.shape
        if flag == 'zy':
            ROI_ZY = np.sum(ROI,axis=1)
            # ROI_ZY = (ROI[:,nxx//2,:])
            # ROI_ZY = np.max(ROI,axis=1)
            Space_ROI = _U.ifftn(ROI_ZY)
        else:
            ROI_ZX = np.sum(ROI,axis=2)
            # ROI_ZX = (ROI[:,:,nyy//2])
            # ROI_ZX = np.max(ROI,axis=2)
            Space_ROI = _U.ifftn(ROI_ZX)



    #step3 计算逆变换，获得指数项
    Phase_ROI = np.exp(1j*np.angle(Space_ROI))


    #step4 SVD分解 得到U  V
    assert len(Phase_ROI.shape)==2
    u,sigma,vt = la.svd(Phase_ROI)   #进行SVD时 需要是2D
    
    #step5 最小二乘法拟合
    dx_z,u[:,0] = Eigenvector_Fitting(u[:,0],j)
    dy,vt[0,:] = Eigenvector_Fitting(vt[0,:],-j)

    #计算拟合出来的初相
    # sigma[1:]=0   
    # Phase_ROI_new = (u @ np.diag(sigma)) @ vt
    return dx_z,dy

def Eigenvector_Fitting(eigenone,j):
    #eigenone 对u来说是[:,0]；对v来说是[0,:]
    #j 用来表示这是第几阶, +表示u  -表示v
    Eigenvector_angle= np.unwrap(np.angle(eigenone))  #取angle 并消除-pi pi的区别
    Eigenvector_angle_fit = Eigenvector_angle[1:-1].reshape(-1)  #因为有padding 去掉头尾不准确的值
    cai_size = Eigenvector_angle_fit.shape[0]         #长度

    FIT = np.polyfit(np.arange(cai_size), Eigenvector_angle_fit, 1) # 用1次多项式拟合，依次为系数1，最后为bias
    sub = -FIT[0]*cai_size/(2*np.pi)                  #计算亚像素

    if sub>=1:
        _U.save_tif(f'Eigenvector_angle-{j}.tif',Eigenvector_angle)
        print(f'sub：{sub}有问题！请查看整数像素，请修改mask窗口大小')
    else:
        if abs(j)<3:  #2阶
            if j>0:  
                print('dx=',sub)   #还需要判断，这个值应该小于1
                                   #在zy中，2阶z的改变量很小 可以相信
            elif j<0:
                print('dy=',sub)
        elif abs(j)>=3:
            if j>0:  
                print('dz=',sub)   #还需要判断，这个值应该小于1
                                   #在zy中，2阶z的改变量很小 可以相信
            elif j<0:
                print('检验dy/dx=',sub) #这个dy来用判断的 不参与更新

    #拟合出理想的Eigenvector
    New_angle = np.polyval(FIT,np.arange(eigenone.shape[0]))
    # _U.save_tif('3d-u(new)_angle.tif',New_U_angle)
    # u[:,0] = New_U_angle
    return sub,New_angle




def get_subpzxy_PCA01(opt,ckp,j,pxint,pyint,pzint=0,flag='zy'):
    '''用PCA方式得到亚像素，0级与±1级光 '''
    pass

#第一种更新策略 等分pz1为左右10等份
def get_subpz(opt,i,j,ck_0,ck_p,pgx,pgy,pgz,m,func,jindu=0.1,fanwei=6):
    #方法1，等分法
    pass

#第二种更新策略:xyz
def get_subpz2(opt,i,j,ck_0,ck_p,pgx,pgy,pgz,m,func):
    '只更新z方向'
    pass

def get_subpz2_2(opt,i,j,ck_0,ck_p,pgx,pgy,pgz,m,func):
    #这个相对于上面，运行速度更慢，因为在各个方向上拟合
    pass
#pz的更新依据 综合pz oxy

def LiheXYZ(opt,i,j,ck_0,ck_p,kx,ky,kz=0,m=2,cfai=0,mask_ger=0,run_parameter=''):

    "2D平面计算/3D计算都可以,用flag=0（2d）区分"

    if m>2 and opt.flag_N==7 and opt.Nshifts==7 : #±1阶 3D
        flag=1 #根据传入的m 自动判断使用3D 还是2D

    if j <3 and cfai!=4: #2阶P亚像素求解 使用中间层
        piancha = 0
        flag= 0
        filename='' ##不用保存
    elif cfai==4:
        piancha = run_parameter['piancha']
        flag= run_parameter['flagV']
        filename=run_parameter['filename']
    else:
        piancha = 0#run_parameter['piancha']
        flag= 1#run_parameter['flagV']
        filename=''#run_parameter['filename']
    
    

    '【准备工作】'
    mask_ger = 0    
    flag3D = flag
    assert len(ck_0.shape)==3

    nz,nx,ny = ck_0.shape
    #构建空域移动矩阵
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=2,kuodaz=opt.kuodapz)
    ysh  = (np.exp(2j*np.pi*(kx*XV+ky*YV+kz*ZV)))   #全部移动

    #根据偏差重新提取OTF_2Dmask
    opt.OTFmask2D = _DP.otf_mask2D_SIM_PE(opt.OTFmask3D,piancha=piancha) 

    #提取做互相关的频谱、相位谱、otf
    if flag3D==0: #±2阶使用平面2D，加速估计过程
        ysh = ysh[0]        

        #[一] 0阶 【image】
        imgf0AC = ck_0[nz//2+piancha,:,:]  #0 提取中间一层
        imgf0 = imgf0AC/(abs(imgf0AC)+opt.eps)   #相位项
        imgf0 = _U.pad_or_cut(imgf0,x=nx*2 ,y=ny*2,z=1)    #二维情况下加pading 2X
         
        #[二]P阶【image】 --需要移动!!!!
        imgfmAC = ck_p[nz//2+piancha,:,:]
        imgfm = imgfmAC/(abs(imgfmAC)+opt.eps)  #提取为相位角
        imgfm = _U.pad_or_cut(imgfm,x=nx*2 ,y=ny*2,z=1)    # Dr 2x
           
        #[三] 0阶 【OTF2D】-不需要移动
        otf = opt.OTF2D[0]
        otfmask = opt.OTFmask2D[0]

        #[四] P阶【OTF2D】---需要移动!!!!
        otf_j = opt.OTF2D[0]
        otfmask_j = opt.OTFmask2D[0]
        if opt.flag_N==5 and j>=3:  #3D-SIM的±1阶
            otf_j = opt.OTF2D[2]
            otfmask_j = opt.OTFmask2D[2]

    else: #±1阶 且 MP 使用3D估计法
        #[一] 0阶 【image】
        imgf0AC = ck_0  
        imgf0 = imgf0AC/(abs(imgf0AC)+opt.eps)   #相位项
        imgf0 = _U.pad_or_cut(imgf0,x=nx*2 ,y=ny*2,z=opt.nz_re)    #二维情况下加pading 2X
        
        #[二]P阶【image】
        imgfmAC = ck_p
        imgfm = imgfmAC/(abs(imgfmAC)+opt.eps)  #提取为相位角
        imgfm = _U.pad_or_cut(imgfm,x=nx*2 ,y=ny*2,z=opt.nz_re)    # Dr 2x
        
        #[三] 0阶 【OTF3D】
        otf = opt.OTF[0]
        otfmask = opt.OTFmask3D[0]

        #[四] P阶【OTF3D】--不需要移动
        otf_j = opt.OTF[0]
        otfmask_j = opt.OTFmask3D[0]
        if opt.flag_N==5 and j>=3:  #3D-SIM的±1阶
            otf_j = opt.OTF[2]
            otfmask_j = opt.OTFmask3D[2]

    '【GPU加速】'
    #################GPU加速###############
    ysh   = (torch.from_numpy(ysh)).to(device=opt.device)
    imgfm = (torch.from_numpy(imgfm)).to(device=opt.device)
    otf_j = (torch.from_numpy(otf_j)).to(device=opt.device)
    otfmask_j = (torch.from_numpy(otfmask_j)).to(device=opt.device)
    
    if mask_ger==0 and flag3D==1:    #不使用生成mask方式
        yshXY= (np.exp(2j*np.pi*(kx*XV+ky*YV)))   #XY移动
        yshZ = (np.exp(2j*np.pi*(kz*ZV)))   #Z移动
        yshZ1 = (np.exp(2j*np.pi*( np.ceil(kz*opt.nz_re)/opt.nz_re *ZV)))   #Z移动
        yshZ2 = (np.exp(2j*np.pi*( np.floor(kz*opt.nz_re)/opt.nz_re *ZV)))   #Z移动
        yshXY = (torch.from_numpy(yshXY)).to(device=opt.device)
        yshZ  = (torch.from_numpy(yshZ)).to(device=opt.device)
        yshZ1  = (torch.from_numpy(yshZ1)).to(device=opt.device)
        yshZ2  = (torch.from_numpy(yshZ2)).to(device=opt.device)

    ##########################################################
    '【高阶频谱分量-移动】'
    imgfm_move = _U.tfftn(_U.tifftn(imgfm)*(ysh))

    '【j阶- OTFmask移动】' 
    if mask_ger ==0:       #移动的方式                                                                
        if flag3D==0:      #2D移动 2th
            #XY 移动
            otfmask_jmove = abs(_U.tfftn( tifftshift(tfftshift(_U.tifftn(otfmask_j))*ysh)) )     #GPU
        else:
            #先XY移动
            otfmask_jmove = abs(_U.tfftn( tifftshift(tfftshift(_U.tifftn(otfmask_j))*yshXY)) )
            yuzhi = 0.97
            otfmask_jmove[abs(otfmask_jmove)>=yuzhi]=1
            otfmask_jmove[abs(otfmask_jmove)!=1]=0

            #####Mask-way1 #再Z移动 
            otfmask_jmove = abs(_U.tfftn( tifftshift(tfftshift(_U.tifftn(otfmask_jmove))*yshZ)) )

        yuzhi = 0.97
        otfmask_jmove[abs(otfmask_jmove)>=yuzhi]=1
        otfmask_jmove[abs(otfmask_jmove)!=1]=0

    else:                 #直接生成otfmask
        pass
        # if flag3D==0:   #   2D情况 2jie 
        #     _ , otfmask_move = _DP.otf_mask_3d_2d(opt,PGxy =[kx*(2*opt.nx),ky*(2*opt.ny)],PGz = [kz*opt.nz_re], Debug=0) #
        #     #2D 腐蚀一下
        #     otfmask_move = Cir_Erode(otfmask_move,size=0)

        # else:
        #     otfmask_move , _ = _DP.otf_mask_3d_2d(opt,PGxy =[kx*(2*opt.nx),ky*(2*opt.ny)],PGz = [kz*opt.nz_re], Debug=0) #
        #     #3D 腐蚀一下
        #     otfmask_move = Cir_Erode_3D(otfmask_move,size=3)

        # otfmask_move =(torch.from_numpy(otfmask_move)).to(device=opt.device)                  #CPU -> GPU

    '【j阶-OTF移动】'
    otf_jmove = abs(_U.tfftn( tifftshift(tfftshift(_U.tifftn(otf_j))*ysh)) )
    otf_jmove = (otf_jmove)*otfmask_jmove
    

    '【GPU加速取消】'
    ###############################################GPU取消
    ysh = ysh.cpu().numpy()
    imgfm = imgfm.cpu().numpy()
    otf_j = otf_j.cpu().numpy()
    otfmask_j = (otfmask_j).cpu().numpy()

    imgfm_move = imgfm_move.cpu().numpy()
    otfmask_jmove = otfmask_jmove.cpu().numpy()
    otf_jmove = otf_jmove.cpu().numpy() 
       
    otf_jmove = _U.normalize(otf_jmove)    #归一化
    ####################################################
    
    if cfai==0:####################################
        '【计算交叉区域】'
        ## 0阶与2阶 、  0阶与1阶
        overlap_mask = otfmask*otfmask_jmove
        w0 = imgf0 * otf_jmove * overlap_mask  
        w1 = imgfm_move * otf * overlap_mask

        w1w0_sum = abs(np.sum((w1*w0.conj()))); 
        overlap_sum = abs(np.sum((overlap_mask)))
        rate = w1w0_sum /overlap_sum
        return abs(rate)        #方法1

    elif cfai==3: #################################
        #变换到空域
        w0_ifft = _U.ifftn(w0*overlap_mask)
        w1_ifft = _U.ifftn(w1*overlap_mask)
        
        hetest = np.sum((w1_ifft*w0_ifft.conj()) );
        # hetest = np.sum((w1_ifft.conj()*w0_ifft) ); #相位是相反的
        # sumXMagFR = np.sum(abs(w0_ifft)**2)
        sumXMagFR = np.sum((np.real(w0_ifft)**2+np.imag(w0_ifft)**2))
           
        modamp1 = (hetest / sumXMagFR)  #复数线性

        # hetest = ((w1_ifft*w0_ifft.conj()) );
        # # hetest = np.sum((w1_ifft.conj()*w0_ifft) ); #相位是相反的
        # sumXMagFR = (abs(w0_ifft)**2)
        # modamp1 = (hetest / sumXMagFR)  #复数线性


        amp2 = abs(modamp1)**2;         #作为更新标准
        Amp = np.sqrt(amp2)
        Ang = np.angle(modamp1)
        # if flag==0:
        #     _U.save_tif('Amp2d.tif',Amp)
        #     _U.save_tif('Ang2d.tif',Ang)
        # else:
        #     _U.save_tif('Amp3d.tif',Amp)
        #     _U.save_tif('Ang3d.tif',Ang)

        return np.mean(Amp),np.mean(Ang)                  #用区域内之和相当于均值，计算c，fai值

    else:   #####################################     #统计区域类，用众数计算c，fai
        '准备工作'
        if flag3D==0: #2D
            #[一] 0阶 【image】
            imgf0 = _U.pad_or_cut(imgf0AC,x=nx*2 ,y=ny*2,z=1)    #二维情况下加pading 2X
            #[二] P阶 【image】
            imgfm = _U.pad_or_cut(imgfmAC,x=nx*2 ,y=ny*2,z=1)    # Dr 2x
        
        else:  #3D
            #【一】 0阶 【image】
            imgf0 = _U.pad_or_cut(imgf0AC,x=nx*2 ,y=ny*2,z=opt.nz_re)    #二维情况下加pading 2X
            #【二】 P阶 【image】
            imgfm = _U.pad_or_cut(imgfmAC,x=nx*2 ,y=ny*2,z=opt.nz_re)    # Dr 2x
        
        '【高阶频谱分量-移动】'
        imgfm_move = _U.fftn(_U.ifftn(imgfm)*(ysh))

        '【计算overlap比值】 with Amp and Angle'
        overlap_mask = otfmask*otfmask_jmove
        #腐蚀区域
        if flag==0:
            size = 20
            print('2D Cir_Erode size=:',size)
            overlap_mask = Cir_Erode(overlap_mask,size=size)
        else: #3D
            overlap_mask = Cir_Erode(overlap_mask,size=1)
        # print('overlap_mask-sum=',np.sum(overlap_mask))
        w0 = imgf0 * otf_jmove * overlap_mask  
        w1 = imgfm_move * otf * overlap_mask
     
        overlap_cfai = w1/(w0+opt.eps)


        # if flag3D==1: #3d
        #     #取出中间一层？
        #     overlap_cfai = overlap_cfai_all[(nz)+piancha,:,:]
        #     overlap_cfai = overlap_cfai_all[(nz)+piancha,:,:]
        #     # _U.save_tif('overlap_cfai_3D_med.tif',abs(overlap_cfai))
        # else:
        # overlap_cfai = np.copy(overlap_cfai_all)

        ' '
        overlap_c    = np.abs(overlap_cfai)
        overlap_fai  = np.angle(overlap_cfai/(abs(overlap_cfai)+opt.eps)); 

        del overlap_cfai
        _U.save_tif(f'overlap_abs-{i,j}.tif',abs(overlap_c))
        _U.save_tif(f'overlap_fai-{i,j}.tif',(overlap_fai))
        # _U.save_tif('overlap_angle.tif',np.angle(overlap_cfai/(abs(overlap_cfai)+opt.eps)))

        # _U.save_tif('overlap_abs.tif',abs(overlap_).astype(np.bool_))
        ######################

        '保存结果'
        if flag3D==0:
            opt.abscm2d.append(overlap_c)
            opt.angcm2d.append(overlap_fai)
            opt.overlap_mask2d.append(overlap_mask)
        else:
            opt.abscm3d.append(overlap_c)
            opt.angcm3d.append(overlap_fai)
            opt.overlap_mask3d.append(overlap_mask)

        'EMD'
        if flag==1:
            print('————————下为 体估计——————————')
        else:
            print('————————下为 面估计——————————')
        overlap_c_1d = overlap_c.flatten()[np.argwhere(overlap_c.flatten())].squeeze()
        overlap_fai_1d = overlap_fai.flatten()[np.argwhere(overlap_fai.flatten())].squeeze()
        ######################
        jdjd=0.02    #fai的间隔
        jdc = 0.01   #C的间隔
        f= np.arange(-np.pi,np.pi,jdjd) # fai
        ff = np.arange(jdc+0,1+jdc,jdc) #c

        e = overlap_fai_1d.T
        ee =  overlap_c_1d.T

        if flag3D==0:             #2D 前后相乘
            testee = ee*ee[::-1]
        else:                   #3D 不翻转相乘
            testee = ee**2
        testee = testee**0.5

        ee = testee
        c_mean = np.mean(testee[(testee<1) & (testee>0.03)])

        #g = fai
        g,g_ = np.histogram(e,f)
        g[int(np.pi/jdjd)]=(g[int(np.pi/jdjd)+1]+g[int(np.pi/jdjd)-1])/2
        fai,g,g2,rate,h = extend(g,j,jdjd)  #fai

        #gg = c
        gg,gg_ = np.histogram(ee,ff)
        emd = EMD()
        imf2 = emd(gg)  #c
        gg2 = np.sum(imf2[0:,:],0)
        gg2[0:10]=0   #
        hh = np.where(gg2 == np.max(gg2)) #不要前面10个
        cvalue = jdc*np.mean(hh)+jdc+0

        print('c_mean:',c_mean) #这个是全
        print("fai=",fai)
        print("c=",cvalue)

        ###画图####
        #返回值没用上
        draw_c_fai(gg,gg2,g,g2,i,j,np.mean(h),c_mean,cvalue,fai,filename,run_parameter,np.mean(hh),jdc,jdjd)

        if rate>0.2: #使用EMD
            return cvalue, fai,c_mean
        else:       #其他情况均使用均值
            return c_mean,fai,cvalue
        
        
def draw_c_fai(c,c2,fa,fa2,i,j,h,c_mean,c_EMD,fai,filename,run_parameter,hh,jdc,jdjd):
    import matplotlib.pyplot as plt
    '画图保存'
    fig = plt.figure(figsize=(15,3))
    sub1 = plt.subplot(1, 2, 1)  
    
    c_max = np.max(c2)
    # plt.plot()
    start = 10
    plt.plot(c2)
    plt.title(f'{i+1,j}-C-Modulation depth',fontname='Times New Roman',weight ='bold')
    # sub1.text(start+1,  0,"MEAN:",ha='center', va='bottom')
    sub1.text(0.98, 0.98,'MEAN:'+str(c_mean.round(2)),
             horizontalalignment='right',
             verticalalignment='top',
             transform = sub1.transAxes)
    # sub1.text(start+1,  0,str(c_mean.round(2)),ha='center', va='bottom')    
    sub1.text(hh, 0,str((c_EMD).round(2)),ha='center', va='bottom')
    
    #转变X轴下标
    keduc = [20,40,60,80,100]
    # keduc = np.array(keduc)-20
    keduc2 = []
    for ii in range(len(keduc)):
            keduc2.append( np.round((keduc[ii])*jdc,1) )
    sub1.set_xlim((start, 100))
    sub1.set_xticks(keduc)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    sub1.set_xticklabels(keduc2)
    sub1.set_ylim(bottom=0.)  #Y从0开始

    #####################画fai############
    sub2 = plt.subplot(1, 2, 2)  
    plt.plot(fa)
    plt.plot(fa2)
    plt.title(f'{i+1,j}-φ-phase',fontname='Times New Roman',weight ='bold')
    sub2.text(h, 0,fai.round(2),ha='center', va='bottom')
    kedu = [0,157,314,471,628]
    kedu2 = []
    for ii in range(len(kedu)):
        if kedu[ii]<=314:
            pianyi = np.pi
        else:
            pianyi = 3*np.pi

        if kedu[ii]==314:
            kedu2.append( '±'+str(abs(np.round((kedu[ii]*jdjd-pianyi),2))) )
        else:
            kedu2.append( np.round((kedu[ii]*jdjd-pianyi),2) )
    sub2.set_xticks(kedu)
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    sub2.set_xticklabels(kedu2)
    sub2.set_ylim(bottom=0.)  #Y从0开始


    
    cfai_png = plt2np(plt)

    # name = _U.save_path(f'cfai/{i+1,j}-cfai-{filename}.png',run_parameter)
    name = _U.save_path(f'{i+1,j}-cfai-{filename}.png',run_parameter,sub_filename='cfai')

    plt.savefig(name)

    # plt.show() #cmd 模式下 不显示图片
    return cfai_png

def plt2np(plt):
    # 引入 FigureCanvasAgg
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    # 引入 Image
    import PIL.Image as Image
    # 将plt转化为numpy数据
    canvas = FigureCanvasAgg(plt.gcf())
    # 绘制图像
    canvas.draw()
    # 获取图像尺寸
    w, h = canvas.get_width_height()
    # 解码string 得到argb图像
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    
    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    
    return rgb_image


def extend(Initial_phase_curve,j,jdjd):
    #如果没在中心，则数据copy一份拼接在右边
    IPC = Initial_phase_curve
    IPC_copy = np.copy(IPC)
    IPC_4pi = np.concatenate((IPC, IPC_copy), axis=0)


    emd = EMD()
    imf = emd(IPC_4pi)

    '按照占比求EMD'
    rate = np.max(IPC_4pi) / np.sum(IPC_copy)
    print( '占比：',rate )
    if rate > 0.02 :              #尖锐
        EMD_curve = np.sum(imf[:,:],0)

    elif rate > 0.013:            #一般
        EMD_curve = np.sum(imf[1:,:],0)

    elif rate > 0.008:            #不太尖锐
        EMD_curve = np.sum(imf[2:,:],0)

    else:# rate > 0.4:           #非常不尖锐
        EMD_curve = np.sum(imf[3:,:],0)

    nx = IPC_4pi.reshape(-1).shape

    '目前IPC_4pi是4pi'
    EMD_curve_pre = np.copy(EMD_curve)
    EMD_curve_pre[int(nx[0]/2):]=0  #后半部分置0
    idx_max = np.where(EMD_curve_pre==np.max(EMD_curve_pre)) #在前半部分找最大值
    fai = np.mean(idx_max)*jdjd - np.pi
    if fai < -2:  #去后半部分拟合
        EMD_curve_later = np.copy(EMD_curve)
        EMD_curve_later[:int(nx[0]/2)]=0  #前半部分置0
        idx_max = np.where(EMD_curve_later==np.max(EMD_curve_later)) #在前半部分找最大值
        fai = np.mean(idx_max)*jdjd - 3*np.pi

        return fai,IPC_4pi,EMD_curve_later,rate,idx_max

    
    # EMD_curve[int(nx[0]/2):]=0
    # idx_max = np.where(EMD_curve==np.max(EMD_curve))
    

    # if fai > 2 * np.pi:
    #     fai = fai - 3*np.pi
    # else:
    #     fai = fai - np.pi
    return fai,IPC_4pi,EMD_curve_pre,rate,idx_max




###################################################

































###################################################
def parameter_estimation(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    #正常2阶求1阶
    assert len(ck.shape)==4
    #超参数xy
    if run_parameter['pg']!=0:
        opt.pg = run_parameter['pg']
        opt.fanwei = run_parameter['fanwei']    
    else:
        opt.pg = int(opt.pxy)
        opt.fanwei = run_parameter['fanwei']


    #PZ范围 得把Zpadding考虑进去 
    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数
    #并把最中间的置为0
    
    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N ==7 and opt.Nshifts==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')
    elif opt.flag_N ==7 and opt.Nshifts==7 and opt.nz==1: #MP slice的情况
        opt.pgz=0
        opt.fanweiz = 0

    PGxy = np.zeros((opt.Nangles,opt.Nshifts,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Nshifts))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Nshifts,3)) #
    pgz = 0
    if opt.flag_N==7 and opt.Nshifts==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re#(opt.nz+2*opt.Zpadding )*kuodapz   #*2

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Nshifts+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):# 无解卷积
                RL_output.append(ck[i*opt.Nshifts+j])
        
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        # _U.save_tif('opt.imgf0.tif',_U.visual(RL_output[0]))  #与里程碑7 相同
        # _U.save_tif('opt.imgf_f2.tif',_U.visual(RL_output[1]))
        # _U.save_tif('opt.imgf_f1_f1.tif',_U.visual(RL_output[3]))

        print(f'【第{i+1}个方向】:')

        for j in range(1,opt.Nshifts ):
            # ±2            
            if j <=1 : #######################
                #整数
                # opt.pg = int(opt.pxy)

                pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') # 2D，中间层互相关
                print(f'±2整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                
                #亚像素
                pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #2D 中间层精确估计p
                print(f'±2亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                
                #求C fai
                c,fai,c_EMD= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4,run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                
                #保存pgxy c fai
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai,c_EMD])


            elif j == 2: ##########################
                PGxy[i,j,:] = -PGxy[i,1,:]  
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,2] = cfai[i,1,2]
                cfai[i,j,1] = -cfai[i,1,1]
            
            #±1
            elif j==3 and opt.Nshifts == 5:  # ±1 ##################################
                ##########-------------------1阶重新计算法--------------
                # #1阶 P
                # opt.pg = int(opt.pxy//2)
                # pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') #大致估计p
                # print(f'±1整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                
                # #亚像素
                # pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #精确估计p
                # print(f'±1亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                ##########-------------------直接2阶赋值法--------------

                #用2阶数直接赋值
                PGxy[i,j,:] = PGxy[i,j//2,:]/2  # -2 2 -1 1
                                                #  1 2  3 4
                pgx = PGxy[i,j,0]
                pgy = PGxy[i,j,1]

                ##########-------------------求c-----------------------
                #求CFAI
                c,fai,c_EMD= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4,run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                #保存
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                
            elif j==4 and opt.Nshifts == 5:  # ±1 ##################################
                PGxy[i,j,:] = -PGxy[i,3,:]  
                cfai[i,j,0] = cfai[i,3,0]
                cfai[i,j,2] = cfai[i,3,2]
                cfai[i,j,1] = -cfai[i,3,1]
            
            #±1
            #情况1 知晓±2 拿来直接用 
            elif j>=3 and opt.Nshifts == 7  and opt.nz!=1: ###########MP stack##################
                if j==3 or j==4:
                    PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
                else :
                    PGxy[i,j,:] = PGxy[i,2,:]/2
                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]
                
                #求pz整数像素
                opt.pg = int(opt.pxy//2)
                pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
                print(f'整数像素pzint = {pzint},pxint= {pxint},pyint= {pyint}')

                #求亚像素
                # 自己求pz
                if run_parameter['pzzz']==0:
                     #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                # 给一个亚像素值，但还是要算
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] !=0 : #按照给的理想值 range=1 再微调一下
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
 
                    #策略1
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                    # 再次调用
                    pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
                    #新方法 总和/总和 得到c fai
                    #策略2 
                    # maxz_y,maxx_y,maxy_y = get_subpz2(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)  #仅在整数上判断方向
                    
                    #策略3
                    # maxz_y,maxx_y,maxy_y = get_subpz2_2(opt,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)
                    # pgx =  maxx_y-opt.nx
                    # pgy =  maxy_y-opt.ny
                    # pgz =  maxz_y-(opt.nz+2*opt.Zpadding)

                # 固定pz值
                elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] ==0 : #按照给的lai
                    if j == 3 or j == 5:
                        pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding) 
                    elif j == 4 or j == 6:
                        pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                else:
                    raise('参数错误')
                print(f'亚像素pz{j}阶=',pgz,'亚整数pgx=',pgx,'亚整数pgy = ',pgy) 


                #求CFAI
                c,fai,c_EMD = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=4,run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)
                
                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGxy[i,j,:] = np.array([pgx,pgy]) 
                PGz[i,j] = pgz        #得到精确的Pz


            elif j>=3 and opt.Nshifts == 7  and opt.nz==1: ###########MP slice##################
                if j==3 or j==4:
                    PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
                else :
                    PGxy[i,j,:] = PGxy[i,2,:]/2
                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]
                pgz = 0

                #求CFAI
                c,fai,c_EMD = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=4,run_parameter=run_parameter)#三维拟合,求c 和 φ
                print('c fai :',c,fai,c_EMD)
                
                #保存CFAI
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                PGxy[i,j,:] = np.array([pgx,pgy]) 
                PGz[i,j] = pgz        #得到精确的Pz
                
    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime)
    plong=[]
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2) )
        for j in range(3,opt.Nshifts):   #±1
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2)
    plong = np.mean(np.array(plong))
        

    if Debug==True: #4

        overlap_mask2d = np.array(opt.overlap_mask2d) 
        abscm2d = np.array(opt.abscm2d)
        angcm2d = np.array(opt.angcm2d)
        lihe2d = np.array(opt.lihe2d)
        
        kzz,kxx,kyy = lihe2d.shape
        lihe2d[:,kxx//2,kyy//2] = 0.1
        lihe2d = np.insert(lihe2d,0,values = np.max(lihe2d,axis=0),axis=0) #加一个投影
        lihe2d = np.insert(lihe2d,0,values = np.min(lihe2d,axis=0),axis=0) #加一个投影
        # _U.save_tif(f'overlap2d_±2-[{filename}].tif',overlap_mask2d.astype(np.bool_))
        # _U.save_tif(f'overlap2d_c_±2-[{filename}].tif',abscm2d)
        # _U.save_tif(f'overlap2d_fai_±2-[{filename}].tif',angcm2d)
        
        _U.save_tif_v21(f'lihe_±2-[{filename}].tif',lihe2d,run_parameter)

        if opt.Nshifts==7:
            overlap_mask3d = np.array(opt.overlap_mask3d) 
            abscm3d = np.array(opt.abscm3d)
            angcm3d = np.array(opt.angcm3d)
            lihe3d = np.array(opt.lihe3d)
            lihe3dxy = np.array(opt.lihe3dxy)
            _U.save_tif_v21(f'overlap3d_±1-[{filename}].tif',overlap_mask3d.astype(np.bool_),run_parameter)
            # _U.save_tif(f'overlap3d_c_±1-[{filename}].tif',abscm3d)
            # _U.save_tif(f'overlap3d_fai_±1-[{filename}].tif',angcm3d)
            _U.save_tif_v21(f'lihe_±1-[{filename}].tif',lihe3d,run_parameter)
            _U.save_tif_v21(f'lihe_±1xy-[{filename}].tif',lihe3dxy,run_parameter)
    del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d

    return cfai,PGxy,PGz,plong,pgz



'#传入PGxy，计算cfai而已'
def parameter_estimation2(opt,ck,PGxy,filename='',RL=False,Debug=False,run_parameter=''):
    
    assert len(ck.shape)==4
    #超参数xy
    opt.pg = int(opt.pxy)         #pg这里是2阶的pxy=r/2
    opt.fanwei = 10#50            #范围在±20之内
    
    #PZ范围 得把Zpadding考虑进去 
    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数
    #并把最中间的置为0
    
    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N ==7 and opt.Nshifts==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')
    elif opt.flag_N ==7 and opt.Nshifts==7 and opt.nz==1: #MP slice的情况
        opt.pgz=0
        opt.fanweiz = 0

    # PGxy = np.zeros((opt.Nangles,opt.Nshifts,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Nshifts))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Nshifts,3)) #
    pgz = 0
    if opt.flag_N==7 and opt.Nshifts==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re#(opt.nz+2*opt.Zpadding )*kuodapz   #*2

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Nshifts+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):# 无解卷积
                RL_output.append(ck[i*opt.Nshifts+j])
        
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        

        print(f'【第{i+1}个方向】:')

        Ns = opt.Nshifts
        # for j in range(1,opt.Nshifts):
        for j in range(1,opt.Nshifts):
            pgx = PGxy[i,j,0]
            pgy = PGxy[i,j,1]          
            if j ==1:  #±2阶求P
                
                print('2jie',pgx,pgy,np.sqrt(pgx**2+pgy**2).round(1))
                #已知亚像素坐标 求C fai

                c,fai,c_EMD= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4,run_parameter=run_parameter) # cfai=3(均值) 4（众数）            
                print('c fai :',c,fai,c_EMD)

                cfai[i,j,:] = np.array([c,fai,c_EMD])

            elif j == 2 :
                PGxy[i,j,:] = -PGxy[i,1,:]  
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,2] = cfai[i,1,2]
                cfai[i,j,1] = -cfai[i,1,1]
                
            elif j == 3 and opt.Nshifts==5:  #SLICE ±1

                print('slice 1th',pgx,pgy,np.sqrt(pgx**2+pgy**2).round(1))
                #求CFAI

                c,fai,c_EMD= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4,run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                cfai[i,j,:] = np.array([c,fai,c_EMD])
            elif j == 4 and opt.Nshifts==5:
                PGxy[i,j,:] = -PGxy[i,3,:]  
                cfai[i,j,0] = cfai[i,3,0]
                cfai[i,j,2] = cfai[i,3,2]
                cfai[i,j,1] = -cfai[i,3,1]


    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime )
    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Nshifts):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)

    if Debug==True: #4

        # overlap_mask2d = np.array(opt.overlap_mask2d) 
        abscm2d = np.array(opt.abscm2d)
        angcm2d = np.array(opt.angcm2d)
        # lihe2d = np.array(opt.lihe2d)
        
        # kzz,kxx,kyy = lihe2d.shape
        # lihe2d[:,kxx//2,kyy//2] = 0.1
        # lihe2d = np.insert(lihe2d,0,values = np.max(lihe2d,axis=0),axis=0) #加一个投影
        # lihe2d = np.insert(lihe2d,0,values = np.min(lihe2d,axis=0),axis=0) #加一个投影
        # _U.save_tif(f'overlap2d_±2-[{filename}].tif',overlap_mask2d.astype(np.bool_))
        # _U.save_tif(f'overlap2d_c_±2-[{filename}].tif',abscm2d)
        # _U.save_tif(f'overlap2d_fai_±2-[{filename}].tif',angcm2d)
        
        # _U.save_tif(f'lihe_±2-[{filename}].tif',lihe2d)

        # if opt.Nshifts==5:
        #     # overlap_mask3d = np.array(opt.overlap_mask3d) 
        #     abscm3d = np.array(opt.abscm3d)
        #     angcm3d = np.array(opt.angcm3d)
        #     # lihe3d = np.array(opt.lihe3d)
        #     # lihe3dxy = np.array(opt.lihe3dxy)
        #     # _U.save_tif(f'overlap3d_±1-[{filename}].tif',overlap_mask3d.astype(np.bool_))
        #     _U.save_tif(f'overlap3d_c_±1-[{filename}].tif',abscm3d)
        #     _U.save_tif(f'overlap3d_fai_±1-[{filename}].tif',angcm3d)
        #     # _U.save_tif(f'lihe_±1-[{filename}].tif',lihe3d)
        #     # _U.save_tif(f'lihe_±1xy-[{filename}].tif',lihe3dxy)
    del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d

    return cfai,PGxy,PGz,plong,pgz





def parameter_estimation3(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    #0级与1级光 估计
    assert len(ck.shape)==4
    #超参数xy
    opt.pg = np.round(opt.pxy,0)        #pg这里是2阶的pxy=r/2
    opt.fanwei = 10            #范围在±20之内
    
    #PZ范围 得把Zpadding考虑进去 
    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数
    #并把最中间的置为0
    
    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N ==7 and opt.Nshifts==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')

    elif opt.flag_N ==7 and opt.Nshifts==7 and opt.nz==1: #MP slice的情况
        opt.pgz=0
        opt.fanweiz = 0

    PGxy = np.zeros((opt.Nangles,opt.Nshifts,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Nshifts))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Nshifts,3)) #
    pgz = 0
    if opt.flag_N==7 and opt.Nshifts==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re#(opt.nz+2*opt.Zpadding )*kuodapz   #*2

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Nshifts+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):# 无解卷积
                RL_output.append(ck[i*opt.Nshifts+j])
        
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        # _U.save_tif('opt.imgf0.tif',_U.visual(RL_output[0]))  #与里程碑7 相同
        # _U.save_tif('opt.imgf_f2.tif',_U.visual(RL_output[1]))
        # _U.save_tif('opt.imgf_f1_f1.tif',_U.visual(RL_output[3]))

        print(f'【第{i+1}个方向】:')
        Ns = opt.Nshifts
        # for j in range(1,opt.Nshifts):
        for j in range(1,opt.Nshifts ):
            # ±2            
            # if j <=1 : #######################

            #     #整数
            #     opt.pg = int(opt.pxy)

            #     pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') # 2D，中间层互相关
            #     print(f'±2整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                
            #     #亚像素
            #     pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #2D 中间层精确估计p
            #     print(f'±2亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
            #     print(pgx,pgy)
            #     #求C fai
            #     # c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值)            
            #     # print('c fai-mean :',c,fai)
            #     print(pgx/(2*opt.nx),pgy/(2*opt.ny))
            #     c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4) # 4（众数）            
            #     print('c fai-zongshu :',c,fai)

                
            #     #保存pgxy c fai
            #     PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
            #     cfai[i,j,:] = np.array([c,fai])
            #     # print(pgx,pgy)

            # elif j == 2: ##########################
            #     PGxy[i,j,:] = -PGxy[i,1,:]  
            #     cfai[i,j,0] = cfai[i,1,0]
            #     cfai[i,j,1] = -cfai[i,1,1]
            
            # #±1
            # elif j>=3 and opt.Nshifts == 5:  #SLICE ±1 ##################################
            #     #1阶 P
            #     opt.pg = int(opt.pxy//2)
            #     pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') #大致估计p
            #     print(f'±1整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                
            #     #亚像素
            #     pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #精确估计p
            #     print(f'±1亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')

            #     #求CFAI
            #     # c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值)            
            #     # print('c2 fai2 :',c,fai)
            #     print(pgx/(2*opt.nx),pgy/(2*opt.ny))
            #     c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4) # 4（众数）            
            #     print('c fai :',c,fai)

            #     PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
            #     cfai[i,j,:] = np.array([c,fai])
            #     print(pgx,pgy)

            # #±1
            # #情况1 知晓±2 拿来直接用 
            # elif j>=3 and opt.Nshifts == 7 : #############################
            #     if j==3 or j==4:
            #         PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
            #     else :
            #         PGxy[i,j,:] = PGxy[i,2,:]/2
            #     pgx =  PGxy[i,j,0] #
            #     pgy =  PGxy[i,j,1]
                
            #     #求pz整数像素
            #     opt.pg = int(opt.pxy//2)
            #     pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN')
            #     print(f'pzint = {pzint},pxint= {pxint},pyint= {pyint}')

            #     #求亚像素
            #     # 自己求pz
            #     if run_parameter['pzzz']==0:
            #          #策略1
            #         pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
            #         # 再次调用
            #         pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
            #     # 给一个亚像素值，但还是要算
            #     elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] !=0 : #按照给的理想值 range=1 再微调一下
            #         if j == 3 or j == 5:
            #             pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
            #             # PGz[i,j] = -run_parameter['pzzz'] 
            #         elif j == 4 or j == 6:
            #             pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
 
            #         #策略1
            #         pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
            #         # 再次调用
            #         pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                    
            #         #新方法 总和/总和 得到c fai
            #         #策略2 
            #         # maxz_y,maxx_y,maxy_y = get_subpz2(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)  #仅在整数上判断方向
                    
            #         #策略3
            #         # maxz_y,maxx_y,maxy_y = get_subpz2_2(opt,(RL_output[0]),(RL_output[j]),(pgx+opt.nx),(pgy+opt.ny),maxz,m=j,func=LiheXYZ)
            #         # pgx =  maxx_y-opt.nx
            #         # pgy =  maxy_y-opt.ny
            #         # pgz =  maxz_y-(opt.nz+2*opt.Zpadding)

            #     # 完全就是给一个值
            #     elif run_parameter['pzzz']!=0 and run_parameter['pzzz_range'] ==0 : #按照给的理想值 range=1 再微调一下
            #         if j == 3 or j == 5:
            #             pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding) 
            #         elif j == 4 or j == 6:
            #             pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
            #     else:
            #         arise('参数错误')



            #     #求CFAI
            #     # c,fai = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=3)                
            #     # print('c2 fai2 :',c,fai)

            #     c,fai = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=4)#三维拟合,求c 和 φ
            #     print(f'c = {c},fai = {fai}')
                
            #     #保存CFAI
            #     cfai[i,j,:] = np.array([c,fai])
            #     PGxy[i,j,:] = np.array([pgx,pgy]) 
            #     PGz[i,j] = pgz        #得到精确的Pz
                
            #     print(f'亚像素pz{j}阶=',PGz[i,j],'亚整数pgx=',pgx,'亚整数pgy = ',pgy) 



            #情况2 自己求±1
            if j<3 and opt.Nshifts==3:      # 0级与1级光干涉 #########################
                opt.pg = int(opt.pxy//2)
                
                #3D互相关求整数
                pzint,pxint,pyint = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN') #3D 做互相关
                print(f'±1整数估计{pzint, pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                

                #方法1 PCA ---------------不太准确
                pzsub,pysub = get_subpzxy_PCA01(opt,(RL_output[j]),j,pxint,pyint,pzint,flag='zy')
                pgz = pzint + pzsub
                pzsub,pxsub = get_subpzxy_PCA01(opt,(RL_output[j]),j,pxint,pyint,pzint,flag='zx')
                pgz2 = pzint + pzsub

                pgy = pyint + pysub  #根据1阶的来算 
                pgx = pxint + pxsub 
                # print(f'亚像素pz{j}阶2=',pgz.round(3),'亚整数pgx=',pgx.round(3),'亚整数pgy = ',pgy.round(3))
                # print(f'亚像素pz2{j}阶2=',pgz2.round(3))

                #+
                #策略1


                # pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pzint,m=j,fanwei=7,func=LiheXYZ) 
                # # 再次调用
                # pgz = get_subpz(opt,i,j,(RL_output[0]),(RL_output[j]),(pgx),(pgy),pgz,m=j,jindu=0.01,fanwei=6,func=LiheXYZ) 
                # if j == 2:
                #     pgz = -run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding) 
                # elif j ==1:
                #     pgz =  run_parameter['pzzz'] #+ (opt.nz+2*opt.Zpadding)
                print(f'!!亚像素pz{j}阶=',pgz,'亚整数pgx=',pgx.round(3),'亚整数pgy = ',pgy.round(3))


                #----------------------------


                #方法2，三个方向互相关策略--不太准确

                    #策略2 
                # pgz,pgx,pgy = get_subpz2(opt,i,j,(RL_output[0]),(RL_output[j]),(pxint),(pyint),pzint,m=j,func=LiheXYZ)  #仅在整数上判断方向
                

                    #策略3
                # pgz,pgx,pgy = get_subpz2_2(opt,i,j,(RL_output[0]),(RL_output[j]),(pxint),(pyint),pzint,m=j,func=LiheXYZ)

                # print(f'±1亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                # print(f'±1亚像素估计pz={pgz},')

                #--------------------------------------


                #方法3，xy 投影计算，Z投影计算




                # 计算cfai
                # c,fai = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=3)                
                # print('c2 fai2 :',c,fai)

                c,fai = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=4)#三维拟合,求c 和 φ
                print(f'c = {c},fai = {fai}')
                
                #保存 P CFAI
                cfai[i,j,:] = np.array([c,fai])
                PGz[i,j] = pgz
                PGxy[i,j,0] = pgx
                PGxy[i,j,1] = pgy

    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime)
    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Nshifts):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)
    

    if Debug==True: #4

        overlap_mask2d = np.array(opt.overlap_mask2d) 
        abscm2d = np.array(opt.abscm2d)
        angcm2d = np.array(opt.angcm2d)
        lihe2d = np.array(opt.lihe2d)
        
        kzz,kxx,kyy = lihe2d.shape
        lihe2d[:,kxx//2,kyy//2] = 0.3
        lihe2d = np.insert(lihe2d,0,values = np.max(lihe2d,axis=0),axis=0) #加一个投影
        _U.save_tif(f'overlap2d_±2-[{filename}].tif',overlap_mask2d.astype(np.bool_))
        _U.save_tif(f'overlap2d_c_±2-[{filename}].tif',abscm2d)
        _U.save_tif(f'overlap2d_fai_±2-[{filename}].tif',angcm2d)
        
        _U.save_tif(f'lihe_±2-[{filename}].tif',lihe2d)

        if opt.Nshifts==7:
            overlap_mask3d = np.array(opt.overlap_mask3d) 
            abscm3d = np.array(opt.abscm3d)
            angcm3d = np.array(opt.angcm3d)
            lihe3d = np.array(opt.lihe3d)
            lihe3dxy = np.array(opt.lihe3dxy)
            _U.save_tif(f'overlap3d_±1-[{filename}].tif',overlap_mask3d.astype(np.bool_))
            _U.save_tif(f'overlap3d_c_±1-[{filename}].tif',abscm3d)
            _U.save_tif(f'overlap3d_fai_±1-[{filename}].tif',angcm3d)
            _U.save_tif(f'lihe_±1-[{filename}].tif',lihe3d)
            _U.save_tif(f'lihe_±1xy-[{filename}].tif',lihe3dxy)
    del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d

    return cfai,PGxy,PGz,plong,pgz



def parameter_estimation_1(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    #先求1阶再给2阶
     #正常2阶求1阶
    assert len(ck.shape)==4
    #超参数xy
    if run_parameter['pg']!=0:
        opt.pg = run_parameter['pg']
        opt.fanwei = run_parameter['fanwei']    
    else:
        opt.pg = int(opt.pxy)
        opt.fanwei = run_parameter['fanwei']


    #PZ范围 得把Zpadding考虑进去 
    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数
    #并把最中间的置为0
    
    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N ==7 and opt.Nshifts==7 and opt.nz!=1 and (opt.fanweiz <0 or opt.pgz>=opt.nz):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')
    elif opt.flag_N ==7 and opt.Nshifts==7 and opt.nz==1: #MP slice的情况
        opt.pgz=0
        opt.fanweiz = 0

    PGxy = np.zeros((opt.Nangles,opt.Nshifts,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Nshifts))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Nshifts,3)) #
    pgz = 0
    if opt.flag_N==7 and opt.Nshifts==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re#(opt.nz+2*opt.Zpadding )*kuodapz   #*2

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Nshifts+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):# 无解卷积
                RL_output.append(ck[i*opt.Nshifts+j])
        
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        # _U.save_tif('opt.imgf0.tif',_U.visual(RL_output[0]))  #与里程碑7 相同
        # _U.save_tif('opt.imgf_f2.tif',_U.visual(RL_output[1]))
        # _U.save_tif('opt.imgf_f1_f1.tif',_U.visual(RL_output[3]))

        print(f'【第{i+1}个方向】:')

        for j in [3,4,1,2]:
            if j <=2:
                ########################-------2阶赋值
                pgx = PGxy[i,j,0]
                pgy = PGxy[i,j,1]

                print(f'±2亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')

                ########################

                #求C fai
                c,fai,c_EMD= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4,run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                
                #保存pgxy c fai
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai,c_EMD])
            elif j==3:
                ##########-------------------1阶重新计算法--------------
                #1阶 P
                opt.pg = int(opt.pxy//2)
                pxint,pyint = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') #大致估计p
                print(f'±1整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                
                #亚像素
                pgx,pgy = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),pxint,pyint,func=LiheXYZ) #精确估计p
                print(f'±1亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')

                PGxy[i,1,:] = 2*np.array([pgx,pgy])  #给2阶整数倍的p/2

                ##########-------------------直接2阶赋值法--------------

                # #用2阶数直接赋值
                # PGxy[i,j,:] = PGxy[i,j//2,:]/2  # -2 2 -1 1
                #                                 #  1 2  3 4
                # pgx = PGxy[i,j,0]
                # pgy = PGxy[i,j,1]

                ##########-------------------求c-----------------------
                #求CFAI
                c,fai,c_EMD= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4,run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                #保存
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                # print(j,PGxy)

            elif j==4:
                PGxy[i,j,:] = -PGxy[i,3,:]
                pgx = PGxy[i,j,0]
                pgy = PGxy[i,j,1]

                print(f'±1亚像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                PGxy[i,2,:] = 2*np.array([pgx,pgy])  #给2阶整数倍的p/2

                #求CFAI
                c,fai,c_EMD= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4,run_parameter=run_parameter) # 4（众数）            
                print('c fai :',c,fai,c_EMD)

                #保存
                # PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai,c_EMD])
                # print(j,PGxy)

    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime)
    plong=[]
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2) )
        for j in range(3,opt.Nshifts):   #±1
            plong.append( np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2)
    plong = np.mean(np.array(plong))
    

    print(PGxy)
    print(PGz)
    print(cfai)
    return cfai,PGxy,PGz,plong,pgz

# 根据互相关求得整数

# 得到整数后
def parameter_estimation_PCA(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    assert len(ck.shape)==4
   
    opt.pg = int(opt.pxy)         #pg这里是2阶的pxy=r/2
    opt.fanwei = 10            #范围在±20之内

    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数

    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N ==7 and opt.Nshifts==7 and (opt.fanweiz <0 or opt.pgz>=opt.nz):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')


    PGxy = np.zeros((opt.Nangles,opt.Nshifts,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Nshifts))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Nshifts,2)) #
    pgz = 0

    if opt.flag_N==7 and opt.Nshifts==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Nshifts+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output = np.copy(ck[i*opt.Nshifts:(i+1)*opt.Nshifts])
            # RL_output=[]
            # for j in range(opt.Nshifts):# 无解卷积
            #     RL_output.append(ck[i*opt.Nshifts+j])
        
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        # _U.save_tif('opt.imgf0.tif',_U.visual(RL_output[0]))  #与里程碑7 相同
        # _U.save_tif('opt.imgf_f2.tif',_U.visual(RL_output[1]))
        # _U.save_tif('opt.imgf_f1_f1.tif',_U.visual(RL_output[3]))

        print(f'【第{i+1}个方向】:')
        Ns = opt.Nshifts
        # for j in range(1,opt.Nshifts):
        for j in range(1,opt.Nshifts):
            if j <=1 :  #±2阶求P
                #整数
                opt.pg = int(opt.pxy)
                maxx,maxy = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') #大致估计p
                pxint = maxx - opt.nx 
                pyint = maxy - opt.ny 
                print(f'±2整数估计{maxx- opt.nx , maxy- opt.ny},长度为：{np.sqrt((maxx- opt.nx)**2+(maxy- opt.ny)**2)}')
                lihe2d = np.array(opt.lihe2d)
                _U.save_tif('lihe2d.tif',lihe2d)
                #亚像素
                print((RL_output[0]).shape)

                ##不同的就是这一个函数，仅传入高频分量+整数移动量
                pxsub,pysub = get_subpzxy_PCA(opt,(RL_output[j]),j,pxint,pyint) #精确估计p
                ###返回亚像素值

                pgx = pxint+pxsub
                pgy = pyint+pysub
                print(f'pgx={pgx.round(3)},pgy={pgy.round(3)}')
                #已知亚像素坐标 求C fai
                c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值) 4（众数）            
                print('c2 fai2 :',c,fai)

                c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4) # cfai=3(均值) 4（众数）            
                print('c fai :',c,fai)

                # c,fai= LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值) 4（众数）            
                # print('c2 fai2 :',c,fai)
                print(f'±2压像素估计{pgx , pgy},长度为：{np.sqrt((pgx)**2+(pgy)**2)}')
                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai])
                print(pgx,pgy)
            elif j == 2:
                PGxy[i,j,:] = -PGxy[i,1,:]  
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,1] = -cfai[i,1,1]
            elif j>=3 and opt.Nshifts==5:  #SLICE ±1
                #1阶 P
                opt.pg = int(opt.pxy//2)
                maxx,maxy = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') #大致估计p
                print(f'±2整数估计{maxx- opt.nx , maxy- opt.ny},长度为：{np.sqrt((maxx- opt.nx)**2+(maxy- opt.ny)**2)}')
                
                #亚像素
                maxx_y,maxy_y = get_subpxy(opt,(RL_output[0]),(RL_output[j]),maxx,maxy,func=LiheXYZ) #精确估计p
                pgx =  maxx_y-opt.nx
                pgy =  maxy_y-opt.ny

                #求CFAI

                c,fai= LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值)            
                print('c2 fai2 :',c,fai)

                c,fai= LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4) # 4（众数）            
                print('c fai :',c,fai)

                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai])
                print(pgx,pgy)

            elif j>=3 and opt.flag_N==7 and opt.Nshifts==7:      # -1，-1。 pgx，pgy用2阶的， pz这里尝试 取横向一般位置 竖着互相关，得到整数pz
                opt.pg = int(opt.pxy//2)
                #主要是求pz pxy用之前结果
                if j==3 or j==4:
                    PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
                else :
                    PGxy[i,j,:] = PGxy[i,2,:]/2

                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]
                print('±2的一半为：',pgx,pgy)
                #【整数求pz】
                # maxz,_ = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='SHAO') #竖着取，做互相关
                maxz,_ = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN') #竖着取，做互相关
                # maxz,_ = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='HIFI') #竖着取，做互相关

                # PGxy[i,j,:] = np.array([pgx,pgy])
                pzint = maxz-(opt.nz+2*opt.Zpadding)
                # if pzint<0:
                #     pzint=pzint-1
                # else:
                #     pzint +=1
                print(f'整数{j}阶pz为',pzint) 

                if run_parameter['pzzz']!=0: #按照给的理想值
                    if j == 3 or j == 5:
                        maxz = -run_parameter['pzzz'] + (opt.nz+2*opt.Zpadding)
                        # PGz[i,j] = -run_parameter['pzzz'] 
                    elif j == 4 or j == 6:
                        maxz =  run_parameter['pzzz'] + (opt.nz+2*opt.Zpadding)

                #【求亚像素pzsub】
                pxint = int(pgx)
                pyint = int(pgy)
                print()
                print('zy')
                pzsub,pysub= get_subpzxy_PCA(opt,(RL_output[j]),j,pxint,pyint,pzint,flag='zy')

                pgz = pzint+pzsub
                # pgy = pyint+pysub  #根据1阶的来算 
                print(f'亚像素pz{j}阶1=',pgz.round(3),'亚整数pgx=',pgx.round(3),'亚整数pgy = ',pgy.round(3))

                print()
                print('zx')
                pzsub,pxsub= get_subpzxy_PCA(opt,(RL_output[j]),j,pxint,pyint,pzint,flag='zx')
                print(f'亚像素pzsub{j}阶1=',pzsub.round(3))

                pgz = pzint+pzsub
                # pgy = pyint+pysub  #根据1阶的来算 
                # pgx = pxint+pxsub 
                print(f'亚像素pz{j}阶2=',pgz.round(3),'亚整数pgx=',pgx.round(3),'亚整数pgy = ',pgy.round(3))
                print(f'验证py的亚像素pyint+pysub={pyint}+{pysub} = {pyint+pysub}') 
                #print(f'±1亚像素估计[{pgx,pgy}],长度为：{np.sqrt(pgx**2+pgy**2)}')
                PGz[i,j] = pgz
                PGxy[i,j,0] = pgx
                PGxy[i,j,1] = pgy

                c,fai = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=3)                
                print('c2 fai2 :',c,fai)

                c,fai = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=4)#三维拟合,求c 和 φ
                print(f'c = {c},fai = {fai}')
                
                cfai[i,j,:] = np.array([c,fai])
    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime)
    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Nshifts):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)
    

    if Debug==True:

        overlap_mask2d = np.array(opt.overlap_mask2d) 
        abscm2d = np.array(opt.abscm2d)
        angcm2d = np.array(opt.angcm2d)
        lihe2d = np.array(opt.lihe2d)
        _U.save_tif(f'overlap2d_±2-[{filename}].tif',overlap_mask2d.astype(np.bool_))
        _U.save_tif(f'overlap2d_c_±2-[{filename}].tif',abscm2d)
        _U.save_tif(f'overlap2d_fai_±2-[{filename}].tif',angcm2d)
        _U.save_tif(f'lihe_±2-[{filename}].tif',lihe2d)
        if opt.Nshifts==7:
            overlap_mask3d = np.array(opt.overlap_mask3d) 
            abscm3d = np.array(opt.abscm3d)
            angcm3d = np.array(opt.angcm3d)
            lihe3d = np.array(opt.lihe3d)
            lihe3dxy = np.array(opt.lihe3dxy)
            _U.save_tif(f'overlap3d_±1-[{filename}].tif',overlap_mask3d.astype(np.bool_))
            _U.save_tif(f'overlap3d_c_±1-[{filename}].tif',abscm3d)
            _U.save_tif(f'overlap3d_fai_±1-[{filename}].tif',angcm3d)
            _U.save_tif(f'lihe_±1-[{filename}].tif',lihe3d)
            _U.save_tif(f'lihe_±1xy-[{filename}].tif',lihe3dxy)
    del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d

    return cfai,PGxy,PGz,plong,pgz




def parameter_estimation_PCA_COR(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    assert len(ck.shape)==4
   
    opt.pg = int(opt.pxy)         #pg这里是2阶的pxy=r/2
    opt.fanwei = 10            #范围在±20之内

    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数

    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N ==7 and opt.Nshifts==7 and (opt.fanweiz <0 or opt.pgz>=opt.nz):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')


    PGxy = np.zeros((opt.Nangles,opt.Nshifts,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Nshifts))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Nshifts,2)) #
    pgz = 0

    if opt.flag_N==7 and opt.Nshifts==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(3):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Nshifts+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output = np.copy(ck[i*opt.Nshifts:(i+1)*opt.Nshifts])
            # RL_output=[]
            # for j in range(opt.Nshifts):# 无解卷积
            #     RL_output.append(ck[i*opt.Nshifts+j])
        
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        # _U.save_tif('opt.imgf0.tif',_U.visual(RL_output[0]))  #与里程碑7 相同
        # _U.save_tif('opt.imgf_f2.tif',_U.visual(RL_output[1]))
        # _U.save_tif('opt.imgf_f1_f1.tif',_U.visual(RL_output[3]))

        print(f'【第{i+1}个方向】:')
        Ns = opt.Nshifts
        # for j in range(1,opt.Nshifts):
        for j in range(1,opt.Nshifts):
            if j <=1 :  #±2阶求P
                #整数
                opt.pg = int(opt.pxy)
                maxx,maxy = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') #大致估计p
                pxint = maxx - opt.nx 
                pyint = maxy - opt.ny 
                print(f'±2整数估计{maxx- opt.nx , maxy- opt.ny},长度为：{np.sqrt((maxx- opt.nx)**2+(maxy- opt.ny)**2)}')
                lihe2d = np.array(opt.lihe2d)
                _U.save_tif('lihe2d.tif',lihe2d)

                #亚像素
                print((RL_output[0]).shape)
                maxx_y,maxy_y = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),maxx,maxy,func=LiheXYZ) #精确估计p
                pgx =  maxx_y-opt.nx
                pgy =  maxy_y-opt.ny
                print(f'pgx={pgx.round(3)},pgy={pgy.round(3)}')
                #已知亚像素坐标 求C fai

                c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值) 4（众数）            
                print('c2 fai2 :',c,fai)

                c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4) # cfai=3(均值) 4（众数）            
                print('c fai :',c,fai)

                # c,fai= LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值) 4（众数）            
                # print('c2 fai2 :',c,fai)

                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai])
                print(pgx,pgy)
            elif j == 2:
                PGxy[i,j,:] = -PGxy[i,1,:]  
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,1] = -cfai[i,1,1]
            elif j>=3 and opt.Nshifts==5:  #SLICE ±1
                #1阶 P
                opt.pg = int(opt.pxy//2)
                maxx,maxy = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') #大致估计p
                print(f'±2整数估计{maxx- opt.nx , maxy- opt.ny},长度为：{np.sqrt((maxx- opt.nx)**2+(maxy- opt.ny)**2)}')
                
                #亚像素
                maxx_y,maxy_y = get_subpxy(opt,i,j,(RL_output[0]),(RL_output[j]),maxx,maxy,func=LiheXYZ) #精确估计p
                pgx =  maxx_y-opt.nx
                pgy =  maxy_y-opt.ny

                #求CFAI

                c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值)            
                print('c2 fai2 :',c,fai)

                c,fai= LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4) # 4（众数）            
                print('c fai :',c,fai)

                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai])
                print(pgx,pgy)

            elif j>=3 and opt.flag_N==7 and opt.Nshifts==7:      # -1，-1。 pgx，pgy用2阶的， pz这里尝试 取横向一般位置 竖着互相关，得到整数pz
                opt.pg = int(opt.pxy//2)
                #主要是求pz pxy用之前结果
                if j==3 or j==4:
                    PGxy[i,j,:] = PGxy[i,1,:]/2  #-1阶只有-2阶的一半
                else :
                    PGxy[i,j,:] = PGxy[i,2,:]/2

                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]
                print('±2的一半为：',pgx,pgy)
                #【整数求pz】
                # maxz,_ = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='SHAO') #竖着取，做互相关
                maxz,_ = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN') #竖着取，做互相关
                # maxz,_ = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='HIFI') #竖着取，做互相关

                # PGxy[i,j,:] = np.array([pgx,pgy])
                pzint = maxz-(opt.nz+2*opt.Zpadding)
                print(f'整数{j}阶pz为',pzint) 

                if run_parameter['pzzz']!=0: #按照给的理想值
                    if j == 3 or j == 5:
                        maxz = -run_parameter['pzzz'] + (opt.nz+2*opt.Zpadding)
                        # PGz[i,j] = -run_parameter['pzzz'] 
                    elif j == 4 or j == 6:
                        maxz =  run_parameter['pzzz'] + (opt.nz+2*opt.Zpadding)

                #【求亚像素pzsub】
                pxint = int(pgx)
                pyint = int(pgy)
                print()
                print('zy')
                pzsub,pysub= get_subpzxy_PCA(opt,(RL_output[j]),j,pxint,pyint,pzint,flag='zy')

                pgz = pzint+pzsub
                # pgy = pyint+pysub  #根据1阶的来算 
                print(f'亚像素pz{j}阶1=',pgz.round(3),'亚整数pgx=',pgx.round(3),'亚整数pgy = ',pgy.round(3))

                print()
                print('zx')
                pzsub,pxsub= get_subpzxy_PCA(opt,(RL_output[j]),j,pxint,pyint,pzint,flag='zx')
                print(f'亚像素pzsub{j}阶1=',pzsub.round(3))

                pgz = pzint+pzsub
                # pgy = pyint+pysub  #根据1阶的来算 
                # pgx = pxint+pxsub 
                print(f'亚像素pz{j}阶2=',pgz.round(3),'亚整数pgx=',pgx.round(3),'亚整数pgy = ',pgy.round(3))
                print(f'验证py的亚像素pyint+pysub={pyint}+{pysub} = {pyint+pysub}') 
                #print(f'±1亚像素估计[{pgx,pgy}],长度为：{np.sqrt(pgx**2+pgy**2)}')
                PGz[i,j] = pgz
                PGxy[i,j,0] = pgx
                PGxy[i,j,1] = pgy
                
                c,fai = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=3)                
                print('c2 fai2 :',c,fai)

                c,fai = LiheXYZ(opt,i,j,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=4)#三维拟合,求c 和 φ
                print(f'c = {c},fai = {fai}')
                
                cfai[i,j,:] = np.array([c,fai])
    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime)
    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Nshifts):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)
    

    if Debug==True:

        overlap_mask2d = np.array(opt.overlap_mask2d) 
        abscm2d = np.array(opt.abscm2d)
        angcm2d = np.array(opt.angcm2d)
        lihe2d = np.array(opt.lihe2d)
        _U.save_tif(f'overlap2d_±2-[{filename}].tif',overlap_mask2d.astype(np.bool_))
        _U.save_tif(f'overlap2d_c_±2-[{filename}].tif',abscm2d)
        _U.save_tif(f'overlap2d_fai_±2-[{filename}].tif',angcm2d)
        _U.save_tif(f'lihe_±2-[{filename}].tif',lihe2d)
        if opt.Nshifts==7:
            overlap_mask3d = np.array(opt.overlap_mask3d) 
            abscm3d = np.array(opt.abscm3d)
            angcm3d = np.array(opt.angcm3d)
            lihe3d = np.array(opt.lihe3d)
            lihe3dxy = np.array(opt.lihe3dxy)
            _U.save_tif(f'overlap3d_±1-[{filename}].tif',overlap_mask3d.astype(np.bool_))
            _U.save_tif(f'overlap3d_c_±1-[{filename}].tif',abscm3d)
            _U.save_tif(f'overlap3d_fai_±1-[{filename}].tif',angcm3d)
            _U.save_tif(f'lihe_±1-[{filename}].tif',lihe3d)
            _U.save_tif(f'lihe_±1xy-[{filename}].tif',lihe3dxy)
    del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d

    return cfai,PGxy,PGz,plong,pgz



def parameter_estimation_PCA2(opt,ck,filename='',RL=False,Debug=False,run_parameter=''):
    '先算1阶，再赋值给2阶'
    '目前效果不太好'
    assert len(ck.shape)==4
   
    opt.pg = int(opt.pxy)         #pg这里是2阶的pxy=r/2
    opt.fanwei = 10            #范围在±20之内

    opt.cutoff = 1e-4
    opt.pgz = round(opt.pz+opt.Zpadding+2) # opt.PZ= 2.6 留下0~opt.pgz中间的层数  缩小范围
    opt.fanweiz = 0 #round(opt.pz-2) #留下 fanweiz~pgz 层数

    print(f'opt.pg = {opt.pg}')
    print(f'opt.fanwei = {opt.fanwei}')
    print(f'opt.pgz = {opt.pgz}')
    print(f'opt.fanweiz = {opt.fanweiz}')
    
    if opt.flag_N ==7 and opt.Nshifts==7 and (opt.fanweiz <0 or opt.pgz>=opt.nz):
        print('pgz=',opt.pgz)
        print("fanweiz=",opt.fanweiz)
        raise('fanweiz error')


    PGxy = np.zeros((opt.Nangles,opt.Nshifts,2)) #只用算出 2阶的px，py，还有1阶的pz就行
    PGz = np.zeros((opt.Nangles,opt.Nshifts))               # 0，-2，2 ；-1-1 ；-1 1 ；1 -1 ；1 1 ；
                            # 0， 1, 2    3    4        5    6
    cfai = np.zeros((opt.Nangles,opt.Nshifts,2)) #
    pgz = 0

    if opt.flag_N==7 and opt.Nshifts==7 :
        kuodapz = 2           # 当 raw data 在Z轴上降采样时，这里需要设置为2； 
    else:
        kuodapz = 1           # 当在Z轴上没有降采样时设置为1

    nz = opt.nz_re

    try:
        ck=ck.numpy()
    except:
        pass
    
    opt.lihe2d = []
    opt.overlap_mask2d = [] 
    opt.abscm2d = []
    opt.angcm2d = []

    opt.lihe3d = []
    opt.lihe3dxy = []
    opt.overlap_mask3d = [] 
    opt.abscm3d = []
    opt.angcm3d = []
    ''' <<<<<<<<<<<<<<<parameter estimate >>>>>>>>>>>>>>>>>>'''
    starttime = datetime.datetime.now()
    for i in range(1):         #这个三表示三个方向，做测试可改为1
        
        if RL:
            print('参数估计时，执行RL解卷积')
            RL_output=[]
            for j in range(opt.Nshifts):#RL解卷积
                RL_output.append(_U.RL_deconvolve(ck[i*opt.Nshifts+j], opt.OTF,fre = True) )
        else:
            print('参数估计时，不执行RL解卷积')
            RL_output = np.copy(ck[i*opt.Nshifts:(i+1)*opt.Nshifts])
            # RL_output=[]
            # for j in range(opt.Nshifts):# 无解卷积
            #     RL_output.append(ck[i*opt.Nshifts+j])
        
        RL_output = np.array(RL_output)
        assert len(RL_output.shape)==4
        cfai[i,0,0] = 1 
        
        # _U.save_tif('opt.imgf0.tif',_U.visual(RL_output[0]))  #与里程碑7 相同
        # _U.save_tif('opt.imgf_f2.tif',_U.visual(RL_output[1]))
        # _U.save_tif('opt.imgf_f1_f1.tif',_U.visual(RL_output[3]))

        print(f'【第{i+1}个方向】:')
        Ns = opt.Nshifts
        # for j in range(1,opt.Nshifts):
        # for j in range(1,opt.Nshifts):
        for j in [3,4,5,6,1,2]:
            if j <=1 :  #±2阶求P
                #整数
                if opt.Nshifts==7 and PGxy[i,3,0]!=0:
                    PGxy[i,j,:] = PGxy[i,3,:]*2  #-1阶只有-2阶的一半
                

                # opt.pg = int(opt.pxy)
                # maxx,maxy = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') #大致估计p
                
                # pxint = maxx - opt.nx 
                # pyint = maxy - opt.ny 
                # print(f'±2整数估计{maxx- opt.nx , maxy- opt.ny},长度为：{np.sqrt((maxx- opt.nx)**2+(maxy- opt.ny)**2)}')
                # lihe2d = np.array(opt.lihe2d)
                # _U.save_tif('lihe2d.tif',lihe2d)
                # #亚像素
                # print((RL_output[0]).shape)

                # ##不同的就是这一个函数，仅传入高频分量+整数移动量
                # pxsub,pysub = get_subpzxy_PCA(opt,(RL_output[j]),j,pxint,pyint) #精确估计p
                # ###返回亚像素值

                # pgx = pxint+pxsub
                # pgy = pyint+pysub
                # print(f'pgx={pgx.round(3)},pgy={pgy.round(3)}')
                
                pgx =  PGxy[i,j,0] #
                pgy =  PGxy[i,j,1]

                #已知亚像素坐标 求C fai
                c,fai= LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值) 4（众数）            
                print('c2 fai2 :',c,fai)

                c,fai= LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4) # cfai=3(均值) 4（众数）            
                print('c fai :',c,fai)

                # c,fai= LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值) 4（众数）            
                # print('c2 fai2 :',c,fai)

                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai])
                print(pgx,pgy)
            elif j == 2:
                PGxy[i,j,:] = -PGxy[i,1,:]  
                cfai[i,j,0] = cfai[i,1,0]
                cfai[i,j,1] = -cfai[i,1,1]
            elif j>=3 and opt.Nshifts==5:  #SLICE ±1
                #1阶 P
                opt.pg = int(opt.pxy//2)
                maxx,maxy = get_intpxy(opt,RL_output[0],RL_output[j],i,j,methods='FAN') #大致估计p
                print(f'±2整数估计{maxx- opt.nx , maxy- opt.ny},长度为：{np.sqrt((maxx- opt.nx)**2+(maxy- opt.ny)**2)}')
                
                #亚像素
                maxx_y,maxy_y = get_subpxy(opt,(RL_output[0]),(RL_output[j]),maxx,maxy,func=LiheXYZ) #精确估计p
                pgx =  maxx_y-opt.nx
                pgy =  maxy_y-opt.ny

                #求CFAI

                c,fai= LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=3) # cfai=3(均值)            
                print('c2 fai2 :',c,fai)

                c,fai= LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),cfai=4) # 4（众数）            
                print('c fai :',c,fai)

                PGxy[i,j,:] = np.array([pgx,pgy])  #保存数据
                cfai[i,j,:] = np.array([c,fai])
                print(pgx,pgy)

            elif j>=3 and opt.flag_N==7 and opt.Nshifts==7:      # -1，-1。 pgx，pgy用2阶的， pz这里尝试 取横向一般位置 竖着互相关，得到整数pz
                # opt.pg = int(opt.pxy//2)

                #利用2阶来求整数pgxy
                if j==3 or j==4:
                    maxx,maxy = get_intpxy(opt,RL_output[0],RL_output[1],i,j,methods='FAN') #大致估计p
                elif j==5 or j==6:    
                    maxx,maxy = get_intpxy(opt,RL_output[0],RL_output[2],i,j,methods='FAN') #大致估计p

                pxint = (maxx - opt.nx)
                pyint = (maxy - opt.ny)
                print(f'±2整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')

                pxint = (maxx - opt.nx)//2
                pyint = (maxy - opt.ny)//2
                print(f'±1整数估计{pxint , pyint},长度为：{np.sqrt((pxint)**2+(pyint)**2)}')
                

                #求亚像素
                pxsub,pysub = get_subpzxy_PCA(opt,(RL_output[j]),j,pxint,pyint) #精确估计p
                pgx = pxint+pxsub
                pgy = pyint+pysub
                print(f'pgx={pgx.round(3)},pgy={pgy.round(3)}')

                
                #【整数求pz】
                # maxz,_ = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='SHAO') #竖着取，做互相关
                maxz,_ = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='FAN') #竖着取，做互相关
                # maxz,_ = get_intpz(opt,(RL_output[0]),(RL_output[j]),i,j,methods='HIFI') #竖着取，做互相关

                pzint = maxz-(opt.nz+2*opt.Zpadding)
                print(f'整数{j}阶pz为',pzint) 

                if run_parameter['pzzz']!=0: #按照给的理想值
                    if j == 3 or j == 5:
                        maxz = -run_parameter['pzzz'] + (opt.nz+2*opt.Zpadding)
                        # PGz[i,j] = -run_parameter['pzzz'] 
                    elif j == 4 or j == 6:
                        maxz =  run_parameter['pzzz'] + (opt.nz+2*opt.Zpadding)

                #【求亚像素pzsub】
                pxint = int(pgx)
                pyint = int(pgy)
                print()
                print('zy')
                pzsub,pysub= get_subpzxy_PCA(opt,(RL_output[j]),j,pxint,pyint,pzint,flag='zy')

                pgz = pzint+pzsub
                # pgy = pyint+pysub  #根据1阶的来算 
                print(f'亚像素pz{j}阶1=',pgz.round(3),'亚整数pgx=',pgx.round(3),'亚整数pgy = ',pgy.round(3))

                print()
                print('zx')
                pzsub,pxsub= get_subpzxy_PCA(opt,(RL_output[j]),j,pxint,pyint,pzint,flag='zx')
                print(f'亚像素pzsub{j}阶1=',pzsub.round(3))

                pgz = pzint+pzsub
                # pgy = pyint+pysub  #根据1阶的来算 
                # pgx = pxint+pxsub 
                print(f'亚像素pz{j}阶2=',pgz.round(3),'亚整数pgx=',pgx.round(3),'亚整数pgy = ',pgy.round(3))
                print(f'验证py的亚像素pyint+pysub={pyint}+{pysub} = {pyint+pysub}') 
                #print(f'±1亚像素估计[{pgx,pgy}],长度为：{np.sqrt(pgx**2+pgy**2)}')
                PGz[i,j] = pgz
                PGxy[i,j,0] = pgx
                PGxy[i,j,1] = pgy

                c,fai = LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=3)                
                print('c2 fai2 :',c,fai)

                c,fai = LiheXYZ(opt,(RL_output[0]),(RL_output[j]),pgx/(2*opt.nx),pgy/(2*opt.ny),pgz/(nz),m=j,cfai=4)#三维拟合,求c 和 φ
                print(f'c = {c},fai = {fai}')
                
                cfai[i,j,:] = np.array([c,fai])
    endtime = datetime.datetime.now()
    print ('参数估计时间：',endtime - starttime)
    plong=0
    for i in range(opt.Nangles):
        for j in range(1,3):   #±2
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)
        for j in range(3,opt.Nshifts):   #±1
            plong += np.sqrt(PGxy[i,j,0]**2+PGxy[i,j,1]**2)*2
    plong = plong/(opt.nc-3)
    

    if Debug==True:

        overlap_mask2d = np.array(opt.overlap_mask2d) 
        abscm2d = np.array(opt.abscm2d)
        angcm2d = np.array(opt.angcm2d)
        lihe2d = np.array(opt.lihe2d)
        _U.save_tif(f'overlap2d_±2-[{filename}].tif',overlap_mask2d.astype(np.bool_))
        _U.save_tif(f'overlap2d_c_±2-[{filename}].tif',abscm2d)
        _U.save_tif(f'overlap2d_fai_±2-[{filename}].tif',angcm2d)
        _U.save_tif(f'lihe_±2-[{filename}].tif',lihe2d)
        if opt.Nshifts==7:
            overlap_mask3d = np.array(opt.overlap_mask3d) 
            abscm3d = np.array(opt.abscm3d)
            angcm3d = np.array(opt.angcm3d)
            lihe3d = np.array(opt.lihe3d)
            lihe3dxy = np.array(opt.lihe3dxy)
            _U.save_tif(f'overlap3d_±1-[{filename}].tif',overlap_mask3d.astype(np.bool_))
            _U.save_tif(f'overlap3d_c_±1-[{filename}].tif',abscm3d)
            _U.save_tif(f'overlap3d_fai_±1-[{filename}].tif',angcm3d)
            _U.save_tif(f'lihe_±1-[{filename}].tif',lihe3d)
            _U.save_tif(f'lihe_±1xy-[{filename}].tif',lihe3dxy)
    del opt.overlap_mask2d,opt.abscm2d,opt.angcm2d
    del opt.overlap_mask3d,opt.abscm3d,opt.angcm3d

    return cfai,PGxy,PGz,plong,pgz




#仿真数据的相位理论值
 #±2 fai = 1.04    ±1 fai 1.46，-0.418，0.418，-1.46

