import numpy as np
from numpy.fft import fftshift,ifftshift
from fractions import Fraction
from skimage import restoration
# from flowdec import data as fd_data
# from flowdec import restoration as fd_restoration
# from flowdec import psf as fd_psf

import WindowF as _WF
import Utils as _U
#分离ck
def sepmatrix7_7(opt,Faixy=None): #创建cp矩阵
    '''多层使用全部7个相位，按照2pi7进行分离
    '''
    print('MP使用全部相位：0246135 相位间隔：',Fraction(2, opt.Nshifts))
    fai = 2/opt.Nshifts
    Dfai = np.pi*fai
    ps = np.array((0,2*Dfai,4*Dfai,6*Dfai,1*Dfai,3*Dfai,5*Dfai))
    phase_3 = 0.40*np.pi
    print(f'phase_3:{phase_3*180/np.pi}')
    psz = np.array((phase_3,-phase_3))
    # psz = np.array((-np.pi/2,np.pi/2))

    cp = np.array([[1,1,1,1,1,1,1],
                   [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1])          ,np.exp(1j*ps[1])          ,np.exp(-1j*ps[1])          ,  np.exp(-1j*ps[1])],
                   [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2])          ,np.exp(1j*ps[2])          ,np.exp(-1j*ps[2])          ,  np.exp(-1j*ps[2])],
                   [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3])          ,np.exp(1j*ps[3])          ,np.exp(-1j*ps[3])          ,  np.exp(-1j*ps[3])],
                   [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*(ps[4]+psz[0])),np.exp(1j*(ps[4]+psz[1])),np.exp(1j*(-ps[4]+psz[0])),np.exp(1j*(-ps[4]+psz[1]))],
                   [1,np.exp(2j*ps[5]),np.exp(-2j*ps[5]),np.exp(1j*(ps[5]+psz[0])),np.exp(1j*(ps[5]+psz[1])),np.exp(1j*(-ps[5]+psz[0])),np.exp(1j*(-ps[5]+psz[1])) ],
                   [1,np.exp(2j*ps[6]),np.exp(-2j*ps[6]),np.exp(1j*(ps[6]+psz[0])),np.exp(1j*(ps[6]+psz[1])),np.exp(1j*(-ps[6]+psz[0])),np.exp(1j*(-ps[6]+psz[1]))]]) 
    
    cpx,cpy =cp.shape
    print('cp的秩为：',np.linalg.matrix_rank(cp))
    return np.repeat(cp.reshape(1,cpx,cpy),opt.Nangles,axis=0) #给CP增加一维度，表示方向

def sepmatrix7_5(opt,Faixy=None): #创建cp矩阵
    ''' 多层挑选5个相位，按照2pi7进行分离 without 轴向相位
    '''
    # if isinstance(Faixy,np.ndarray):
    print('MP使用全部相位：02461 相位间隔：',Fraction(2, 7))
    fai  = 2/7

    Dfai = np.pi*fai
    ps = np.array((0,2*Dfai,4*Dfai,6*Dfai,1*Dfai))

    cp = np.array([[1,1,1,1,1],
                   [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1]),np.exp(-1j*ps[1])],
                   [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2]),np.exp(-1j*ps[2])],
                   [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3]),np.exp(-1j*ps[3])],
                   [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*ps[4]),np.exp(-1j*ps[4])]]) 
    
    cpx,cpy =cp.shape
    print('cp的秩为：',np.linalg.matrix_rank(cp))
    return np.repeat(cp.reshape(1,cpx,cpy),opt.Nangles,axis=0) #给CP增加一维度，表示方向

def sepmatrix5_5(opt): #创建cp矩阵
    ''' create matrix to separate orders from raw data '''
    
    print('tr三光束，分离矩阵5x5 相位间隔：',Fraction(2, opt.Nshifts))
    
    #正常phase
    # phase = 2/opt.Nshifts
    # Dfai = np.pi*phase

    #对phase添加扰动
    phase = np.array([2,2,2,2,2])/5
    Dfai = np.pi*phase

    ps = np.array((0,1*Dfai[1],2*Dfai[2],3*Dfai[3],4*Dfai[4]))
    cp = np.array([[1,1,1,1,1],
                   [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1]),np.exp(-1j*ps[1])],
                   [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2]),np.exp(-1j*ps[2])],
                   [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3]),np.exp(-1j*ps[3])],
                   [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*ps[4]),np.exp(-1j*ps[4])]]) 

    cpx,cpy =cp.shape
    print('cp的秩为：',np.linalg.matrix_rank(cp))
    return np.repeat(cp.reshape(1,cpx,cpy),opt.Nangles,axis=0) #给CP增加一维度，表示方向



def sepmatrix5_5_7pi(opt): #创建cp矩阵
    ''' create matrix to separate orders from raw data '''
    
    print('tr三光束，分离矩阵5x5 相位间隔：',Fraction(2, opt.Nshifts))
    
    #正常phase
    # phase = 2/opt.Nshifts
    # Dfai = np.pi*phase

    #对phase添加扰动
    phase = 2/7
    Dfai = np.pi*phase

    ps = np.array((0,2*Dfai,3*Dfai,4*Dfai,6*Dfai))
    cp = np.array([[1,1,1,1,1],
                   [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1]),np.exp(-1j*ps[1])],
                   [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2]),np.exp(-1j*ps[2])],
                   [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3]),np.exp(-1j*ps[3])],
                   [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*ps[4]),np.exp(-1j*ps[4])]]) 

    cpx,cpy =cp.shape
    print('cp的秩为：',np.linalg.matrix_rank(cp))
    return np.repeat(cp.reshape(1,cpx,cpy),opt.Nangles,axis=0) #给CP增加一维度，表示方向

def sepmatrix3_3(opt,Faixy=None): #创建cp矩阵
    ''' create matrix to separate orders from raw data 
        使用新的相位矩阵
    '''
    phase = 2/opt.Nshifts
    print('双光束，分离矩阵3x3 相位间隔：',phase,'=',Fraction(2, opt.Nshifts))
    Dfai = np.pi*phase  
    ps = np.array((0,1*Dfai,2*Dfai))

    cp = np.array([[1,1,1],
                   [1,np.exp(2j*ps[1]) ,np.exp(-2j*ps[1]) ],
                   [1,np.exp(2j*ps[2]) ,np.exp(-2j*ps[2]) ]])


    cpx,cpy =cp.shape
    # print('cp的秩为：',np.linalg.matrix_rank(cp))

    return np.repeat(cp.reshape(1,cpx,cpy),opt.Nangles,axis=0) #给CP增加一维度，表示方向

def Matual(opt,Dk,cp):
    #Dk 4d ； cp 3d
    #V2.6
    Nshifts = opt.Nshifts
    
    assert len(Dk.shape)==4
    nc,nz,nx,ny = Dk.shape   #12,1,nx,ny
    print('Matual-nc,nz,nx,ny',nc,nz,nx,ny)

    Ck = np.zeros((nc//Nshifts,Nshifts,nz,nx,ny),dtype=np.complex64) #3个方向，每个方向解出来3个变量

    Dk_1D = Dk.reshape(nc//Nshifts,Nshifts,-1) #方向，相位，sk3,4
    
    print(Ck.shape,Dk_1D.shape)

    for i in range(nc//Nshifts):   #遍历每个方向   
        Ck[i] = (cp[i] @ Dk_1D[i]).reshape(Nshifts,nz,nx,ny)  # 每个方向 为3x5 @ 5x1

    return Ck.reshape(nc,nz,nx,ny)




#分离ck
def separate(opt,Dr,run_parameter,mask_flag=0,RL=0,re_mode=0,Debug=0):
    #3D/2d/传统
    #v2.6-2023-7-27 19:39:17
    '分离矩阵'
    #多层 opt.flag_N = 7
    if opt.flag_N==7:   #多层7x7
        if opt.Nshifts == 7:    #标准，带轴向相位，21-21
            print('MP 7x7')
            cp=sepmatrix7_7(opt)

        elif opt.Nshifts == 5:  #不带轴向，可做slice重建 21-15
            print('slice 7x5')
            cp=sepmatrix7_5(opt)
    # tr3D
    elif opt.flag_N == 5: # 传统的5x5
        print('tr 5X5')
        cp = sepmatrix5_5(opt)         #正常均匀光
        # cp = sepmatrix5_5_7pi(opt)   #尝试不均匀的7pi光--有一些影响
    # 2d
    elif opt.flag_N ==3 :  #2D 3x3
        print('2D 3x3')
        cp = sepmatrix3_3(opt)
    
    '矩阵求逆'#V2版本
    cp_1 = np.zeros_like(cp)
    for i in range(opt.Nangles):
        cp_1[i] = np.linalg.pinv(cp[i])

    
    print('cp-shape',cp_1.shape)
    
    # '对Dr的xOy边缘平滑'
    # if mask_flag==1:    #参数估计用
    #     mask = _WF.mask2(opt,sigma=0.25)
    # else:               #重建用
    #     mask = 1


    #分离之前先对频谱做截断    #注意这里的频谱截断02 与1阶的不同
    _,z,x,y = Dr.shape
    OTFmask3d = _U.pad_or_cut(opt.OTFmask3D,x=x,y=y,z=z)
    Dk = np.zeros_like(Dr).astype(opt.dtype_com)
    RL_I=[]
    for i in range(opt.nc):
        I_Edg_Blur = Dr[i]
        if RL != 0 : #是否RL
            I_Edg_Blur = restoration.richardson_lucy(np.copy(I_Edg_Blur),opt.psf,RL,clip=False)
            RL_I.append(I_Edg_Blur)            
        
        temp = _U.fftn(I_Edg_Blur)


        # Dr[i] = _U.ifftn(temp) #空域分离
        Dk[i] = (temp)           #频域分离

        if i == 3 :              #保存raw data
            name = run_parameter['filename']
            _U.save_tif_v21(f'RAW-DATA-K—ori1-{name}.TIF',abs(temp),run_parameter)
            # _U.save_tif_v21(f'I_Edg_Blur-{name}.TIF',abs(I_Edg_Blur),run_parameter)
            # _U.save_tif_v21(f'I_Edg_Blur-k{name}.TIF',abs(_U.fftn(I_Edg_Blur)),run_parameter)

    #频域分离
    ck = Matual(opt,(Dk),cp_1) 

    # #空域分离
    # Cr = Matual(opt,(Dr),cp_1)   
    # print('分离出来的Cr.shape',Cr.shape)

    # #乘上 窗函数，使空域图像更加平滑
    # ck = np.zeros_like(Cr,dtype=opt.dtype_com)
    # for i in range(opt.nc):
    #     ck[i] = _U.fftn( Cr[i]*(mask**3))
    

    if Debug == 1:
        if opt.Nshifts == 3:
            list = [0,1,2,4,7]
        elif opt.Nshifts == 5:   #
            # list = [0,1,2,3,4,6,11]
            list = [0,1,2,3,4,6,7,8,9,10,11,12,13,14]
        elif opt.Nshifts == 7:
            # list= [1,2]
            # list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            list = [0,1,3, 7,8,10, 14,15,17]
        for i in list:
            name = run_parameter['filename']
            # _U.save_tif_v21(f'②Ck-real{i}-{name}.tif', np.real(ck[i]) ,run_parameter)            
            # _U.save_tif_v21(f'②Ck-imag{i}-{name}.tif', np.imag(ck[i]) ,run_parameter)
            
            if (i % 5) < 3:
                temp = ck[i]#*OTFmask3d[0]
            else:
                temp = ck[i]#*OTFmask3d[2]
            
            # _U.save_tif_v21(f'②Ck-abs{i}-{name}.tif', abs(temp) ,run_parameter) 
            # if i==0 or i==1 or i==2 or i==3 or i==4:
            #     _U.save_tif_v21(f'②Cr-real{i}-{name}.tif', np.real((ck[i])) ,run_parameter)
            #     _U.save_tif_v21(f'②Cr-img{i}-{name}.tif', np.imag((ck[i])) ,run_parameter)         
           
            # print(f'ck-{i,j}-{(ck)[i,8,126,219]}')
            # print(f'ck-{i,j}-{(ck)[i,8,256,256]}')
            if i == 0:
                
                
                temp0 = ( _U.ifftn(ck[i])).real
                # CK0 = _U.pad_or_cut(ck[i],x=2*opt.nx,y=2*opt.ny,z=opt.nz_re)
                
                # part1 = opt.OTFmask3D[0]*(opt.OTF[0]*CK0)
                # part2 = (3*(opt.OTF[0]**2)+0.00001)
                # temp0_deconv =  part1/ part2
                # temp1 = ( _U.ifftn(temp0_deconv)).real
                
                
                _U.save_tif_v21(f'②Cr_{i}.tif',temp0,run_parameter)
                # _U.save_tif_v21(f'②Cr_deconv{i}.tif',temp1,run_parameter)
                # _U.save_tif_v21(f'②Ck_deconv{i}.tif',abs(temp0_deconv),run_parameter)
                # _U.save_tif_v21(f'part1{i}.tif',abs(part1),run_parameter)
                # _U.save_tif_v21(f'part2{i}.tif',abs(part2),run_parameter)
                # asd=asd
                

            # _U.save_tif(f'②Ck{i}-XY.tif',abs( np.sum(ck[i],axis=0)) )
            # _U.save_tif(f'②Ck{i}-ZY.tif',abs( np.sum(ck[i],axis=1)) )
            # _U.save_tif(f'②Ck{i}-ZX.tif',abs( np.sum(ck[i],axis=2)) )
    return ck