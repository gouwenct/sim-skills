import numpy as np
from numpy.fft import fftshift,ifftshift
from fractions import Fraction
from skimage import restoration

import WindowF as _WF
import Utils as _U
#分离ck Matrix

def SepMatrix7(faiXY,faiZ,order,sequence):
    """
    example cp = SepMatrix7(2/5,0.52,[0,2,4,6,1,3,5],43)

    Args:
        faiXY (_type_): Phase interval
        faiZ (_type_): Axial phase interval
        order (_type_): Phase order

    Returns:
        _type_: cp (3,7,7)
    """
 
    faiPi = faiXY*np.pi
    faiPiZ = faiZ*np.pi
    ps = faiPi*np.array(order)
    psz = np.array((faiPiZ,-faiPiZ))
    
    if sequence==43 or sequence==5343:
        cp = Matrix43(ps,psz)
    elif sequence==52 or sequence==5352:
        cp = Matrix52(ps,psz)
    else:
        raise('sequence error')
    
    print(f'faiPi={faiPi},faiPiZ={faiPiZ},order={order}')
    print('cp7的秩为：',np.linalg.matrix_rank(cp))
    print('cp7的条件数为：',np.linalg.cond(cp))
    
    return np.repeat(cp[np.newaxis,:,:],3,axis=0)

def Matrix43(ps,psz):
    cp = np.array([[1,1,1,1,1,1,1],
            [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1])          ,np.exp(1j*ps[1])          ,np.exp(-1j*ps[1])          ,  np.exp(-1j*ps[1])],
            [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2])          ,np.exp(1j*ps[2])          ,np.exp(-1j*ps[2])          ,  np.exp(-1j*ps[2])],
            [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3])          ,np.exp(1j*ps[3])          ,np.exp(-1j*ps[3])          ,  np.exp(-1j*ps[3])],
            [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*(ps[4]+psz[0])),np.exp(1j*(ps[4]+psz[1])),np.exp(1j*(-ps[4]+psz[0])),np.exp(1j*(-ps[4]+psz[1]))],
            [1,np.exp(2j*ps[5]),np.exp(-2j*ps[5]),np.exp(1j*(ps[5]+psz[0])),np.exp(1j*(ps[5]+psz[1])),np.exp(1j*(-ps[5]+psz[0])),np.exp(1j*(-ps[5]+psz[1]))],
            [1,np.exp(2j*ps[6]),np.exp(-2j*ps[6]),np.exp(1j*(ps[6]+psz[0])),np.exp(1j*(ps[6]+psz[1])),np.exp(1j*(-ps[6]+psz[0])),np.exp(1j*(-ps[6]+psz[1]))]]) 
    return cp

def Matrix52(ps,psz):
    cp = np.array([[1,1,1,1,1,1,1],
            [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1])          ,np.exp(1j*ps[1])          ,np.exp(-1j*ps[1])          ,  np.exp(-1j*ps[1])],
            [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2])          ,np.exp(1j*ps[2])          ,np.exp(-1j*ps[2])          ,  np.exp(-1j*ps[2])],
            [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3])          ,np.exp(1j*ps[3])          ,np.exp(-1j*ps[3])          ,  np.exp(-1j*ps[3])],
            [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*ps[4])          ,np.exp(1j*ps[4])          ,np.exp(-1j*ps[4])          ,  np.exp(-1j*ps[4])],
            [1,np.exp(2j*ps[5]),np.exp(-2j*ps[5]),np.exp(1j*(ps[5]+psz[0])),np.exp(1j*(ps[5]+psz[1])),np.exp(1j*(-ps[5]+psz[0])),np.exp(1j*(-ps[5]+psz[1]))],
            [1,np.exp(2j*ps[6]),np.exp(-2j*ps[6]),np.exp(1j*(ps[6]+psz[0])),np.exp(1j*(ps[6]+psz[1])),np.exp(1j*(-ps[6]+psz[0])),np.exp(1j*(-ps[6]+psz[1]))]]) 
    return cp

def SepMatrix5(faiXY,order):
    """
    example cp = SepMatrix5(2/5,[01234])
    Args:
        faiXY (_type_): Phase interval
        order (_type_): Phase order

    Returns:
        _type_: cp (3,5,5)
    """
 
    faiPi = faiXY*np.pi
    # faiPiZ = faiZ*np.pi
    ps = faiPi*np.array(order)
    # psz = np.array((faiPiZ,-faiPiZ))
    
    cp = np.array([[1,1,1,1,1],
                   [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1]),np.exp(-1j*ps[1])],
                   [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2]),np.exp(-1j*ps[2])],
                   [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3]),np.exp(-1j*ps[3])],
                   [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*ps[4]),np.exp(-1j*ps[4])]]) 

    print(f'faiPi={faiPi},order={order}')
    print('cp5的秩为：',np.linalg.matrix_rank(cp))
    print('cp5的条件数为：',np.linalg.cond(cp))
    
    return np.repeat(cp[np.newaxis,:,:],3,axis=0)


def sepmatrix7_5(opt,Faixy=None): #创建cp矩阵
    ''' 
    多层挑选5个相位，按照2pi7进行分离 without 轴向相位
    '''
    # if isinstance(Faixy,np.ndarray):
    print('MP使用全部相位：02461 相位间隔：',Fraction(2, 7))
    fai  = 2/7

    Dfai = np.pi*fai
    ps = np.array((0,2*Dfai,4*Dfai,6*Dfai,1*Dfai))
    phase_3 = 0.0*np.pi
    print(f'phase_3:{phase_3*180/np.pi}')
    psz = np.array((phase_3,-phase_3))
    cp = np.array([[1,1,1,1,1],
                   [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1]),np.exp(-1j*ps[1])],
                   [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2]),np.exp(-1j*ps[2])],
                   [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3]),np.exp(-1j*ps[3])],
                   [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*ps[4]),np.exp(-1j*ps[4])]] )
    
    cpx,cpy =cp.shape
    print('cp的秩为：',np.linalg.matrix_rank(cp))
    return np.repeat(cp.reshape(1,cpx,cpy),opt.Nangles,axis=0) #给CP增加一维度，表示方向

def sepmatrix5_5(opt,Nshifts=5): #创建cp矩阵
    ''' create matrix to separate orders from raw data '''
    
    print('tr三光束，分离矩阵5x5 01234 相位间隔：',Fraction(2, Nshifts))
    phase = 2/Nshifts
    Dfai = np.pi*phase

    ps = np.array((0,1*Dfai,2*Dfai,3*Dfai,4*Dfai))
    cp = np.array([[1,1,1,1,1],
                   [1,np.exp(2j*ps[1]),np.exp(-2j*ps[1]),np.exp(1j*ps[1]),np.exp(-1j*ps[1])],
                   [1,np.exp(2j*ps[2]),np.exp(-2j*ps[2]),np.exp(1j*ps[2]),np.exp(-1j*ps[2])],
                   [1,np.exp(2j*ps[3]),np.exp(-2j*ps[3]),np.exp(1j*ps[3]),np.exp(-1j*ps[3])],
                   [1,np.exp(2j*ps[4]),np.exp(-2j*ps[4]),np.exp(1j*ps[4]),np.exp(-1j*ps[4])]]) 

    print('cp的秩为：',np.linalg.matrix_rank(cp))
    print('cp的条件数为：',np.linalg.cond(cp))
    return np.repeat(cp[np.newaxis,:,:],opt.Nangles,axis=0)

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

    print('cp的秩为：',np.linalg.matrix_rank(cp))
    print('cp的条件数为：',np.linalg.cond(cp))

    return np.repeat(cp[np.newaxis,:,:],opt.Nangles,axis=0)




def Matual(opt,Dr,cp):
    #Dr 4d ； cp 3d
    nc,nz,nx,ny = Dr.shape   #12,1,nx,ny
    print('Matual-nc,nz,nx,ny',nc,nz,nx,ny)

    Cr = np.zeros((opt.Nangles,opt.Nshifts,nz,nx,ny),dtype=np.complex64) #3个方向，每个方向解出来3个变量

    Dr_1D = Dr.reshape(opt.Nangles,opt.Nshifts,-1) #方向，相位，sk3,4
    print(Cr.shape,Dr_1D.shape)

    for i in range(opt.Nangles):   #遍历每个方向   
        temp = (cp[i] @ Dr_1D[i])  # 每个方向 为3x5 @ 5x1
        Cr[i] = (temp).reshape(opt.Nshifts,nz,nx,ny)  #

    return Cr.reshape(nc,nz,nx,ny)

def Matual_self(opt,Dr,cp,Nangles=3,Nshifts=5):
    
    #Dr 4d ； cp 3d
    
    nc,nz,nx,ny = Dr.shape   #12,nz,nx,ny
    print('Matual-nc,nz,nx,ny',nc,nz,nx,ny)

    Cr = np.zeros((Nangles,Nshifts,nz,nx,ny),dtype=np.complex64) #3个方向，每个方向解出来3个变量

    Dr_1D = Dr.reshape(Nangles,Nshifts,-1) #方向，相位，sk3,4
    print(Cr.shape,Dr_1D.shape)

    for i in range(Nangles):   #遍历每个方向   
        temp = (cp[i] @ Dr_1D[i])  # 每个方向 为3x5 @ 5x1
        Cr[i] = (temp).reshape(Nshifts,nz,nx,ny)  #

    return Cr.reshape(nc,nz,nx,ny)


#分离ck
def separate(opt,Dr,run_parameter,Nshifts=7,alpha=0,Debug=1):
    #一步到位 分离出7阶/5阶/3阶
    "分离矩阵"
    # cp = select_cp(opt,Nshifts=Nshifts)
    if opt.flag_N=='MP' and  Nshifts==7: 
        cp = SepMatrix7(opt.faiXY,opt.faiZ,opt.order7,opt.sequence)
    elif opt.flag_N=='2D':
        cp = sepmatrix3_3(opt)
        
    "求逆"
    # 各自求逆----V2版本
    cp_1 = np.zeros_like(cp)
    for i in range(opt.Nangles):
        cp_1[i] = np.linalg.pinv(cp[i])
    
    #添加Tikhonov 正则项---V3.6版本
    # cp_1 = np.zeros_like(cp)
    # alpha = alpha   #外界参数
    # for i in range(opt.Nangles):
    #     cp_1[i] = np.linalg.pinv( cp[i].T @ cp[i] + alpha*np.eye(cp[i].shape[1]) ) @ cp[i].T  #(A.T @ A + λI)^-1 @ A.T
        # print('cp-Tikhonov的秩为：',np.linalg.matrix_rank(cp_1[i]))

    # print('cp-shape',cp_1.shape)
    
    "FFT"
    nc = opt.Nangles * opt.Norders  #一步到位，这里应该为21/15/9  与Dr 方的nc 应该一致
    assert nc == Dr.shape[0]
    
    _,z,x,y = Dr.shape
    Dk = np.zeros_like(Dr).astype(opt.dtype_com)
    for i in range(nc):
        Dk[i] = _U.fftn(Dr[i])
        
    '保存rawdata'
    _U.save_tif_v21(f'RAW-DATA-K—ori123-.TIF',abs(Dk[[0,Nshifts,2*Nshifts]]),run_parameter)

    "频域分离"
    # ck = Matual(opt,(Dk),cp_1)   # 一步到位分离 分离出7阶
    ck = Matual_self(opt,(Dk),cp_1,Nangles=3,Nshifts=Nshifts)
    "保存频谱"
    # if Debug == 1:
    #     if opt.Nshifts == 3:
    #         list = [0,1,2,4,7]
    #     elif opt.Nshifts == 5:
    #         list = [0,1,2,3,4,6,11]
    #     elif opt.Nshifts == 7:
    #         # list= [0,1,3, 7,8,10, 14,15,17]
    #         # list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #         # list = [0,1,2,3,4,5,6, 7,8,10, 14,15,17]
    #         list = [0,1,2,3,4,5,6]
    #         # list= [0,1,3]
    #     for i in list:
    #         name = run_parameter['filename']
    #         _U.save_tif_v21(f'②Ck{i}-{name}.tif',( abs(ck[i])),run_parameter,sub_filename = 'ck-alpha='+str(alpha))
            # _U.save_tif_v21(f'②Ck{i}-{name}-{(np.round(error_phase,3))}-{(np.round(error_phase2,3))}-{(np.round(error_phase3,3))}-{np.round(error_phase_z,3)}.tif',( abs(ck[i]) ),run_parameter,sub_filename = 'ck')
            # if i%7== 1:
            #     _U.save_tif_v21(f'②Ck{i}-{name}-{(np.round(error_phase,3))}-{(np.round(error_phase2,3))}-{(np.round(error_phase3,3))}-{np.round(error_phase_z,3)}.tif',( abs(ck[i]) ),run_parameter,sub_filename = 'ck-1')

    return ck



#第一步分离：通常5x5,先分离2阶
def separate_step_one(opt,Dr,run_parameter,Debug=0,Nshifts=5):
    'get 0 -2,+2,-1,+1'
    assert Dr.shape[0]==15  #传入时 只要5个相位差
    "SepMatri"
    cp = SepMatrix5(opt.faiXY,opt.order5)  #2/5  [01234]
    
    "pinv"
    #各自求逆----V2版本
    cp_1 = np.zeros_like(cp)
    for i in range(opt.Nangles):
        cp_1[i] = np.linalg.pinv(cp[i])
        
    "FFT"
    nc = opt.Nangles * 5  #，这里nc应该为15
    assert nc == Dr.shape[0]
    
    Dk = np.zeros_like(Dr).astype(opt.dtype_com)
    for i in range(nc):
        Dk[i] = _U.fftn(Dr[i])

    'save rawdata'
    if Debug==1:
        _U.save_tif_v21(f'RAW-DATA-K—ori123-.TIF',abs(Dk[[0]]),run_parameter)
    
    "bands sep"
    ck = Matual_self(opt,(Dk),cp_1,Nangles=3,Nshifts=Nshifts)   

    return ck

#第二步 2x2分离
def GETck(opt,ck1,ck2,run_parameter):
    'combine  (-1,-1) (-1,+1) (+1,+1),(+1,-1)'
    _,nz,nx,ny = ck1.shape
    ck = np.zeros((21,nz,nx,ny),dtype=np.complex64)
    for i in range(3):
        for j in range(5):
            if j < 3: #2阶
                ck[i*7+j]=ck1[i*5+j]    #取了第一个的2阶
            elif j==3 : # 1jie 3 4
                ck[i*7+3],ck[i*7+4] = comb1jie(opt,ck1[i*5+j],ck2[i*5+j],opt.faiZ*np.pi)
            elif j==4 : # 1jie 5 6
                ck[i*7+5],ck[i*7+6] = comb1jie(opt,ck1[i*5+j],ck2[i*5+j],opt.faiZ*np.pi)    
    return ck
                

def comb1jie(opt,ckk1,ckk2,fai7XY=0.52*np.pi):
    #传进来的是一个方向的
    # psz = (-1)**(j+1) * np.array([0,  0.52*np.pi])   #j=3为正， j=4为负 错误
    psz = np.array([0,  fai7XY])
    cp = np.array([[np.exp(1j*psz[0]),np.exp(-1j*psz[0])],
                   [np.exp(1j*psz[1]),np.exp(-1j*psz[1])]]) 
    cp_1 = np.linalg.pinv(cp)
    
    # print('2x2條件是',nc,nz,nx,ny)
    # print('cp2的秩为：',np.linalg.matrix_rank(cp))
    # print('cp2的条件数为：',np.linalg.cond(cp))
    #combine ckk1,ckk2
    nz,nx,ny = ckk1.shape
    stack_ = np.zeros((2,nz,nx,ny),dtype=opt.dtype_com)
    stack_[0] = ckk1
    stack_[1] = ckk2
    
    
    #get ck3 ck4/ ck5 ck6
    nc,nz,nx,ny = stack_.shape   #2,8,nx,ny
    # print('Matual-nc,nz,nx,ny',nc,nz,nx,ny)
    
    Dr_1D = stack_.reshape(2,-1) #2个相位，其他3d拉成一条线
    # print(Ck_1jie.shape,Dr_1D.shape)
    

    temp = (cp_1 @ Dr_1D)  #  2x2 @ 2x-1
    Ck_1jie = (temp).reshape(2,nz,nx,ny)  #

    return Ck_1jie[0],Ck_1jie[1]


# #5252
#  def GETck34(opt,Dk_6_1jie,Dk_7_1jie,fai7XY,run_parameter):
#      #传进来的是一个方向的
#     psz = np.array([0,  fai7XY])
#     cp = np.array([[np.exp(1j*psz[0]),np.exp(-1j*psz[0])],
#                    [np.exp(1j*psz[1]),np.exp(-1j*psz[1])]]) 
#     cp_1 = np.linalg.pinv(cp)
    
    
    