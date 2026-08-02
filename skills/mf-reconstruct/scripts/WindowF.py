import numpy as np
import torch
import math

import Utils as _U
import DataProcess as _DP


def mask2(opt,kuodaxy=1,sigma = 0.25,padsize=0):
    """边缘矩形mask"""
    ny = opt.nx*kuodaxy
    nx = opt.ny*kuodaxy
    nz = opt.nz

    x = torch.arange(1,nx+1).reshape(1,-1)
    y = torch.arange(1,ny+1).reshape(-1,1)

    sigma=sigma;
    mask2d = ((torch.sigmoid(sigma*(x))-torch.sigmoid(sigma*(x-ny-1))).repeat(nx,1)
                * (torch.sigmoid(sigma*(y))-torch.sigmoid(sigma*(y-nx-1))).repeat(1,ny))

    return mask2d.numpy();

def mask2_self(nz,nx,ny,sigma = 0.25):
    """边缘矩形mask"""
    # ny = opt.nx*kuodaxy
    # nx = opt.ny*kuodaxy
    # nz = opt.nz

    x = torch.arange(1,nx+1).reshape(1,-1)
    y = torch.arange(1,ny+1).reshape(-1,1)

    sigma=sigma
    mask2d = ((torch.sigmoid(sigma*(x))-torch.sigmoid(sigma*(x-ny-1))).repeat(nx,1)
                * (torch.sigmoid(sigma*(y))-torch.sigmoid(sigma*(y-nx-1))).repeat(1,ny))

    return mask2d.numpy()


#############################切趾#########################
def apod(opt,kuodaxy=2,kuodaz=2): #3D切趾函数+截止频率
    ''' apodization to avoid ringing '''
    rxy = 2*opt.fc    #传入OTF的半径
    rz =  2*opt.fc_z
    #这是一个切趾
    opt.axy=0.8
    opt.az=0.8
    opt.expn = 1
    ZV,XV,YV = _U.meshgrid(opt,kuodaxy=kuodaxy,kuodaz=kuodaz)
    apo = ( 1 - opt.axy * np.sqrt(XV**2 + YV**2) / rxy )**opt.expn * ( 1 - opt.az * np.sqrt(ZV**2) / rz )**opt.expn
    #这mask是在rxy rz内的范围
    rhxy = np.sqrt(XV**2 + YV**2 + 0.*ZV**2)/rxy
    rhz = np.sqrt(0.*XV**2 + 0.*YV**2 + ZV**2)/rz
    msk_xy = (rhxy<=1.0).astype(np.float32)
    msk_z = (rhz<=1.0).astype(np.float32)
    _U.save_tif('msk_z.tif',msk_z)
    # _U.save_tif('apo.tif',apo)
    # _U.save_tif('msk_xy.tif',msk_xy)
    msk = msk_xy * msk_z
    apodiz = apo * msk
    return apodiz

def build_gaussian_3D(opt,mean,standard_deviation,kuodaxy=2,kuodaz=2):
    nz,nx,ny = (opt.nz+2*opt.Zpadding)*kuodaz,opt.nx*kuodaxy,opt.ny*kuodaxy
    len = 1
    x = np.linspace(-len,len, nx)
    y = np.linspace(-len, len, ny);
    z = np.linspace(-len, len, nz);

    x,z,y= np.meshgrid(x,z,y) 
    z = np.exp(-((y-mean)**2 + (x - mean)**2 +(z-mean)**2) /(2*(standard_deviation**2)))  #这里用的三维高斯公式
    z = z/(math.pow((2*np.pi)*standard_deviation,3/2))   #这里是三维高斯
    z = z/np.max(z)
    # z = z/np.sum(z)
    return z;

def bhs_3D(opt,plong,fc,kuodaxy=2,kuodaz=1,Debug=0):
    nx = opt.nx*kuodaxy
    ny = opt.ny*kuodaxy
    nz = opt.nz*kuodaz
    ZV, XV,YV = _U.meshgrid(opt,kuodaxy=kuodaxy,kuodaz=kuodaz)

    k_r = np.sqrt(XV**2+YV**2+ZV**2)
    k_max = plong+fc
    bhs = np.cos(np.pi*k_r /(2*k_max))
    indi = np.where(k_r>k_max)
    bhs[indi] = 0

    if Debug==1:
        _U.save_tif('①bhs.tif',bhs)
    return bhs

def bhs_3D_Z(opt, plong,fc,kuodaxy=2,kuodaz=1,Debug=0,lenz=2,Zjust=200):
    """
    增加一个轴向衰减
    """
    nx = opt.nx*kuodaxy
    ny = opt.ny*kuodaxy
    nz = (opt.nz+2*opt.Zpadding)*kuodaz
    NA = opt.na
    
    pixelsizeZ =  opt.dz/Zjust  #转变为um
    wavelength = opt.lamdem/1000  #转变为um
    
    zv,xv,yv = _U.meshgrid(opt,kuodaxy=kuodaxy,kuodaz=kuodaz)

    dpz = 1/(nz*pixelsizeZ)
    k_xymax = plong+fc;
    k_zmax = 2* ((NA**2/(2*wavelength))/dpz);
    k_xy = np.sqrt(xv**2 + yv**2);
    k_z = np.sqrt(zv**2);
    rhox = abs(xv)/(2*k_xymax);
    rhoy = abs(yv)/(2*k_xymax);
    rhoz = abs(zv)/(2*k_zmax);
    rho = np.sqrt(rhox*rhox + rhoy*rhoy + rhoz*rhoz);
    apodiz = np.cos(np.pi*rho);
    indi_xy = k_xy > k_xymax;
    indi_z = k_z > k_zmax;
    msk = indi_xy|indi_z;
    apodiz[msk==1] = 0;
    apodiz[apodiz<0] = 0;

    #调节轴向衰减
    # zATT = build_gaussian_z(opt,z=nz,x=nx,y=ny,lenz=lenz)
    # _U.save_tif('zATT.tif',zATT[:,nx//2,:])

    # if Debug==1:
    _U.save_tif('apodiz1.tif',apodiz)
    apodiz = apodiz#*zATT
    # _U.save_tif('apodiz2.tif',apodiz)
    return apodiz


#HIFI
#################notch_filter############################
def valAttenuation(opt,px=0,py=0,pz=0,Astr=1,fwhm=1):
    #传统的notch  
    #if notch==1:
    #   Atte = _WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],Astr=1,fwhm=1)  #W1
    #   Atte = _WF.valAttenuation(opt,PGxy[i,j,0],PGxy[i,j,1],Astr=1,fwhm=1)  #W2 
    #注意0阶与其他阶的区别
    
    nz,nx,ny = (opt.nz_re),opt.nx*2,opt.ny*2
    # nz = 16
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))
    zv, xv, yv = np.meshgrid(z, x, y, indexing='ij', sparse=True)
    
    rad1 = np.sqrt( (xv-px) **2 + (yv-py)**2)*opt.cyclesPerMicron   # 0.0248
    # rad1 = np.sqrt( (xv-(150.0309)) **2 + (yv+144.8457)**2)*0.0248
    va=1-Astr*(np.exp(-np.power(rad1,2)/(np.power(0.5*fwhm,2))))
    va1 = 1-(1-va)
    #增加外面的cutoff
    # rad2 = np.sqrt( (xv-px) **2 + (yv-py)**2)
    # va1[rad2>opt.fc] = 0
    
    return va1

        


# def Notch_fliter(opt,kuodaxy=1,kuodaz=1,notch_1=0.2,notch_2 = 3.5):
#     nx = opt.nx*kuodaxy
#     ny = opt.ny*kuodaxy
#     # nz = (opt.nz+2*opt.Zpadding)*kuodaz
#     xv,yv = _U.meshgridxy(opt,kuodaxy=kuodaxy)
#     # n.meshgrid(opt,kuodaxy=kuodaxy)
#     k_r = np.sqrt(xv**2+yv**2);
#     notch_filter = 1-np.exp(-notch_1*(abs(k_r)**notch_2));
#     # notch_filter = notch_filter.reshape(1,nx,ny)

#     return notch_filter
#     # return np.repeat(notch_filter,(nz),axis=(0))



#调用
def getotfAtt(opt,px=0,py=0,pz=0,Astr=1,fwhm=1):    
    #参数估计 #这里不需要增大nz？   
    # nz,nx,ny = opt.nz,opt.nx,opt.ny  #对ck进行低频抑制 所以没扩大
    nz,nx,ny = (opt.nz_re),opt.nx*2,opt.ny*2
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))
    zv, xv, yv = np.meshgrid(z, x, y, indexing='ij', sparse=True)
    
    rad1 = np.sqrt( (xv-px) **2 + (yv-py)**2) *opt.cyclesPerMicron   # 0.0248

    rad2 = np.sqrt( (zv-pz)**2)*opt.cyclesPerMicron
    rad2= rad2/np.max(rad2)

    va=1-Astr*(np.exp(-np.power(rad1,4)/(2*np.power(fwhm,4)))); #用于参数估计的
    #但在pz范围内需要急速恢复
    zATT = build_gaussian_z(opt,z=nz,x=nx,y=ny)
      
    _U.save_tif('zATT2.tif',zATT)
    return 1-(1-va)*(1-rad2)#*zATT 

def build_gaussian_z(opt,z,x,y,pz=0,mean=0,std=1,lenz=2):
    #仅仅在Z轴上的衰减
    # nz,nx,ny = (opt.nz+2*opt.Zpadding)*kuodaz,opt.nx*kuodaxy,opt.ny*kuodaxy
    nz,nx,ny = z,x,y 
    lenz = lenz#10   #值越大 衰减越快
    x = np.linspace(-lenz,lenz, nx)
    y = np.linspace(-lenz, lenz, ny);
    z = np.linspace(-lenz+pz, lenz+pz, nz);

    x,z,y= np.meshgrid(x,z,y) 
    z = np.exp(-((z-mean)**2) /(2*(std**2)))  #
    z = z/(math.pow((2*np.pi)*std,2))   #
    z = z/np.max(z)
    return z;



############ open-3DSIM #############
def Notch_fliter(opt,PGxy,pz,i,j,Astr=0.92,fwhm=0.4):
    nz,nx,ny = (opt.nz_re),opt.nx*2,opt.ny*2
    # nz = 16
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))
    zv, xv, yv = np.meshgrid(z, x, y, indexing='ij', sparse=True)

    #基本参数 利用1阶Pxy 计算 pz, 需要用真实的长度计算
    PGzz,PGxy1 = _DP.get_pzp1(opt,PGxy)
    qvector_xy = np.mean(PGxy1)/2  # (3,1)
    qvector_z  = np.mean(abs(PGzz[:,3:]))
    xmove=PGxy[i,j,0]
    ymove=PGxy[i,j,1]
    zmove=pz
    'notch' #XYZ陷波部分
    if opt.nz!=1:    #可以这样移动
        qradsq = ( (xv-xmove)/qvector_xy  )**2+( (yv-ymove)/qvector_xy )**2+( (zv-zmove)/qvector_z )**2;
    else:
        qradsq = ( (xv-xmove)/(0.5*qvector_xy) )**2+( (yv-ymove)/(0.5*qvector_xy) )**2+( (zv-zmove)/(0.5*qvector_z) )**2;
    # _U.save_tif('qradsq.tif',qradsq)
    notch = 1-Astr*np.exp( -qradsq/2/fwhm**2  )
    # _U.save_tif('notch.tif',notch)

    # 'Mask'  #限定外部范围 外面乘上
    # Mask = opt.OTFmask3D
    # _U.save_tif('Mask.tif',Mask)

    # 'OTF'   #外面加一层OTF衰减
    # OTFatt = abs(opt.OTF)**(0)

    'result'
    notch0 = notch#*OTFatt
    notch0 = notch0/np.max(notch0)
    # _U.save_tif('notch0.tif',notch0)
    return notch0 # 单一的notch，需要移动自己移动



######################4B-SIM notch################
# %% Creates an 3D attenuation matrix.
import numpy as np
import Utils as _U
def order0damping(radius, zindex, rlimit, zlimit):
    # % radius = rdist1
    # % zindex = Z0
    # % rlimit = rdistcutoff
    # % zlimit = zdistcutoff(1)
    rfraction = radius / rlimit
    if (zlimit > 0):
        # % 3D case
        zfraction = abs(zindex / zlimit)
    else:
        # % 2D case
        zfraction = 0
    out = rfraction**2 + zfraction**3
    out[out > 1] = 1
    return out

def suppress_singularities(x, dampenFactor):
    x6 = x**6
    out = 1.0 / (1 + (20 * dampenFactor) / (x6 + 20))
    '''
    %{
        if x1 = 0; y1 = 0; rdist1 = 0; out = 1 / 1001 = 9.9900e-04;
        if x1 = 6; y1 = 8; rdist1 = 10; out = 0.9804;
        if x1 = 100; y1 = 100; rdist1 = 141.42; out = 1;
    %}
    '''
    return out
        

def getAttenuationMatrix3D(order, nx, ny, kx, ky, nz, rdistcutoff):
    
    Z,X,Y = _U.meshgridxy_self(nx,ny,nz,index='ij')
    rdist = np.sqrt( (X-kx)**2 + (Y-ky)**2);
    
    if order != 0:
        dampenFactor=1000
        attval = suppress_singularities(rdist,dampenFactor);
        attval[:,rdist[0] > 10] = 1;  #默认为10就行
        # _U.save_tif(fr'attval-{order}.tif',(attval))
    # elseif (~params.bDampenOrder0 && params.bSuppress_singularities && order == 0)
    #     attval = obj.suppress_singularities(rdist, params.dampenFactor);
    #     attval(rdist > params.suppression_radius) = 1;
    elif order == 0: #0阶
        attval = order0damping(rdist, Z, rdistcutoff, nz//2);
    
    # _U.save_tif(r'attval.tif',(attval))
    # % if no kz=0 plane is used:
    # if (order == 0)
    #     attval[Z == 0] = 0;
    
    # print(rdistcutoff)
    mask = rdist > rdistcutoff
    attval[:,mask[0]] = 0
    return attval

# attval = getAttenuationMatrix3D(0,512,512,0,0,opt.fc_z,156) #如何计算nz 长度，OTF的轴向长度
# _U.save_tif(r'attval2.tif',(attval))
