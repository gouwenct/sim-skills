import Utils as _U
from tifffile import imread,imwrite
import numpy as np
from pathlib import Path
import torch


def Interpolate(image,nx,ny,nz,strt = 'trilinear'):
    """
    image为4D,3D都可以，但必须是numpy
     mode (str): algorithm used for upsampling:
        ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
    """
    
    before_dim = len(image.shape)

    image = _U.image_reshape(image,dim=5)   #1,cc,nz,nx,ny)

    #转变为torch
    image = _U.ToTensor(image) 

    if torch.is_complex(image)!=True : #非复数
        image_inter = torch.nn.functional.interpolate((image),size=[nz,nx,ny],mode=strt,align_corners=False)
    else: #复数拆开插值，再组合
        image_real = image.real
        image_imag = image.imag
        image_real_inter = torch.nn.functional.interpolate((image_real),size=[nz,nx,ny],mode=strt,align_corners=False)
        image_imag_inter = torch.nn.functional.interpolate((image_imag),size=[nz,nx,ny],mode=strt,align_corners=False)
        image_inter = image_real_inter+1j*image_imag_inter
        
    return _U.image_reshape(_U.ToNumpy(image_inter),dim=before_dim)


def main(data,dx=30,dx2=30,dz=97.5,dz2=60):

    if len(data.shape)==3:
        #单个3D 插值
        nz,nx,ny = data.shape
        nx2 = dx/dx2 * nx 
        nz2 = dz/dz2 * nz
        print(f"nx2={nx2},nz2={nz2}")
        nx2 = int(np.round(nx2,0))
        nz2 = int(np.round(nz2,0))
        data2 = Interpolate(data,nx=int(nx2),ny=int(nx2),nz=int(nz2))

    elif len(data.shape)==4:
        print(r'4D')
        nc,nz,nx,ny = data.shape
        
        nx2 = dx/dx2 * nx 
        nz2 = dz/dz2 * nz
        print(f"nx2={nx2},nz2={nz2}")
        nx2 = int(np.round(nx2,0))
        nz2 = int(np.round(nz2,0))
        data2 = np.zeros((nc,nz2,nx2,nx2),dtype=np.float32)
        for ii in range(nc):
            data2[ii] = Interpolate(data[ii],nx=int(nx2),ny=int(nx2),nz=int(nz2))
    #转变16int
    data2 = data2/np.max(data2) * 65535

    return data2

# path3D = r'E:\code\Ev4.0\MP-SIM\MP-recon\dataOut\20240521_only\07、MP_OTm_PKMito_22-V4.9\561-20240521_07、MP_OTm_PKMito_22_W=1e-03NF0(0.95)_V4.9t.tif'
# data = imread(path3D)
# data = data/1.0
# data2 = main(data)

# #保存dataout中
# path_name = Path(path3D).stem
# if len(data2.shape)==4:
#     _U.save_tif(f"just_{data2.shape}-{path_name}.tif",data2,dtypee = np.uint16,axes='TZYX')
# else:
#     _U.save_tif(f"just_{data2.shape}-{path_name}.tif",data2,dtypee = np.uint16,axes='CYX')
# print('success save')