from pathlib import Path
from tifffile import imread, imwrite
import numpy as np
import Utils as _U
from skimage.transform import resize
import torch
# from torch.fft import ifftshift as tifftshift
tifftshift = torch.fft.ifftshift
tfftshift = torch.fft.fftshift

def Ftabs(data):
    return  _U.tfftn(tifftshift(data))
def iFtabs(data):
    return  tfftshift(_U.tifftn(data))


# blur_img_defocus = imread(r'F:\\4TB图片-续\\2023\11月\11.6\\MP_er1_116\\Dr\\10\\Dr_stack-11.6日-10-MP_er1_116-2121-21帧-mp-1.tif')
# blur_img_defocus=blur_img_defocus-camera_offset
# blur_img_defocus = torch.from_numpy(blur_img_defocus)

def mask2_self(nx,ny,kuodaxy=1,sigma = 0.25,padsize=0):
    """边缘矩形mask"""
    # ny = nx
    # nx = ny
    # nz = nz

    x = torch.arange(1,nx+1).reshape(-1,1)
    y = torch.arange(1,ny+1).reshape(1,-1)

    sigma=sigma;
    mask2d = ((torch.sigmoid(sigma*(x))-torch.sigmoid(sigma*(x-nx-1)))
                * (torch.sigmoid(sigma*(y))-torch.sigmoid(sigma*(y-ny-1))))

    return mask2d.numpy();

def BFmain(blur_img_defocus,offset=0):
    depth_in=16
    depth_out=50
    camera_offset = offset
    print("BF处理，PSF参数为RW_lambda=525-1.33-1.27-60-97.5")
    # PSF_3D = imread(r'F:\\code-F\\PSF RW_lambda=525-1.33-1.33-60-100.tif')
    PSF_3D = imread(r'../../PSF RW_lambda=525-1.33-1.27-60-97.5.tif')
    imgz_psf,_,_= PSF_3D.shape
    
    #OTF处理
    PSF_3D = PSF_3D / np.sum( PSF_3D );  #每一层归一化
    PSF_3D = torch.from_numpy(PSF_3D)
    OTF_z=torch.zeros_like(PSF_3D)
    for i in range(imgz_psf):
        OTF_z[i,:,:]=abs(Ftabs(PSF_3D[i,:,:,]))

    # _U.save_tif(r'OTF_z.tif',OTF_z)
    z_depth_in_S = int(np.floor(imgz_psf/2)+1-depth_in)   -1
    z_depth_in_E = int(np.floor(imgz_psf/2)+1+depth_in)

    z_depth_out_S1 = int(np.floor(imgz_psf/2)+1-depth_out)-1    
    z_depth_out_E1 = int(np.floor(imgz_psf/2)-depth_in)

    z_depth_out_S2 = int(np.floor(imgz_psf/2)+depth_in+2) -1
    z_depth_out_E2 = int(np.floor(imgz_psf/2)+depth_out+1)

    OTF_sum_infocus=torch.sum(OTF_z[z_depth_in_S:z_depth_in_E,:,:],dim=0)
    OTF_sum_outfocus=torch.sum(OTF_z[z_depth_out_S1:z_depth_out_E1,:,:],dim=0)+torch.sum(OTF_z[z_depth_out_S2:z_depth_out_E2,:,:],dim=0)

    # _U.save_tif(r'OTF_sum_infocus.tif',OTF_sum_infocus)
    # _U.save_tif(r'OTF_sum_outfocus.tif',OTF_sum_outfocus)
    
    mask_OTF_in=torch.clone(OTF_sum_infocus); #根据Hin的分布构建一个mask,
    thre=torch.max(OTF_sum_infocus)
    mask_OTF_in[mask_OTF_in<thre*0.008]=0;
    mask_OTF_in[mask_OTF_in>0]=1;
    # OTF_sum_infocus_mask2=OTF_sum_infocus*mask_OTF_in; #原代码也没用到


    mask_OTF_out=torch.clone(OTF_sum_outfocus); #根据Hout的分布构建一个mask,
    thre=torch.max(OTF_sum_outfocus);
    mask_OTF_out[mask_OTF_out< (thre*0.008)]=0;
    mask_OTF_out[mask_OTF_out>0]=1;
    OTF_sum_outfocus_mask2=OTF_sum_outfocus*mask_OTF_out;


    OTF_l_mask=OTF_sum_outfocus_mask2/((OTF_sum_infocus+OTF_sum_outfocus)*mask_OTF_in+np.finfo(float).eps);           
    OTF_l_mask_256=torch.clone(OTF_l_mask[256,:]); #取出中间一行，找到小于0之外的最小值 
    # _U.save_tif(r'OTF_l_mask.tif',OTF_l_mask)
    
    OTF_l_mask_256[OTF_l_mask_256==0]=99
    min_OTF_256 = torch.min(OTF_l_mask_256)
    print(min_OTF_256)
    OTF_l_mask1=OTF_l_mask-min_OTF_256
    # _U.save_tif(r'OTF_l_mask111.tif',OTF_l_mask1)
    OTF_l_mask1[OTF_l_mask1<0]=0
    # _U.save_tif(r'OTF_l_mask11.tif',OTF_l_mask1)
        
    #开始处理背景
    blur_img_defocus = _U.image_reshape(blur_img_defocus,dim=4)
    # assert len(blur_img_defocus.shape)==4
    
    nc,nz,nx,ny = blur_img_defocus.shape
    blur_img_defocus = blur_img_defocus.reshape(-1,nx,ny)
    imgz_img,imgx_img,imgy_img=blur_img_defocus.shape
    OTF_l_mask1=torch.from_numpy(resize(OTF_l_mask1.numpy(),(imgx_img,imgy_img)))
    
    blur_img_defocus=blur_img_defocus-camera_offset  #老的减去offset
    blur_img_defocus = torch.from_numpy(blur_img_defocus)
    # _U.save_tif(r'blur_img_defocus.tif',blur_img_defocus.numpy())
    blur_img_defocus_new=torch.zeros_like(blur_img_defocus) #构建新的张量

    # _U.save_tif(r'OTF_l_mask1.tif',OTF_l_mask1)

    #加窗
    mask = mask2_self(imgx_img,imgy_img,sigma=0.25)
    # _U.save_tif(r'mask.tif',mask**9)
    
    
    for ii in range(imgz_img):
        bg=np.real(iFtabs(Ftabs(blur_img_defocus[ii]*mask**9)*abs(OTF_l_mask1)))
        blur_img_defocus_new[ii]=blur_img_defocus[ii]-bg;
        
    # _U.save_tif(r'blur_img_defocus_new.tif',blur_img_defocus_new)
    
    blur_img_defocus_new = blur_img_defocus_new.numpy()
    blur_img_defocus_new[blur_img_defocus_new<0]=0
    # _U.save_tif('blur_img_defocus_new.tif',blur_img_defocus_new.astype(np.uint16))
    blur_img_defocus_new = blur_img_defocus_new.reshape(nc,nz,nx,ny)
    return blur_img_defocus_new



# blur_img_defocus = imread(r'F:\4TB图片-续\2023\11月\11.2 MP-1.42-1.514-1.27-actin-er-mc外膜\MP_actin1_112\Dr\Dr_stack-11.2日-10-MP_actin1_112-2121-21帧-mp-0.tif')
# blur_img_defocus = torch.from_numpy(blur_img_defocus)

# BFmain(blur_img_defocus,offset=0)