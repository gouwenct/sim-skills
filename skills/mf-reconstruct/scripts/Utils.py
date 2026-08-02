import numpy as np
from numpy import pi, cos, sin,exp
import torch 
from pathlib import Path
# from loguru import logger
# import pickle
import math
import cv2

from tifffile import imread, imwrite,TiffWriter
"""
define base Funcations
example FFT/
image 全部为4D形式  nc，nz，nx，ny  如果为2d nz=1

"""

# utils utils utils utils utils utils
def fftn(data, axes=None):
    """utility method that includes shifting"""
    return np.fft.fftshift(np.fft.fftn(data))

def ifftn(data, axes=None):
    """utility method that includes shifting"""
    return np.fft.ifftn(np.fft.ifftshift(data))

# def fft2(data, axes=None):
#     """utility method that includes shifting"""
#     return np.fft.fftshift(np.fft.fft2(data))
# def ifft2(data, axes=None):
#     """utility method that includes shifting"""
#     return np.fft.ifft2(np.fft.ifftshift(data))

def tfftn(data, axes=None):
    """utility method that includes shifting"""
    return torch.fft.fftshift(torch.fft.fftn(data))
def tifftn(data, axes=None):
    """utility method that includes shifting"""
    return torch.fft.ifftn(torch.fft.ifftshift(data))



def fftshift(data):
    return np.fft.fftshift(data)
def visual(ik): #输入空域中图片，行进visual
    return np.log(abs(ik)+1)

#改变dim
def image_reshape(image,dim=4):
    """
    对转变Dr为4D，nc，nz，nx，ny
    dim parameter 设定为想要的维度
    """
    cur_dim = len(image.shape)
    
    if cur_dim==dim:
        return image           #是想要的维度      
    
    while (cur_dim<dim):       #少了就增加
        image=np.expand_dims(image, axis=0)
        cur_dim = len(image.shape)

    while (cur_dim>dim):      #多了就减少
        shape_array = image.shape
        shape_array2 = []
        shape_array2.append(shape_array[0]*shape_array[1])
        shape_array2[1:] = shape_array[2:]
        image = image.reshape(shape_array2)
        cur_dim = len(image.shape)
    return image


#pad or cut： xy，z
def pad_or_cut(image,x=512,y=512,z=1,mode = 'constant',biasx=0,biasy=0):
    """
    可裁剪，可补零，需要传入最终图像大小
    可操作3D图像， 可操作2D图像（检测2D变为3D）
    mode = 'constant' (default)
        Pads with a constant value.
    'edge'
        Pads with the edge values of array.
     ....
    """

    #记录type
    flag_dtype=0
    if isinstance(image,torch.Tensor) :
        flag_dtype = 1
    #记录之前维度
    dim_before = len(image.shape)



    nz,nx,ny = z,x,y
    image = image_reshape(image,dim = 4) #设置一下大小
    raw_nz,raw_nx,raw_ny = image.shape[1:]  
    
    #pad
    nz_left,nz_right =  pad_or_cut_cal(nz,raw_nz)
    nx_left,nx_right =  pad_or_cut_cal(nx,raw_nx)
    ny_left,ny_right =  pad_or_cut_cal(ny,raw_ny)
    
    image = image_reshape(image,dim = 4) #设置一下大小
    pad = ( (0,0),
            (nz_left,nz_right),
            (nx_left,nx_right), 
            (ny_left,ny_right) )
    # print(pad)
    image_adjust = np.pad(image,pad,mode=mode) 
    

    #cut
    if nx<raw_nx or ny<raw_ny or nz<raw_nz:
        # print('image cut')
        image_adjust = crop(image_adjust,nx=nx,ny=ny,nz=nz,biasx=biasx,biasy=biasy)

    if flag_dtype == 1 :  #恢复为torch
        image_adjust = torch.from_numpy(image_adjust)


    image_adjust = image_reshape(image_adjust,dim = dim_before) 
    return image_adjust
    
def pad_or_cut_cal(target_length,raw_length):
    left=0
    right=0
    if target_length > raw_length:
        Diff_nx = (target_length-raw_length)/2
        left,right = Diff_nx,Diff_nx # 
        if((target_length-raw_length)%2==1 and right !=0 ):    #奇数那么就得在左边多加1,对应裁剪就是左边裁剪
            left +=1 
    return int(left),int(right)

def crop(image,scale=0.5,nx=0,ny=0,nz=0,biasx=0,biasy=0):
    ##裁剪到dx，dy 大小，不设置则默认裁剪一半
    #仅仅xy方向，Z轴暂不考虑
    #如果为奇数的话，优先裁剪左边那个
    image = image_reshape(image,dim = 4) #设置一下大小
    _,raw_nz,raw_nx,raw_ny = image.shape
    if scale==0.5 and nx ==0 and ny==0: #矩阵大小裁剪一半
        nx = raw_nx//2
        ny = raw_ny//2
        
    nx_stare = int(np.ceil((raw_nx-nx)/2))
    nx_end = nx_stare+nx
    ny_stare = int(np.ceil((raw_ny-ny)/2))
    ny_end = ny_stare+ny
    # print(nx_stare,ny_stare)
    nz_stare = int(np.ceil((raw_nz-nz)/2))
    nz_end = nz_stare+nz 
    image_small = image[:,nz_stare:nz_end,nx_stare+biasx:nx_end+biasx ,ny_stare+biasy:ny_end+biasy ]
    
    return image_small


#转变为torch
def ToTensor(h):
    try:
        h = torch.from_numpy(h)
    except:
        pass
    return h
def ToNumpy(h):
    try:
        h = h.cpu()
    except:
        pass

    try:
        h = h.numpy()
    except:
        pass

    return h

#坐标轴构建
def meshgrid(opt,kuodaxy=1,kuodaz=1,index='ij'):
    #这里计算xyz 三维的向量
    #分奇偶 构建坐标系
    nx = opt.nx*kuodaxy
    ny = opt.ny*kuodaxy
    nz = (opt.nz+2*opt.Zpadding)*kuodaz 

    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))

    zv, xv, yv = np.meshgrid(z, x, y, indexing=index, sparse=True)
    return zv,xv,yv

def meshgridxy(opt,kuodaxy=1,kuodaz=1,index='ij'):
    #这里计算xyz 三维的向量
    #分奇偶 构建坐标系
    nx = opt.nx*kuodaxy
    ny = opt.ny*kuodaxy

    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    xv, yv = np.meshgrid(x, y, indexing=index, sparse=True)
    return xv,yv
def meshgridxy_self(nx,ny,nz=1,index='ij',pianyi = 0):
    #这里计算xyz 三维的向量
    #pianyi = 0 Z在中间 默认
    #

    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2)+pianyi,np.ceil(nz/2)+pianyi)  #移动Z轴坐标
    # z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))
    zv, xv, yv = np.meshgrid(z, x, y, indexing=index, sparse=True)
    print(np.ceil(-nz/2)+pianyi,np.ceil(nz/2)+pianyi)
    return zv,xv,yv

#空域移动
def shiftmat(opt,kx=0,ky=0,kz=0):  #单边移动
    '''
    使用前需要搭配meshgrid 设置网格边长
    Z方向上为单边移动
    '''
    movemat = np.exp(2j*np.pi*(kx*opt.xv+ky*opt.yv)) * np.exp(2j*np.pi*kz*opt.zv)
    return movemat

# torch的插值函数
def Interpolate(image,nx,ny,nz,strt = 'trilinear'):
    """
    image为4D,3D都可以，但必须是numpy
     mode (str): algorithm used for upsampling:
        ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
    """

    # if len(image.shape) == 4:
    #     cc,nz,nx,ny = image.shape
    # elif len(image.shape) == 3:
    #     nz,nx,ny = image.shape
    #     cc=1
    # else:
    #     print('为什么这个是个5D对象？')  
    
    before_dim = len(image.shape)

    image = image_reshape(image,dim=5)   #1,cc,nz,nx,ny)

    image = torch.from_numpy(image)

    image = torch.nn.functional.interpolate((image),size=[nz,nx,ny],mode=strt,align_corners=True)
    
    return image_reshape(image.numpy(),dim=before_dim)

#保存文件为tif
def save_tif(filename,data,sub_filename='',subsub_filename='',dtypee = np.float32,axes = 'ZYX'):
    """
    input：data名，data
    process: if new_dir!= None  make folder
    无需传入地址
    """
    #code 文件上一个目录构建一个 dataOut 并同名的文件夹
    if axes=='ZYX':
        data = image_reshape(data,dim=3)
    path = Path.cwd()/'dataOut'
    if not path.exists():
        path.mkdir()
        # print(f' maker new folder :{path}')
    
    path_data_dir = path/Path.cwd().name  #
    if not path_data_dir.exists():
        path_data_dir.mkdir()
        # print(f' maker new folder :{path_data_dir}')

    '次级文件夹'
    if sub_filename!='':
        path_data_dir = path_data_dir/sub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()
            # logger.info(f' maker new folder :{path_data_dir}')
    'ci次级文件夹'
    if subsub_filename!='':
        path_data_dir = path_data_dir/subsub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()

    '文件保存'
    #将文件保存到这里
    savefile = path_data_dir/filename
    try:
        data = data.numpy()
    except:
        pass

    #将32的保存为16
    imwrite(
        savefile,data.astype(dtypee),#.transpose(),
        imagej=True,
        photometric='minisblack',
        metadata={'axes': axes}
        )
    print(f"data successly save in ({savefile})")
    return


def save_tif_v21(filename,data,run_parameter,sub_filename='',subsub_filename=''):
    """
    input：需要保存的数据名字，data
    process: if new_dir!= None  make folder
    无需传入地址

    V2.1 ： 自动在dataout目录下新建一个文件夹，命名方式为：/X月-xx-遍历文件名/
    """
    # LOG(run_parameter)
    data = image_reshape(data,dim =3)
    'code 文件上一个目录,构建dataOut文件夹'
    path = Path.cwd()/'dataOut'
    if not path.exists():
        path.mkdir()
        # logger.info(f' maker new folder :{path}')
    
    path_data = path/run_parameter['data_name']
    if not path_data.exists():
        path_data.mkdir()
        # logger.info(f' maker new folder :{path_data}')


    path_data_dir = path_data/(run_parameter['dir_name']+"-"+run_parameter['ver'])
    if not path_data_dir.exists():
        path_data_dir.mkdir()
        # logger.info(f' maker new folder :{path_data_dir}')

    '次级文件夹'
    if sub_filename!='':
        path_data_dir = path_data_dir/sub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()
            # logger.info(f' maker new folder :{path_data_dir}')
    '次次级文件夹'
    if subsub_filename!='':
        path_data_dir = path_data_dir/subsub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()

    '文件保存'
    savefile = path_data_dir/filename
    try:
        data = data.numpy()
    except:
        pass

    #将32的保存为16
    imwrite(
        savefile,data.astype(np.float32),#.transpose(),
        imagej=True,
        photometric='minisblack',
        metadata={'axes': 'ZYX'}
        )
    print(f"data successly save in ({savefile})")
    # logger.info(f'data successly save in ({savefile})')
    # logger.remove(handler_id=None)  
    return

def save_tif_v22(filename,data,run_parameter,sub_filename='',subsub_filename='',datatype=np.float32):
    """
    input：需要保存的数据名字，data
    process: if new_dir!= None  make folder
    无需传入地址

    V2.1 ： 自动在dataout目录下新建一个文件夹，命名方式为：/X月-xx-遍历文件名/
    """
    # LOG(run_parameter)
    data = image_reshape(data,dim =3)
    'code 文件上一个目录,构建dataOut文件夹'
    path = Path.cwd()/'dataOut'
    if not path.exists():
        path.mkdir()
        # logger.info(f' maker new folder :{path}')
    
    path_data = path/(run_parameter['data_name']+'_only')
    if not path_data.exists():
        path_data.mkdir()
        # logger.info(f' maker new folder :{path_data}')


    path_data_dir = path_data/(run_parameter['dir_name']+"-"+run_parameter['ver'])
    if not path_data_dir.exists():
        path_data_dir.mkdir()
        # logger.info(f' maker new folder :{path_data_dir}')

    '次级文件夹'
    if sub_filename!='':
        path_data_dir = path_data_dir/sub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()
            # logger.info(f' maker new folder :{path_data_dir}')
    
    if subsub_filename!='':
        path_data_dir = path_data_dir/subsub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()

    '文件保存'
    savefile = path_data_dir/filename
    try:
        data = data.numpy()
    except:
        pass

    #将32的保存为16
    # imwrite(
    #     savefile,data.astype(np.float32),#.transpose(),
    #     imagej=True,
    #     photometric='minisblack',
    #     metadata={'axes': 'ZYX'}
    #     )
    
    min = np.min(data)
    max = np.max(data)
    with TiffWriter(savefile,imagej=True) as tif:
        tif.write(data.astype(datatype), metadata={'axes': 'ZYX','min': min,'max': max})
        
    print(f"data successly save in ({savefile})")
        

    return

def save_txt(filename,data):
    """
    input：data名，data{dict}
    # """
    # path = Path.cwd()/'dataOut'
    # if not path.exists():
    #     path.mkdir()
    #     print(f' maker new folder :{path}')
    


    # savefolder = path/Path.cwd().name  #
    # if not savefolder.exists():
    #     savefolder.mkdir()
    #     print(f' maker new folder :{savefolder}')

    savefile = filename
    
    #将文件保存到这里
    with open(savefile, 'w') as f:
        for key, value in data.items():
            f.write(key)
            f.write(': ')
            f.write(str(value))
            f.write('\n')

    print(f"data successly save in ({savefile})")
    return

# def save_tif_path(pathname,data):
#     """
#     input：path+file name，data
#     需指定地址
#     """
#     path = Path(str(pathname))   #得到运行这个文件的文件夹
    
#     #code 文件上一个目录构建一个指定文件夹
#     path1 = path.parent
#     if not path1.exists():
#         path1.mkdir()
#         print(f' maker new folder :{path1}')

#     imwrite(path,data,photometric='minisblack')
#     print(f"data successly save in ({path})" )
#     return

def save_tif_self(pathname,data,datatype=np.float32):
    """
    input：path+file name，data
    需指定地址
    """
    path = Path(str(pathname))   #得到运行这个文件的文件夹
    
    #code 文件上一个目录构建一个指定文件夹
    path1 = path.parent
    if not path1.exists():
        path1.mkdir()
        print(f' maker new folder :{path1}')

    imwrite(path,data,photometric='minisblack')
    
    min = np.min(data)
    max = np.max(data)
    with TiffWriter(path,imagej=True) as tif:
        tif.write(data.astype(datatype), metadata={'axes': 'ZYX','min': min,'max': max})
    
    print(f"data successly save in ({path})" )
    return

# def save_path(filename,run_parameter):
#     #只需传入需要保存的文件名字就行 eg *.tif
#     'code 文件上一个目录,构建dataOut文件夹'
#     path = Path.cwd()/'dataOut'
#     if not path.exists():
#         path.mkdir()
#         logger.info(f' maker new folder :{path}')
#     print(run_parameter['data_name'])
#     path_data = path/run_parameter['data_name']
#     if not path_data.exists():
#         path_data.mkdir()
#         logger.info(f' maker new folder :{path_data}')


#     path_data_dir = path_data/run_parameter['dir_name']
#     if not path_data_dir.exists():
#         path_data_dir.mkdir()
#         logger.info(f' maker new folder :{path_data_dir}')

#     savefile = path_data_dir/filename
#     return savefile


def save_path(filename,run_parameter,sub_filename=''):
    #只需传入需要保存的文件名字就行 eg *.tif
    'code 文件上一个目录,构建dataOut文件夹'

    path = Path.cwd()/'dataOut'
    print('_U\n',path)
    if not path.exists():
        path.mkdir()
        # logger.info(f' maker new folder :{path}')
    print(run_parameter['data_name'])
    path_data = path/run_parameter['data_name']
    if not path_data.exists():
        path_data.mkdir()
        # logger.info(f' maker new folder :{path_data}')


    path_data_dir = path_data/(run_parameter['dir_name']+"-"+run_parameter['ver'])
    if not path_data_dir.exists():
        path_data_dir.mkdir()
        # logger.info(f' maker new folder :{path_data_dir}')


    '次级文件夹'
    if sub_filename!='':
        path_data_dir = path_data_dir/sub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()
            # logger.info(f' maker new folder :{path_data_dir}')

    '文件路径'
    savefile = path_data_dir/filename

    return savefile

def normalize(data):
    '''
    复数实现归一化操作
    传入的矩阵可以是复数 可以是实数
    input：np or torch
    output：归一化后的矩阵
    并且类型相同，在GPUor 在CPU也相同
    '''
    #类型保存
    toTorch = 0
    if isinstance(data,torch.Tensor) :
        toTorch = 1
        data=data.numpy()


    #最大最小归一化
    maxx = data.max()
    minn = data.min()

    norm = (data-minn) / (maxx-minn) 

    #类型恢复
    if toTorch==1:
        return torch.from_numpy(norm)
    else:
        return norm

# def LOG(run_parameter):
#     # logger.add(run_parameter['log'], format='<green>{time:YYYY-MM-DD HH:mm:ss:SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>', filter=None, colorize=None, serialize=False, backtrace=True, diagnose=True, enqueue=False, catch=True)
#     logger.add(run_parameter['log'],enqueue=True)



def show_memory(unit='MB', threshold=1):
    '''查看变量占用内存情况

    :param unit: 显示的单位，可为`B`,`KB`,`MB`,`GB`
    :param threshold: 仅显示内存数值大于等于threshold的变量
    '''
    from sys import getsizeof
    scale = {'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}[unit]
    for i in list(globals().keys()):
        memory = eval("getsizeof({})".format(i)) // scale
        if memory >= threshold:
            print(i, memory)

def release():
    import gc
    gc.collect() 


def fill0(data,num=1):
    if data>1:
        return str(data).zfill(num)      #前面补零
    else:
        return str(data).ljust(num,'0')  #后边补零



###########################形态学操作#############
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
        save_tif(r'查看data.tif',data)
        save_tif(r'查看data_erode.tif',data_erode)
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
    data2=image_reshape(data,dim=3)
    nz,nx,ny = data2.shape
    for i in range(nz):
        data2[i]=Cir_Erode(data[i],size=size)
    return data2

def Cir_dilate_3D(data,size=3):
    #没有轴向腐蚀，只是分成slice这样腐蚀
    data2=image_reshape(data,dim=3)
    nz,nx,ny = data2.shape
    for i in range(nz):
        data2[i]=Cir_dilate(data[i],size=size)
    return data2
##################################################

def ifftn_RFN(data):
    "real、fushu、norm"
    try:
        data = data.real  # 取real
    except:
        pass
    data[data < 0] = 0  # 负数修正
    data = normalize(data)  # 最大最小归一化
    return data


def T_conv3D(image, psf):
    "要求传入3D的空域图像，psf，返回3D卷积结果"
    from fft_conv_pytorch import FFTConv3d, fft_conv

    assert len(image.shape) == 3
    assert len(psf.shape) == 3

    #     inz,inx,iny = image.shape
    pnz, pnx, pny = psf.shape
    
    "1.转为torch"
    timage = ToTensor(image).to(torch.float64).view(1, 1, *image.shape)
    tpsf = ToTensor(psf).to(torch.float64).view(1, 1, *psf.shape)
    # tpsf[tpsf<0]=0
    # tpsf = tpsf/torch.max(tpsf)
    "2.fft-conv"
    with torch.no_grad():
        fft_conv = FFTConv3d(
            1,
            1,
            (0),
            padding=(pnz // 2, pnx // 2, pny // 2),
            padding_mode="constant",
            bias=False,
        )  #  卷积之后的尺寸为 2p+1
        fft_conv.weight = torch.nn.Parameter(tpsf)
        out = fft_conv(timage)

    "3.数据后处理"
    out = out[0, 0, :-1, :-1, :-1]  # 删除多余的
    out[out<0] = 0
    out = out
    # out = ifftn_RFN(out)

    return image_reshape(out, dim=3)  # 维度修正 #输出torch