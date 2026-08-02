import numpy as np
from numpy import pi, cos, sin,exp
import torch 
from pathlib import Path
from loguru import logger
# import pickle

from tifffile import imread, imwrite
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
        h = (h).numpy()
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
def meshgridxy_self(nx,ny,nz=1,index='ij'):
    #这里计算xyz 三维的向量
    #分奇偶 构建坐标系
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    z = np.arange(np.ceil(-nz/2),np.ceil(nz/2))
    zv, xv, yv = np.meshgrid(z, x, y, indexing=index, sparse=True)
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
def Interpolate(opt,image,nx,ny,nz,strt = 'trilinear'):
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
def save_tif(filename,data,sub_filename=''):
    """
    input：data名，data
    process: if new_dir!= None  make folder
    无需传入地址
    """
    #code 文件上一个目录构建一个 dataOut 并同名的文件夹
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

    '文件保存'
    #将文件保存到这里
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
    return

def save_tif_v21(filename,data,run_parameter,sub_filename='',subsub_filename=''):
    """
    input：需要保存的数据名字，data
    process: if new_dir!= None  make folder
    无需传入地址

    V2.1 ： 自动在dataout目录下新建一个文件夹，命名方式为：/X月-xx-遍历文件名/
    """
    # LOG(run_parameter)
    data = image_reshape(data,dim=3)
    'code 文件上一个目录,构建dataOut文件夹'
    path = Path.cwd()/'dataOut'
    if not path.exists():
        path.mkdir()
        # logger.info(f' maker new folder :{path}')
    
    path_data = path/run_parameter['data_name']
    if not path_data.exists():
        path_data.mkdir()
        # logger.info(f' maker new folder :{path_data}')


    path_data_dir = path_data/run_parameter['dir_name']
    if not path_data_dir.exists():
        path_data_dir.mkdir()
        # logger.info(f' maker new folder :{path_data_dir}')

    '次级文件夹'
    if sub_filename!='':
        path_data_dir = path_data_dir/sub_filename
        if not path_data_dir.exists():
            path_data_dir.mkdir()
            # logger.info(f' maker new folder :{path_data_dir}')
    '次级文件夹'
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

def save_txt(filename,data):
    """
    input：data名，data{dict}
    """
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


def save_tif_self(pathname,data):
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
    print(f"data successly save in ({path})" )
    return

def save_tif_self2(path_data_dir,data,sub_filename='',subsub_filename='',filename = '',axes = 'ZYX'):
    """
    input：path_data_dir，最多两级次文件夹
    process:
    """
    #code 文件上一个目录构建一个 dataOut 并同名的文件夹
    if axes=='ZYX':
        data = image_reshape(data,dim=3)

    # path = Path.cwd()/'dataOut'
    # if not path.exists():
    #     path.mkdir()

    # path_data_dir = path/Path.cwd().name  #
    # if not path_data_dir.exists():
    #     path_data_dir.mkdir()
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
        savefile,data,#.transpose(),
        imagej=True,
        photometric='minisblack',
        metadata={'axes': axes}
        )
    print(f"data successly save in ({savefile})")
    return

# def save_path(filename):
#     #只需传入需要保存的文件名字就行 eg *.tif
#     path = Path.cwd()/'dataOut'
#     if not path.exists():
#         path.mkdir()
#         print(f' maker new folder :{path}')
    
#     savefolder = path/Path.cwd().name  #
#     if not savefolder.exists():
#         savefolder.mkdir()
#         print(f' maker new folder :{savefolder}')

#     savefile = savefolder/filename
#     return savefile

def save_path(filename,run_parameter):
    #只需传入需要保存的文件名字就行 eg *.tif
    'code 文件上一个目录,构建dataOut文件夹'
    path = Path.cwd()/'dataOut'
    if not path.exists():
        path.mkdir()
        logger.info(f' maker new folder :{path}')
    print(run_parameter['data_name'])
    path_data = path/run_parameter['data_name']
    if not path_data.exists():
        path_data.mkdir()
        logger.info(f' maker new folder :{path_data}')


    path_data_dir = path_data/run_parameter['dir_name']
    if not path_data_dir.exists():
        path_data_dir.mkdir()
        logger.info(f' maker new folder :{path_data_dir}')

    savefile = path_data_dir/filename
    return savefile


def sift_analysis_report_path(run_parameter):
    peizhun_file = str(run_parameter['peizhun_file'])
    report_dir = save_path('', run_parameter) / f'SIFT匹配-{peizhun_file}'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f'SIFT分析报告-{peizhun_file}.md'
    return existing_quality_report_path(report_path)


def record_quality_warning(run_parameter, message):
    message = str(message)
    if run_parameter is not None:
        warnings = run_parameter.setdefault('quality_warnings', [])
        if message not in warnings:
            warnings.append(message)
    print(f'【质量告警已记录，流程继续】{message}')
    return message


def print_quality_warning_summary(run_parameter):
    warnings = [] if run_parameter is None else run_parameter.get('quality_warnings', [])
    warnings = list(dict.fromkeys(str(message) for message in warnings))
    if not warnings:
        return 0

    border = '!' * 78
    print(f'\n{border}')
    print(f'!!! 质量告警汇总：处理已完成，但检测到 {len(warnings)} 项质量下降 !!!')
    for index, message in enumerate(warnings, start=1):
        print(f'[{index}] {message}')
    print('!!! 请检查上述质量报告和对应输出，确认后再用于定量分析。!!!')
    print(border)
    return len(warnings)


QUALITY_REPORT_MARKERS = ('BG！', 'foc！')


def _quality_report_parts(report_path):
    report_path = Path(report_path)
    base_stem = report_path.stem
    markers = set()
    while True:
        matched = False
        for marker in QUALITY_REPORT_MARKERS:
            token = f'-{marker}'
            if base_stem.endswith(token):
                base_stem = base_stem[:-len(token)]
                markers.add(marker)
                matched = True
                break
        if not matched:
            return report_path, base_stem, markers


def _quality_report_path(report_path, markers):
    report_path, base_stem, _ = _quality_report_parts(report_path)
    marker_suffix = ''.join(
        f'-{marker}' for marker in QUALITY_REPORT_MARKERS if marker in markers
    )
    return report_path.with_name(f'{base_stem}{marker_suffix}{report_path.suffix}')


def existing_quality_report_path(report_path):
    report_path, _, existing_markers = _quality_report_parts(report_path)
    marker_sets = [
        set(QUALITY_REPORT_MARKERS),
        {'BG！'},
        {'foc！'},
        existing_markers,
        set(),
    ]
    for markers in marker_sets:
        candidate = _quality_report_path(report_path, markers)
        if candidate.exists():
            return candidate
    return _quality_report_path(report_path, existing_markers)


def mark_quality_report_path(report_path, marker):
    if marker not in QUALITY_REPORT_MARKERS:
        raise ValueError(f'未知质量报告标记: {marker}')

    source_path = existing_quality_report_path(report_path)
    _, _, markers = _quality_report_parts(source_path)
    markers.add(marker)
    target_path = _quality_report_path(source_path, markers)
    if source_path == target_path or not source_path.exists():
        return target_path
    if target_path.exists():
        raise FileExistsError(f'质量报告标记目标已存在，未覆盖: {target_path}')
    source_path.rename(target_path)
    return target_path


def record_quality_report_marker(run_parameter, marker):
    if marker not in QUALITY_REPORT_MARKERS:
        raise ValueError(f'未知质量报告标记: {marker}')
    if run_parameter is not None:
        markers = run_parameter.setdefault('quality_report_markers', [])
        if marker not in markers:
            markers.append(marker)
    return marker


def apply_quality_report_markers(report_path, markers):
    marked_path = existing_quality_report_path(report_path)
    for marker in markers:
        marked_path = mark_quality_report_path(marked_path, marker)
    return marked_path

# def normalize(data):
#     '''
#     复数实现归一化操作
#     传入的矩阵可以是复数 可以是实数
#     input：np or torch
#     output：归一化后的矩阵
#     并且类型相同，在GPUor 在CPU也相同
#     '''
#     #类型保存
#     toTorch = 0
#     if isinstance(data,torch.Tensor) :
#         toTorch = 1
#         data=data.numpy()


#     #最大最小归一化
#     maxx = data.max()
#     minn = data.min()

#     norm = (data-minn) / (maxx-minn) 

#     #类型恢复
#     if toTorch==1:
#         return torch.from_numpy(norm)
#     else:
#         return norm
    
def normalize(data):
    # 最大值归一化
    from math import sqrt
    
    if type(data) == complex:
        real = data.real
        imag = data.imag
        temp = sqrt(real*real+imag*imag)
        real = real/temp
        imag = imag/temp * 1j
        norm_data = real + imag
    elif isinstance(data,torch.Tensor):
        mag = torch.abs(data)
        max_mag = torch.max(mag)
        norm_data = data/max_mag
    elif isinstance(data,np.ndarray):
        mag = np.abs(data)
        max_mag = np.max(mag)
        norm_data = data/max_mag
    else:
        raise(r'error')
    
    return norm_data


def LOG(run_parameter):
    # logger.add(run_parameter['log'], format='<green>{time:YYYY-MM-DD HH:mm:ss:SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>', filter=None, colorize=None, serialize=False, backtrace=True, diagnose=True, enqueue=False, catch=True)
    logger.add(run_parameter['log'],enqueue=True)



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


def fill0(data,num=4):
    if data>1:
        return str(data).zfill(num)      #前面补零
    else:
        return str(data).ljust(num,'0')  #后边补零
