from pathlib import Path
from tifffile import imread, imwrite
import numpy as np
import torch as tt
# import BaseFunctions_np_7_1 as n
# import BaseFunctions_tt_7_1 as t

import sys 
sys.path.append('../')
import Utils as _U

import cv2
from scipy.signal import savgol_filter

#读入beads 文件，分割8幅图像，配准？ 切割ROI区域，得到8副PSF，


def meshgrid(image,kuoda=2,index='ij'):
    #这里计算xyz 三维的向量
    #分奇偶 构建坐标系
    nx = image.shape[0] * kuoda
    ny = image.shape[1]*kuoda
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    xv, yv = np.meshgrid(x, y, indexing=index, sparse=True)
    return xv,yv

def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return tt.from_numpy(new_arr.copy())

def dft_corr2d(h,g):# tt  h,g求的互相关
    """进行二维互相关"""
    g = tt.conj(g)   #先求共轭
    g = (flip180(g.numpy()))#再翻转180°

    k= np.array(h.shape)
    L = np.array(g.shape)
    f = tt.zeros_like(tt.from_numpy(k+L),dtype = tt.complex32)
    f = tt.fft.ifft2(tt.fft.fft2(h,s=tuple(k+L) )*tt.fft.fft2((g),s=tuple(k+L)))
    return f
    
def corrlate(imagA,imagB):
    #传入2张 2D 图像 做互相关
    nx,_ = imagA.shape
    pg = 40
    fanwei = 40
    eps = np.finfo(float).eps
    Hzuan = np.ones((nx,nx))
    cishu = dft_corr2d(tt.from_numpy(Hzuan),tt.from_numpy(Hzuan))     #消除面积效应

    jieguo = dft_corr2d(tt.from_numpy(imagA),tt.from_numpy(imagB));          #用m=0阶 和m=2阶来求互相关
    lihe = abs(jieguo/(cishu+eps))
    lihe = lihe.numpy()
    xv2,yv2= meshgrid(imagA,kuoda=2)
    kr = np.sqrt(xv2**2+yv2**2)
    lihe[kr>(pg+fanwei)]=0
    lihe[kr<(pg-fanwei)]=0

    # print((abs(lihe)==np.max(abs(lihe))).nonzero()) #这个为最大的值，坐标
    # print((lihe==np.max(lihe)).nonzero())    #找到最大值的坐标
    save_tif("lihe.tif",abs(lihe))
    maxx = np.where(lihe==np.max(lihe))[0][0] # 
    maxy = np.where(lihe==np.max(lihe))[1][0] # 
    return maxx,maxy

def cal_corrlate(project,base=4):
    #NCC
    coordinate = np.zeros((8,2))*512
    A = project[base]/np.max(project[base]) #以第N幅图
    for i in range(8):
        coordinate[i] = corrlate(A,project[i]/np.max(project[i]))
    return coordinate


def project_pipei(project,coordinate):
    project2 = np.zeros_like(project)
    for i in range(project.shape[0]):
        x_move = int(coordinate[i][0])
        y_move = int(coordinate[i][1])
        
        #移动x
        project2[i] = np.roll(project[i],x_move,axis=0)
        if x_move>=0:
            project2[i][:x_move,:] = 0
        else:
            project2[i][x_move:,:] = 0

        #移动y
        project2[i] = np.roll(project2[i],y_move,axis=1)
        if y_move>=0:
            project2[i][:,:y_move] = 0
        else:
            project2[i][:,y_move:] = 0
    return project2


def save_tif(filename,data):
    _U.save_tif(filename,data)


#切分图片为imag1 imag2 iamg3 iamg4
def load_RAW_DATA_1x4(pathfiname,camer_name,suoxiao=1,Debug=0):
    '有分光系数的调整 其实有漂白校正的话，这个分光系数校准就不重要了'
    Dr_read = imread(pathfiname)/suoxiao

    #减去背景
    # bg1 = imread(r'E:\4TB图片\python 实验数据\仪器配准实验数据\3.beads球差\BG\AVG_cam1_background_MMStack_Pos0.ome.tif')
    # bg2 = imread(r'E:\4TB图片\python 实验数据\仪器配准实验数据\3.beads球差\BG\AVG_cam2_background_MMStack_Pos0.ome.tif')
    bg1 = imread(r'../../AVG_cam1_background.tif')
    bg2 = imread(r'../../AVG_cam2_background.tif')
    
    # Dr_read = Dr_read[1:]    #不要第一张
    if camer_name=='cam2':
        Dr_read = Dr_read -bg2
    elif camer_name=='cam1':
        Dr_read = Dr_read -bg1
    c1=np.array([586,475,527,437])-np.mean(bg2)
    c2=np.array([445,591,483,667])-np.mean(bg2)
    # Dr_read = 100*(Dr_read-np.min(Dr_read))/ (np.max(Dr_read)-np.min(Dr_read))    #进行最大最小归一化

    call = np.sum(c1+c2)
    if len(Dr_read.shape)==2:
        nx,ny4 = Dr_read.shape  # 
        ny = int(ny4/4)
        imag_1 = Dr_read[:,0:ny]
        imag_2 = Dr_read[:,ny:2*ny]
        imag_3 = Dr_read[:,2*ny:3*ny]
        imag_4 = Dr_read[:,3*ny:4*ny]
        nz=1
    
    elif len(Dr_read.shape)==3:  #读入的图片为stack
        nz,nx,ny4 = Dr_read.shape
        ny = int(ny4/4)
        imag_1 = Dr_read[:,:,0:ny]
        imag_2 = Dr_read[:,:,ny:2*ny]
        imag_3 = Dr_read[:,:,2*ny:3*ny]
        imag_4 = Dr_read[:,:,3*ny:4*ny]
        
    else:
        print((Dr_read.shape))
        channle,nz,nx,ny4 = Dr_read.shape
        ny = int(ny4/4)
        imag_1 = Dr_read[0,:,:,0:ny].reshape(nz,nx,ny)
        imag_2 = Dr_read[0,:,:,ny:2*ny].reshape(nz,nx,ny)
        imag_3 = Dr_read[0,:,:,2*ny:3*ny].reshape(nz,nx,ny)
        imag_4 = Dr_read[0,:,:,3*ny:4*ny].reshape(nz,nx,ny)
    
    bili = []
    if camer_name=='cam2': #将镜像对称的图片翻转
        print('将镜像对称的图片翻转')
        imag_1=fanzhuan(imag_1)
        imag_2=fanzhuan(imag_2)
        imag_3=fanzhuan(imag_3)
        imag_4=fanzhuan(imag_4)
        
        imag_1 = imag_1*(call/c2[0])
        imag_2 = imag_2*(call/c2[1])
        imag_3 = imag_3*(call/c2[2])
        imag_4 = imag_4*(call/c2[3])
        print("imag_5",np.mean(imag_1))
        print("imag_6",np.mean(imag_2))
        print("imag_7",np.mean(imag_3))
        print("imag_8",np.mean(imag_4))

        bili.append(Outlier_analysis(imag_1))
        bili.append(Outlier_analysis(imag_2))
        bili.append(Outlier_analysis(imag_3))
        bili.append(Outlier_analysis(imag_4))

    if camer_name=='cam1': #调整分光镜系数
        imag_1 = imag_1*(call/c1[0])
        imag_2 = imag_2*(call/c1[1])
        imag_3 = imag_3*(call/c1[2])
        imag_4 = imag_4*(call/c1[3])
        print("imag_1",np.mean(imag_1))
        print("imag_2",np.mean(imag_2))
        print("imag_3",np.mean(imag_3))
        print("imag_4",np.mean(imag_4))

        bili.append(Outlier_analysis(imag_1))
        bili.append(Outlier_analysis(imag_2))
        bili.append(Outlier_analysis(imag_3))
        bili.append(Outlier_analysis(imag_4))

    return imag_1,imag_2,imag_3,imag_4,nx,ny,nz,bili


def load_RAW_DATA_1x4_2(pathfiname,camer_name,suoxiao=1,Debug=0):
    Dr_read = imread(pathfiname)/suoxiao

    #减去背景
    # bg1 = imread(r'E:\4TB图片\python 实验数据\仪器配准实验数据\3.beads球差\BG\AVG_cam1_background_MMStack_Pos0.ome.tif')
    # bg2 = imread(r'E:\4TB图片\python 实验数据\仪器配准实验数据\3.beads球差\BG\AVG_cam2_background_MMStack_Pos0.ome.tif')
    bg1 = imread(r'../../AVG_cam1_background.tif')
    bg2 = imread(r'../../AVG_cam2_background.tif')
    # Dr_read = Dr_read[1:]    #不要第一张
    if camer_name=='cam2':
        Dr_read = Dr_read -bg2
    elif camer_name=='cam1':
        Dr_read = Dr_read -bg1
    Dr_read[Dr_read<0] = 0
    Dr_read = 100.0* ((Dr_read-np.min(Dr_read))/ (np.max(Dr_read)-np.min(Dr_read)))   #进行最大最小归一化0~100
    if len(Dr_read.shape)==2:
        nx,ny4 = Dr_read.shape  # 
        ny = int(ny4/4)
        imag_1 = Dr_read[:,0:ny]
        imag_2 = Dr_read[:,ny:2*ny]
        imag_3 = Dr_read[:,2*ny:3*ny]
        imag_4 = Dr_read[:,3*ny:4*ny]
        nz=1
    
    elif len(Dr_read.shape)==3:  #读入的图片为stack
        nz,nx,ny4 = Dr_read.shape
        ny = int(ny4/4)
        imag_1 = Dr_read[:,:,0:ny]
        imag_2 = Dr_read[:,:,ny:2*ny]
        imag_3 = Dr_read[:,:,2*ny:3*ny]
        imag_4 = Dr_read[:,:,3*ny:4*ny]
        
    else:
        print((Dr_read.shape))
        channle,nz,nx,ny4 = Dr_read.shape
        ny = int(ny4/4)
        imag_1 = Dr_read[0,:,:,0:ny].reshape(nz,nx,ny)
        imag_2 = Dr_read[0,:,:,ny:2*ny].reshape(nz,nx,ny)
        imag_3 = Dr_read[0,:,:,2*ny:3*ny].reshape(nz,nx,ny)
        imag_4 = Dr_read[0,:,:,3*ny:4*ny].reshape(nz,nx,ny)
    
    bili = []
    if camer_name=='cam2': #将镜像对称的图片翻转
        print('将镜像对称的图片翻转')
        imag_1=fanzhuan(imag_1)
        imag_2=fanzhuan(imag_2)
        imag_3=fanzhuan(imag_3)
        imag_4=fanzhuan(imag_4)
        
        print("imag_5",np.mean(imag_1))
        print("imag_6",np.mean(imag_2))
        print("imag_7",np.mean(imag_3))
        print("imag_8",np.mean(imag_4))

        bili.append(Outlier_analysis(imag_1))
        bili.append(Outlier_analysis(imag_2))
        bili.append(Outlier_analysis(imag_3))
        bili.append(Outlier_analysis(imag_4))

    if camer_name=='cam1': #调整分光镜系数
        print("imag_1",np.mean(imag_1))
        print("imag_2",np.mean(imag_2))
        print("imag_3",np.mean(imag_3))
        print("imag_4",np.mean(imag_4))

        bili.append(Outlier_analysis(imag_1))
        bili.append(Outlier_analysis(imag_2))
        bili.append(Outlier_analysis(imag_3))
        bili.append(Outlier_analysis(imag_4))

    return imag_1,imag_2,imag_3,imag_4,nx,ny,nz,bili

def fanzhuan(arr):
    return arr[:,:,::-1]

#离群值分析
def Outlier_analysis(image):
    import pandas as pd
    import matplotlib.pyplot as plt

    image= image.reshape(-1)


    # fig = plt.figure()
    # view = plt.boxplot(image)
    # plt.show()

    pd_date =pd.Series(image).describe()
    # print(pd_date)
    Q1 = pd_date[4]
    Q3 = pd_date[6]
    Q13_index = np.where( np.logical_and(image > Q1 , image< Q3) ) #计算下标
    Q1Q3_mean = image[Q13_index].mean() #计算Q1~Q3之间的均值
    print('Q1Q3-mean',Q1Q3_mean.round(4))
    return Q1Q3_mean.round(4)

def get_tenengrad_project(images, crop=16, gaussian_sigma=1.0):
    """为每个区域选择 Tenengrad 分数最高的焦面层。"""
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f'images must be (region, z, x, y), got {images.shape}')

    region_n, nz, nx, ny = images.shape
    if crop > 0 and nx > 2 * crop and ny > 2 * crop:
        score_images = images[:, :, crop:-crop, crop:-crop]
    else:
        score_images = images

    scores = np.zeros((region_n, nz), dtype=np.float64)
    for region_i in range(region_n):
        region = score_images[region_i].astype(np.float32, copy=False)
        low, high = np.percentile(region, (1, 99))
        region = np.clip((region - low) / (high - low + 1e-12), 0, 1)

        for z_i in range(nz):
            image = cv2.GaussianBlur(
                region[z_i],
                (0, 0),
                sigmaX=gaussian_sigma,
                sigmaY=gaussian_sigma,
            )
            gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            scores[region_i, z_i] = np.mean(
                gradient_x * gradient_x + gradient_y * gradient_y
            )

    if nz >= 7:
        scores = savgol_filter(scores, window_length=7, polyorder=2, axis=1, mode='interp')

    peak_scores = np.max(scores, axis=1)
    invalid_regions = np.where(~np.isfinite(peak_scores) | (peak_scores <= 0))[0]
    if invalid_regions.size:
        raise ValueError(
            'Tenengrad cannot determine focus for regions '
            f'{(invalid_regions + 1).tolist()}'
        )

    focus_indices = np.argmax(scores, axis=1)
    project = images[np.arange(region_n), focus_indices]
    return project, focus_indices, scores


def get_focus_spacing_analysis(focus_indices, stack_layers):
    focus_indices = np.asarray(focus_indices, dtype=int)
    if focus_indices.shape != (8,):
        raise ValueError(
            f'focus_indices must contain 8 regions, got shape {focus_indices.shape}'
        )

    spacing_limit = {60: 4, 120: 8}.get(int(stack_layers))
    gaps = np.abs(np.diff(focus_indices))
    if spacing_limit is None:
        invalid_pairs = list(range(7))
        status = 'unsupported_stack_layers'
    else:
        invalid_pairs = np.flatnonzero(gaps > spacing_limit).tolist()
        status = 'invalid_spacing' if invalid_pairs else 'ok'

    return {
        'stack_layers': int(stack_layers),
        'spacing_limit_layers': spacing_limit,
        'adjacent_gaps_layers': gaps.tolist(),
        'invalid_pair_indices_0based': invalid_pairs,
        'status': status,
    }


#####55 单纯的投影
def get_project(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,method='max',Debug=0):
    _,nx,ny = imag_1.shape
    project = np.zeros((8,nx,ny))
    if method == 'max':
        project[0] = np.max(imag_1,axis=0)
        project[1] = np.max(imag_2,axis=0)
        project[2] = np.max(imag_3,axis=0)
        project[3] = np.max(imag_4,axis=0)
        project[4] = np.max(imag_5,axis=0)
        project[5] = np.max(imag_6,axis=0)
        project[6] = np.max(imag_7,axis=0)
        project[7] = np.max(imag_8,axis=0)
    elif method == 'mean':
        project[0] = np.mean(imag_1,axis=0)
        project[1] = np.mean(imag_2,axis=0)
        project[2] = np.mean(imag_3,axis=0)
        project[3] = np.mean(imag_4,axis=0)
        project[4] = np.mean(imag_5,axis=0)
        project[5] = np.mean(imag_6,axis=0)
        project[6] = np.mean(imag_7,axis=0)
        project[7] = np.mean(imag_8,axis=0)
    elif method == 'tenengrad':
        images = np.array([imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8])
        project, _, _ = get_tenengrad_project(images)
    else:
        raise ValueError(f'method error: {method}')
    if Debug==1:
        for i in range(8):
            save_tif(f'55 project_{i}.tif',project[i])
    return project

def get_project45(imag_4,imag_5,method='max',Debug=0):
    _,nx,ny = imag_4.shape
    project = np.zeros((2,nx,ny))
    if method == 'max':
        project[0] = np.max(imag_4,axis=0)
        project[1] = np.max(imag_5,axis=0)
    elif method == 'mean':
        project[0] = np.mean(imag_4,axis=0)
        project[1] = np.mean(imag_5,axis=0)
    else:
        raise('method error!')
    if Debug==1:
        for i in range(2):
            save_tif(f'55 project_{i}.tif',project[i])
    return project

def get_slice45(imag_4,imag_5,z,method='max',Debug=0):
    #利用好条纹信息
    _,nx,ny = imag_4.shape
    project = np.zeros((2,nx,ny))
    project[0] = imag_4[z]
    project[1] = imag_5[z]
    if Debug==1:
        for i in range(2):
            save_tif(f'55 project_{i}.tif',project[i])
    return project



#### 666 找ROI
def fbi_glt(dic,imag,index,m,name=''):
    imag2 = np.copy(imag)
    x_set = index[0]
    y_set = index[1]
    for k in list(dic.keys()):
        v = dic[k]
        if len(v)>1:
            v_binahao = k
            for i in v:
                x=x_set[i]
                y=y_set[i]
                imag2[x,y] =v_binahao
    save_tif(f'{m}-归类查看-{name}.tif',imag2)

def fbi_diff_point(dic,image,index,size=30):
    dic2 = dict.copy(dic)
    x_set = index[0]
    y_set = index[1]
    del_list = []
    for k in range(len(dic)): #list(func_dict.keys())
        v = dic[k] # [, , , ]
        ind_ori = findmax2D(index,v,image)
        x = x_set[ind_ori]
        y = y_set[ind_ori]
        for kk in range(k+1,len(dic)):
            vv = dic[kk] # [, , , ]
            ind_orii = findmax2D(index,vv,image)
            xx = x_set[ind_orii]
            yy = y_set[ind_orii]
            if np.sqrt((x-xx)**2+(y-yy)**2)< (size/2)*np.sqrt(2):
                print(f'两个相聚小于{size/2},删除掉:,({x},{y}),({xx},{yy})')
                del_list.append(k)
                del_list.append(kk)  #记录要删除的k值
    #列表去重
    #删除这些字典值
    del_list = set(del_list)
    for kk in del_list:
        del dic2[kk]
    return dic2
    
def del_dic_n(dic,n=1):
    dic2 = dict.copy(dic)
    #删除字典中少于指定数量的key值
    for k in list(dic2.keys()):
        v = dic2[k]
        if len(v)<=n:
            del dic2[k]
    return dic2

def fbi_del_margin(dic,image,index,coordinate,margin=15):
    #应该还需要修正这些点，因为移动后，每边的margin不一样了
    dic2 = dict.copy(dic)
    nx,ny = image.shape
    move_marg_x = coordinate[0]
    move_marg_y = coordinate[1]
    x_set = index[0] 
    y_set = index[1] 

    for k in list(dic2.keys()): 
        v = dic2[k]
        ind_ori = findmax2D(index,v,image)
        x_ori = x_set[ind_ori]
        y_ori = y_set[ind_ori]
        if  x_ori<(margin) or y_ori<(margin) or (nx-x_ori)<(margin) or(ny-y_ori)<(margin) or  (x_ori-move_marg_x)<(margin) or (y_ori-move_marg_y)<(margin) or (nx-(x_ori-move_marg_x))<(margin) or (ny-(y_ori-move_marg_y))<(margin)   :  #边界点判定
            print(f'1:({x_ori},{y_ori})为边界点,跳过这个ROI')
            del dic2[k]
    return dic2
    
def fbi_renum(dic):
    dic2 = {}
    j=0
    for k,v in dic.items():
        dic2[j] = v
        j+=1
    return dic2

def fbi_JL(index,JL_JL2):
    j=0
    x_set = index[0]
    y_set = index[1]
    flag = np.ones_like(x_set)
    dic={}  #每一个区域一个dic
    kk=0
    for i in range(len(x_set)):   #这里双for 很吃速度
        x0 = x_set[i]
        y0 = y_set[i]
        temp = np.zeros((0),dtype=np.int16)
        temp = np.append(temp,int(i))
        if flag[i]==1:
            for j in range(i+1,len(x_set)):
                temp_num=1  #重置为1
                if flag[j]==1:
                    x1 = x_set[j]
                    y1 = y_set[j]
                    r = np.sqrt( (x0-x1)**2+(y0-y1)**2 ) #计算同一副图像区域中的点的
                    if r <= JL_JL2:  #距离小于r_julei的点划分为同一类 建议为8个像素
                        temp = np.append(temp,int(j))
                        flag[j] = 0
                        #更新z0,x0,y0 中心
                        x0 = (x0*temp_num+x1)/(temp_num+1)
                        y0 = (y0*temp_num+y1)/(temp_num+1)
                        temp_num+=1

        dic[kk] = temp[:]   #把numpy 加入字典j中
        kk+=1
    return dic
    
def find_beads_index(project,coordinate,YZ1=200,JL_JL2=5,jugle_noise3=4,size=30,Debug=0,margin=15):
    #input 2D project x 8
    #output index    每个区域一个 [[]] 
    #加入coordinate，解决移动后的边界点问题
    zXY_lists = [] #（#8，rois，2）
    
    for m in range(project.shape[0]):#1 2 3 4 5678 
        image_2D = np.copy(project[m])
        index = np.where((image_2D)>=YZ1)                #得到阈值以上区域的所有点,进行一个γ校正
        nx = image_2D.shape[0]                         #图像属性，用于边界点判定
        ny = image_2D.shape[1]
        
        #标记一下2D图像
        for i in range(len(index[0])):
            xxx = index[0][i]
            yyy = index[1][i]
            image_2D[xxx,yyy] = 0
        if Debug==1:
            save_tif(f"⑥{m}-投影大于阈值的所有做标记2.tif",image_2D)
            save_tif(f"⑥{m}-投影1.tif",project[m])

        #用一个字典来聚类
        dic = fbi_JL(index,JL_JL2)
        
        #删除字典中只有1的value的情况
        dic = del_dic_n(dic,n=1 )
        #归类图 
        if Debug==1:
            fbi_glt(dic,project[m],index,m,name = '⑥del_isolate_point3') #根据字典中的排列，查看分类情况
        
        #对字典重新编号
        dic2 = fbi_renum(dic)

        #删除挨在一着的两个点
        dic2 = fbi_diff_point(dic2,project[m],index,size=size)
        #查看一下
        if Debug==1:
            fbi_glt(dic2,project[m],index,m,name='⑥del_diff_point4')
        
        #删除噪声点 
        dic2 = del_dic_n(dic2,n=jugle_noise3)
        #查看一下
        if Debug==1:
            fbi_glt(dic2,project[m],index,m,name='⑥del_noise_point5')

        #删除边界点
        dic2 = fbi_del_margin( dic2,project[m],index,coordinate[m],margin=margin)
        #查看一下
        if Debug==1:
            fbi_glt(dic2,project[m],index,m,name='⑥del_margin_point6')

        ##
        x_set = index[0]
        y_set = index[1]
        zXY_list=[]
        for k in list(dic2.keys()): # 在m区域中遍历每个ROI，找到每个ROI的最亮点
            v = dic2[k]
            ind_ori = findmax2D(index,v,project[m])
            x_ori = x_set[ind_ori]
            y_ori = y_set[ind_ori]    
            temp = [x_ori,y_ori]
            zXY_list.append(temp)
        zXY_lists.append(zXY_list)
    return zXY_lists   #4个区域上de ROI

def findmax2D(index,v,imag_1):
    #聚类完成后，找到其中的最大值
    #v为聚类的下标
    index_max=0
    maxx=0 
    x_set = index[0]
    y_set = index[1]
#     print('v=',v)
    for i in v:
        i=int(i)
        if maxx<imag_1[x_set[i],y_set[i] ]:
            maxx = imag_1[ x_set[i],y_set[i] ]
            index_max = i
#     print('maxx= ',maxx)
#     print('index = ',index_max)
    return index_max


#777
def get_right_list(zXY_list,MatchXY,Debug=0):
    #input：zXY_list (8,roi_s,2)，MatchXY
    #    process：8个区域上匹配对应点（按照距离MatchXY来计算）
    #             del 那些没匹配上的点
    #output：right_list
    ROI_n = len(zXY_list[0]) #8
    
    right_list = np.zeros((ROI_n,8,2),dtype = np.int16 )    #roi数目，4幅图，xy坐标点

    for i in range(ROI_n): #right_list为一个（8,2）的数组
        right_list[i]  =  Right_List(i,zXY_list,MatchXY=MatchXY) #ROI匹配函数
    if Debug==1:
        print("②删除之前right_list:",right_list) #正确
    #
    i = -1
    while(True):
        i += 1
        end = right_list.shape[0]
        if np.isin(right_list[i],(0,0)).any()==True:
            right_list = np.delete(right_list,i,axis=0)
            i = -1
            end -=1
        else:
            if i>=end-1:
                break
    if Debug==1:
        print("②删除之后right_list:",right_list) #(roi数目，第几副图，坐标点)
    if right_list.shape[0]==0:
        print("②全删完了")
    return right_list


#ROI匹配函数

def Right_List(i,zXY_list,MatchXY):
    #传入对应的ROI区域i  (8 rois 2)
    #返回二维数组，(8 2)
    right_list_recoder = np.zeros((8,2))    
    print(f'{i}th roi matching:')
    zXY_list_1 = zXY_list[0]
    zXY_list_2 = zXY_list[1]
    zXY_list_3 = zXY_list[2]
    zXY_list_4 = zXY_list[3]
    zXY_list_5 = zXY_list[4]
    zXY_list_6 = zXY_list[5]
    zXY_list_7 = zXY_list[6]
    zXY_list_8 = zXY_list[7]
    
    x1 = zXY_list_1[i][0]  #第①个区域的i个roi
    y1 = zXY_list_1[i][1]
    print(f'x1={x1},y1={y1}')
    right_list_recoder[0]=np.array((x1,y1))
    
    temp_julu = MatchXY
    for j in range(len(zXY_list_2)): #遍历第二个区域的所有roi信息
        x2 = zXY_list_2[j][0]
        y2 = zXY_list_2[j][1]
        jjll = juli(x1,y1,x2,y2)
        if jjll<temp_julu :   #计算三维距离，这个Z的计算需要单独拿出来  #这个5 是超参数   
            print(f'2-匹配roi:{x2},{y2},juli={jjll}')
            temp_julu = jjll
            right_list_recoder[1]= np.array((x2,y2))   #遍历所有最小值
            # break

    temp_julu = MatchXY
    for j in range(len(zXY_list_3)):
        x3 = zXY_list_3[j][0]
        y3 = zXY_list_3[j][1]
        jjll = juli(x1,y1,x3,y3)
        if jjll<temp_julu :   #计算三维距离
            print(f'3-匹配roi:{x3},{y3},juli={jjll}')
            temp_julu = jjll
            right_list_recoder[2]= np.array((x3,y3))
            # break

    temp_julu = MatchXY
    for j in range(len(zXY_list_4)):
        x4 = zXY_list_4[j][0]
        y4 = zXY_list_4[j][1]
        jjll = juli(x1,y1,x4,y4)
        if jjll<temp_julu :   #计算三维距离
            print(f'4-匹配roi:{x4},{y4},juli={jjll}')
            temp_julu = jjll
            right_list_recoder[3]= np.array((x4,y4))
            # break
    temp_julu = MatchXY
    for j in range(len(zXY_list_5)):
        x5 = zXY_list_5[j][0]
        y5 = zXY_list_5[j][1]
        jjll = juli(x1,y1,x5,y5)
        if jjll<temp_julu :   #计算三维距离
            print(f'5-匹配roi:{x5},{y5},juli={jjll}')
            temp_julu = jjll
            right_list_recoder[4]= np.array((x5,y5))
            # break

    temp_julu = MatchXY
    for j in range(len(zXY_list_6)):
        x6 = zXY_list_6[j][0]
        y6 = zXY_list_6[j][1]
        jjll = juli(x1,y1,x6,y6)
        if jjll<temp_julu :   #计算三维距离
            print(f'6-匹配roi:{x6},{y6},juli={jjll}')
            temp_julu = jjll
            right_list_recoder[5]= np.array((x6,y6))
            # break
    temp_julu = MatchXY
    for j in range(len(zXY_list_7)):
        x7 = zXY_list_7[j][0]
        y7 = zXY_list_7[j][1]
        jjll = juli(x1,y1,x7,y7)
        if jjll<temp_julu :   #计算三维距离
            print(f'7-匹配roi:{x7},{y7},juli={jjll}')
            temp_julu = jjll
            right_list_recoder[6]= np.array((x7,y7))
            # break
    temp_julu = MatchXY
    for j in range(len(zXY_list_8)):
        x8 = zXY_list_8[j][0]
        y8 = zXY_list_8[j][1]
        jjll = juli(x1,y1,x8,y8)
        if jjll<temp_julu :   #计算三维距离
            print(f'8-匹配roi:{x8},{y8},juli={jjll}')
            temp_julu = jjll
            right_list_recoder[7]= np.array((x8,y8))
            # break
    return right_list_recoder

def juli(x1,y1,x2,y2):
    return np.sqrt( (x1-x2)**2+(y1-y2)**2 )


#8888
def get_right_list3d(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,right_list,project,size=30,Debug=0):
    #根据right_list 2D坐标，以及对应的值，在3D上找到Z轴的位置
    #(ROI_n,8,3)
    ROI_n = right_list.shape[0]
    right_list3dd = np.zeros((ROI_n,8,3),dtype=np.int16)
    print(project.shape)
    for j in range(ROI_n):
        for k in range(8):
            x_index = right_list[j][k][0]
            y_index = right_list[j][k][1]
            value_ = project[k,x_index,y_index]
            if   k == 0:
                z_index = np.where(imag_1[:,x_index,y_index]==value_)[0][0]
            elif k == 1:
                z_index = np.where(imag_2[:,x_index,y_index]==value_)[0][0]
            elif k == 2:
                z_index = np.where(imag_3[:,x_index,y_index]==value_)[0][0]
            elif k == 3:
                z_index = np.where(imag_4[:,x_index,y_index]==value_)[0][0]
            elif k == 4:
                z_index = np.where(imag_5[:,x_index,y_index]==value_)[0][0]
            elif k == 5:
                z_index = np.where(imag_6[:,x_index,y_index]==value_)[0][0]
            elif k == 6:
                z_index = np.where(imag_7[:,x_index,y_index]==value_)[0][0]
            elif k == 7:
                z_index = np.where(imag_8[:,x_index,y_index]==value_)[0][0]
            right_list3dd[j][k] = (z_index,x_index,y_index) 
    
    
    right_list2 = np.copy(right_list3dd)
    ROIS_1 = []
    ROIS_2 = []
    ROIS_3 = []
    ROIS_4 = []
    ROIS_5 = []
    ROIS_6 = []
    ROIS_7 = []
    ROIS_8 = []
    
    i=0
    print("8 保存ROI区域")
    while i < right_list2.shape[0]:
        z_ori_1 = int(right_list2[i,0,0])
        x_ori_1 = int(right_list2[i,0,1])
        y_ori_1 = int(right_list2[i,0,2])
        
        z_ori_2 = int(right_list2[i,1,0])
        x_ori_2 = int(right_list2[i,1,1])
        y_ori_2 = int(right_list2[i,1,2])
        
        z_ori_3 = int(right_list2[i,2,0])
        x_ori_3 = int(right_list2[i,2,1])
        y_ori_3 = int(right_list2[i,2,2])
        
        z_ori_4 = int(right_list2[i,3,0])
        x_ori_4 = int(right_list2[i,3,1])
        y_ori_4 = int(right_list2[i,3,2])
        
        z_ori_5 = int(right_list2[i,4,0])
        x_ori_5 = int(right_list2[i,4,1])
        y_ori_5 = int(right_list2[i,4,2])
        
        z_ori_6 = int(right_list2[i,5,0])
        x_ori_6 = int(right_list2[i,5,1])
        y_ori_6 = int(right_list2[i,5,2])
        
        z_ori_7 = int(right_list2[i,6,0])
        x_ori_7 = int(right_list2[i,6,1])
        y_ori_7 = int(right_list2[i,6,2])
        
        z_ori_8 = int(right_list2[i,7,0])
        x_ori_8 = int(right_list2[i,7,1])
        y_ori_8 = int(right_list2[i,7,2])
     
        #ROI
        print(f"{i}:峰值整数位置",z_ori_1,z_ori_2,z_ori_3,z_ori_4,z_ori_5,z_ori_6,z_ori_7,z_ori_8)
        
        ROI_1 = imag_1[ : ,x_ori_1-size//2:x_ori_1-size//2+size,y_ori_1-size//2:y_ori_1-size//2+size]
        ROI_2 = imag_2[ : ,x_ori_2-size//2:x_ori_2-size//2+size,y_ori_2-size//2:y_ori_2-size//2+size]
        ROI_3 = imag_3[ : ,x_ori_3-size//2:x_ori_3-size//2+size,y_ori_3-size//2:y_ori_3-size//2+size]
        ROI_4 = imag_4[ : ,x_ori_4-size//2:x_ori_4-size//2+size,y_ori_4-size//2:y_ori_4-size//2+size]
        ROI_5 = imag_5[ : ,x_ori_5-size//2:x_ori_5-size//2+size,y_ori_5-size//2:y_ori_5-size//2+size]
        ROI_6 = imag_6[ : ,x_ori_6-size//2:x_ori_6-size//2+size,y_ori_6-size//2:y_ori_6-size//2+size]
        ROI_7 = imag_7[ : ,x_ori_7-size//2:x_ori_7-size//2+size,y_ori_7-size//2:y_ori_7-size//2+size]
        ROI_8 = imag_8[ : ,x_ori_8-size//2:x_ori_8-size//2+size,y_ori_8-size//2:y_ori_8-size//2+size]

        ROIS_1.append(ROI_1)
        ROIS_2.append(ROI_2)
        ROIS_3.append(ROI_3)
        ROIS_4.append(ROI_4)
        ROIS_5.append(ROI_5)
        ROIS_6.append(ROI_6)
        ROIS_7.append(ROI_7)
        ROIS_8.append(ROI_8)
        if Debug==1:
            save_tif(f'ROI_{i}_1.tif',ROI_1)
            save_tif(f'ROI_{i}_2.tif',ROI_2)
            save_tif(f'ROI_{i}_3.tif',ROI_3)
            save_tif(f'ROI_{i}_4.tif',ROI_4)
            save_tif(f'ROI_{i}_5.tif',ROI_5)
            save_tif(f'ROI_{i}_6.tif',ROI_6)
            save_tif(f'ROI_{i}_7.tif',ROI_7)
            save_tif(f'ROI_{i}_8.tif',ROI_8)
        i+=1
    #这个ROIS 全部z轴保留，
    #仅仅在做高斯拟合时，用到z轴信息
    return ROIS_1,ROIS_2,ROIS_3,ROIS_4,ROIS_5,ROIS_6,ROIS_7,ROIS_8,right_list2 #right_list2这个里面保存有z轴信息 用于高斯拟合时使用

#101010
def get_2D_project_match(right_list,project):
    #right_list (roi,8,2)
    imag = np.copy(project)
    roi_n = right_list.shape[0]
    print('roi_n=',roi_n)
    a = 1
    for i in range(roi_n):
        for j in range(8):  #在8个project上画图
            x_ori = right_list[i][j][0]
            y_ori = right_list[i][j][1]
            imag[j,x_ori,y_ori] = a*i
            imag[j,x_ori+1,y_ori+1] = a*i
            imag[j,x_ori+1,y_ori] = a*i
            imag[j,x_ori-1,y_ori] = a*i
            imag[j,x_ori-1,y_ori-1] = a*i
            imag[j,x_ori,y_ori+1] = a*i
            imag[j,x_ori,y_ori-1] = a*i
            imag[j,x_ori+1,y_ori-1] = a*i
            imag[j,x_ori-1,y_ori+1] = a*i
    for i in range(8):
        save_tif(f'project_{i}.tif',imag[i])
    return 

#移动image
def match_image(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,coordinate):
    cor = (coordinate-512+1)
    for i in range(8):
        x_move = int(cor[i][0])
        y_move = int(cor[i][1])
        if i == 0:
            temp_imag = imag_1
        elif i == 1:
            temp_imag = imag_2
        elif i == 2:
            temp_imag = imag_3
        elif i == 3:
            temp_imag = imag_4
        elif i == 4:
            temp_imag = imag_5
        elif i == 5:
            temp_imag = imag_6
        elif i == 6:
            temp_imag = imag_7
        elif i == 7:
            temp_imag = imag_8
        #移动x
        temp_image2 = np.roll(temp_imag,x_move,axis=1)
        if x_move>=0:
            temp_image2[:,:x_move,:] = 0
        else:
            temp_image2[:,x_move:,:] = 0

        #移动y
        temp_image2 = np.roll(temp_image2,y_move,axis=2)
        if y_move>=0:
            temp_image2[:,:,:y_move] = 0
        else:
            temp_image2[:,:,y_move:] = 0

        if i == 0:
            image_1_M = temp_image2
        elif i == 1:
            image_2_M = temp_image2
        elif i == 2:
            image_3_M = temp_image2
        elif i == 3:
            image_4_M = temp_image2
        elif i == 4:
            image_5_M = temp_image2
        elif i == 5:
            image_6_M = temp_image2
        elif i == 6:
            image_7_M = temp_image2
        elif i == 7:
            image_8_M = temp_image2
    return image_1_M,image_2_M,image_3_M,image_4_M,image_5_M,image_6_M,image_7_M,image_8_M


def match_image2(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,coordinate):
    '''亚像素配准'''
    #转变为频域，乘上e指数
    imag_1_M = konyu_move(imag_1,coordinate[1])
    imag_2_M = konyu_move(imag_2,coordinate[1])
    imag_3_M = konyu_move(imag_3,coordinate[2])
    imag_4_M = konyu_move(imag_4,coordinate[3])
    imag_5_M = konyu_move(imag_5,coordinate[4])
    imag_6_M = konyu_move(imag_6,coordinate[5])
    imag_7_M = konyu_move(imag_7,coordinate[6])
    imag_8_M = konyu_move(imag_8,coordinate[7])
    return imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M

def konyu_move(data,corr):
    _,nx,ny = data.shape
    x = np.arange(np.ceil(-nx/2),np.ceil(nx/2))
    y = np.arange(np.ceil(-ny/2),np.ceil(ny/2))
    xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
    kx = -corr[0]
    ky = -corr[1]
    ysh =(np.exp(-2j*np.pi*((kx/nx)*xv+(ky/ny)*yv)))    #空域的移动需要乘负指数
    
    data_M = _U.tifftn(_U.tfftn(tt.from_numpy(data.copy())) * tt.from_numpy(ysh)).real
    data_M[data_M<0]=0
    return data_M.numpy().astype(np.float32)

def match_rotation(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,angles,scales):
    '''
    旋转度数
    '''
    imag_1_R = Rotation(imag_1,angles[0],scales[0])
    imag_2_R = Rotation(imag_2,angles[1],scales[1])
    imag_3_R = Rotation(imag_3,angles[2],scales[2])
    imag_4_R = Rotation(imag_4,angles[3],scales[3])
    imag_5_R = Rotation(imag_5,angles[4],scales[4])
    imag_6_R = Rotation(imag_6,angles[5],scales[5])
    imag_7_R = Rotation(imag_7,angles[6],scales[6])
    imag_8_R = Rotation(imag_8,angles[7],scales[7])
    return imag_1_R,imag_2_R,imag_3_R,imag_4_R,imag_5_R,imag_6_R,imag_7_R,imag_8_R

def Rotation(data,angle,scale):
    nz,nx,ny = data.shape
    # center =tuple(np.array(nx//2,ny//2))
    center = tuple(np.array(data.shape) / 2)[1:3]
    rot = cv2.getRotationMatrix2D(center, -angle, 1.0 + (1.0 - scale)) #调整完旋转后
    data_R = np.zeros_like(data)
    for i in range(nz):
        data_R[i] = cv2.warpAffine(data[i], rot, (ny, nx), flags=cv2.INTER_LANCZOS4)

    return data_R


#one setp： 将旋转和平移结合到一个仿射矩阵中
def one_setp(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,angles,scales,coordinate):
    nz = imag_2.shape[0]

    images = np.array([imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8])
    images_M = np.zeros_like(images)


    for i in range(len(images)):
        for j in range(nz):
            images_M[i,j] = one_setp_sub(images[i,j],angles[i],scales[i],coordinate[i])

    return images_M[0],images_M[1],images_M[2],images_M[3],images_M[4],images_M[5],images_M[6],images_M[7]

def one_setp0(imagesi,anglesi,scalesi,coordinatei):
    '拆分每个区域的移动'
    nz = imagesi.shape[0]
    # images = np.array([imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8])
    images_M = np.zeros_like(imagesi)

    for j in range(nz):
        images_M[j] = one_setp_sub(imagesi[j],anglesi,scalesi,coordinatei)

    return images_M


def one_setp45(imag_4,imag_5,angles,scales,coordinate):
    nz = imag_4.shape[0]

    images = np.array([imag_4,imag_5])
    images_M = np.zeros_like(images)

    for i in range(len(images)):
        for j in range(nz):
            images_M[i,j] = one_setp_sub(images[i,j],angles[i],scales[i],coordinate[i]) 

    return images_M[0],images_M[1]


def one_setp_sub(data,angles,scales,coordinate):
    """
    只能传入2D数据
    """
    rows,cols = data.shape
    cX,cY = rows//2,cols//2

    angle = -(angles)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0+(1.0-scales))
    

    M[0][2] = M[0][2]-coordinate[1]
    M[1][2] = M[1][2]-coordinate[0]
    re_dst = cv2.warpAffine(data,M,(cols,rows))
    return re_dst
###############SIFT###########
def one_setp_Hs(images,Hs):

    nz = images[0].shape[0]

    # images = np.array( [imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8])
    # del imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8
    images_M = np.zeros_like(images)


    for i in range(len(images)):
        for j in range(nz):   #2D数据传入
            images_M[i,j] = one_setp_sub_Hs(images[i,j],Hs[i])

    return images_M[0],images_M[1],images_M[2],images_M[3],images_M[4],images_M[5],images_M[6],images_M[7]

def one_setp_Hs0(imagesi,Hsi):
    '拆分每个区域的移动'
    nz = imagesi.shape[0]
    images_M = np.zeros_like(imagesi)
    for j in range(nz):   #2D数据传入
            images_M[j] = one_setp_sub_Hs(imagesi[j],Hsi)
    return images_M

def one_setp_Hs45(imag_4,imag_5,Hs):

    nz = imag_4.shape[0]

    images = np.array( [imag_4,imag_5])
    images_M = np.zeros_like(images)

    for i in range(len(images)):
        for j in range(nz):   #2D数据传入
            images_M[i,j] = one_setp_sub_Hs(images[i,j],Hs[i])

    return images_M[0],images_M[1]



def one_setp_sub_Hs(data,H):
    nx,ny = data.shape

    # imag_8_stack_new=[]

    imgOut_2D = cv2.warpPerspective(data, H, (ny,nx),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
    # imag_8_stack_new.append(np.array(imgOut_2D).reshape(nx,ny))

    # imag_8_stack_new = np.array(imag_8_stack_new)
    return imgOut_2D.reshape(nx,ny)
    #SIFT算法得到Hs矩阵后，直接作用在真正数据上




def caijian_image(imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,x_s,x_e,y_s,y_e):
    
    imag_1_M_C = imag_1_M[:,x_s:x_e,y_s:y_e]
    imag_2_M_C = imag_2_M[:,x_s:x_e,y_s:y_e]
    imag_3_M_C = imag_3_M[:,x_s:x_e,y_s:y_e]
    imag_4_M_C = imag_4_M[:,x_s:x_e,y_s:y_e]
    imag_5_M_C = imag_5_M[:,x_s:x_e,y_s:y_e]
    imag_6_M_C = imag_6_M[:,x_s:x_e,y_s:y_e]
    imag_7_M_C = imag_7_M[:,x_s:x_e,y_s:y_e]
    imag_8_M_C = imag_8_M[:,x_s:x_e,y_s:y_e]

    return imag_1_M_C,imag_2_M_C,imag_3_M_C,imag_4_M_C,imag_5_M_C,imag_6_M_C,imag_7_M_C,imag_8_M_C



def matcher(image1,image2,keypoints1,keypoints2,descriptors1,descriptors2,matcher,descriptor,ratio_thresh=1):
    if (descriptor == 'SIFT') or (descriptor == 'SURF') or (descriptor == 'KAZE'):
        normType = cv2.NORM_L2
    else:
        normType = cv2.NORM_HAMMING
            
    if matcher == 'BF':
        # Se descritor for um Descritor de Recursos Locais utilizar NOME


        BFMatcher = cv2.BFMatcher()

        matches = BFMatcher.knnMatch(descriptors1,descriptors2,k=2)
        
        ratio_thresh = ratio_thresh #0.5#0.8#1
        # "Good" matches
        good_matches = []
        # Filter matches
        for m, nn in matches:
            if m.distance < ratio_thresh * nn.distance:
                good_matches.append([m])
        

        output = cv2.drawMatchesKnn(img1 = image1,
                                        keypoints1 = keypoints1,
                                        img2 = image2,
                                        keypoints2 = keypoints2,
                                        matches1to2 = good_matches,
                                        outImg = None,
                                        flags = 2)
        # plt.figure()
        # plt.imshow(output)
        return output,good_matches

    if matcher == 'FLANN':
        # 设置FLANN匹配器参数，定义FLANN匹配器，使用 KNN 算法实现匹配
        indexParams = dict(algorithm=0, trees=5)
        searchParams = dict(checks=50)

        flann = cv2.FlannBasedMatcher(indexParams,searchParams)
        matches = flann.knnMatch(descriptors1,descriptors2,k=2)

        # # 根据matches生成相同长度的matchesMask列表，列表元素为[0,0]
        # matchesMask = [[0,0] for i in range(len(matches))]

        # # 去除错误匹配
        # for i,(m,n) in enumerate(matches):
        #     if m.distance < 0.7*n.distance:
        #         matchesMask[i] = [1,0]
                
        
        ratio_thresh = ratio_thresh #0.5#0.8#1
        # "Good" matches
        good_matches = []
        # Filter matches
        for m, nn in matches:
            if m.distance < ratio_thresh * nn.distance:
                good_matches.append([m])
        

        output = cv2.drawMatchesKnn(img1 = image1,
                                        keypoints1 = keypoints1,
                                        img2 = image2,
                                        keypoints2 = keypoints2,
                                        matches1to2 = good_matches,
                                        outImg = None,
                                        flags = 2)
        return output,good_matches

def filtermatcher(matches,kp_base,kp_tager,nx=430,slope_threshold=0.5,length_threshold=10):
    # import sklearn
    from sklearn.cluster import KMeans
    # 提取匹配的关键点
    keypoints1 = [kp_base[m[0].queryIdx].pt for m in matches]
    keypoints2 = [kp_tager[m[0].trainIdx].pt for m in matches]
    

    print(len(keypoints1))
    # 计算线段的斜率和长度
    slopes = []
    lengths = []
    for pt1, pt2 in zip(keypoints1, keypoints2):
        x1, y1 = pt1
        x2, y2 = pt2
        x2 = x2+nx
        # print(f'{x1:=},{y1:=}')
        # print(f'{x2:=},{y2:=}')
        # print()
        # 避免除数为零的情况
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        lengths.append(length)

    # print(slopes)
    # print(lengths)
    
    # 合并斜率和长度为一个矩阵
    features = np.vstack((slopes, lengths)).T
    
    # 使用K均值聚类
    num_clusters = 3  # 设置聚类数，根据实际情况调整
    kmeans = KMeans(n_clusters=num_clusters, random_state=0,n_init='auto').fit(features)

    # 标记每个数据点的聚类结果
    labels = kmeans.labels_

    # 找到数量最多的类别
    most_common_label = np.argmax(np.bincount(labels))

    # 找到属于最多类别的数据点
    most_common_points = features[labels == most_common_label]

    # 计算最多类别中的中位数
    median_slope = np.median(most_common_points[:, 0])
    median_length = np.median(most_common_points[:, 1])


    # 计算斜率和长度的中位数
    # median_slope = np.median(slopes)
    # median_length = np.median(lengths)
    print(f"median = {median_slope},{median_length}")
    
    # 去除差异过大的线段
    filtered_matches = []
    for m, pt1, pt2 in zip(matches, keypoints1, keypoints2):
        x1, y1 = pt1
        x2, y2 = pt2
        x2 = x2+nx
        # 避免除数为零的情况
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # 判断斜率和长度是否在阈值范围内
            if abs(slope - median_slope) <= slope_threshold and np.abs( length - median_length) <= length_threshold:
                filtered_matches.append(m)

    # 输出结果
    print(f"Original number of matches: {len(matches)}")
    print(f"Filtered number of matches: {len(filtered_matches)}")

    return filtered_matches
######################################################
#公共函数



#提取OTF






#计算配准







#配准样本
