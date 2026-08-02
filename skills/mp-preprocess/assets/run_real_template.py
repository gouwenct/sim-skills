import Data_PreProcessing_MP
import Data_Registration_MP
import os
import numpy as np
import tifffile
from pathlib import Path
import Utils as _U
#V1.0 图像指定S E读入,减BG,SIFT配准,区域分光校正,时间点光漂白校正
#V1.5 预处理变为相对路径；预处理变为/4 以应对200 长方形的情况；
#V1.8 更智能分割长时程数据，引入points
#v1.9 去掉漂白校正、BF做高低频谱分离
#V2.0 增加连续读取文件能力，文件的命名规范情况下，可以连续获取raw data 数据
#V2.1 #控制时间点，points_name , points,  point 、BF的psf更新
#V2.2 保存长时间点的WF,16的间隔存储
#V2.3 图像裁剪 规定为左右 上下各裁剪50
#V2.4 增加5+5模式中-提取52模式  zhenshu_sep=15,point>=2, zhenshu=21, func_f=3015
#V2.41 修改WF 16 的保存，    #####增加30nm层间距的WF，另外编写文件处理
#V2.5 增加para保存，WF的不同生成模式
#V2.51 增加488 561 分别保存在01-488,01-561文件夹，需设置其实波长
#V2.52 由于不同通道目前存在第7区域 561通道骤降问题，将数据做一下处理，12，56，910.等488 求平均计算分光系数
#      作用在488通道上；561的也作一次；  然后在处理一个时间点时，做分方向校正
#V2.6  直接读取rawdata（准备有问题）
#V2.7  修正后，直接读取raw data（配准没问题），修正分光矫正问题

def Hs_load(input_path,Registration,FANWEI_peizhun):
    if Registration==3:
        # loop_dir = Path(str(input_path)).iterdir()
        Hs_set = []
        for i in FANWEI_peizhun:
            peizhun_file = i
            # print(f"Hs_load:{peizhun_file}")
            loop_dir = Path(str(input_path)).iterdir()  #每次遍历 重新加载遍历文件 夹
            for ldr_j in loop_dir:
                dir_name = ldr_j.name
                suffix = ldr_j.suffix
                # print("Hs_load-dir_name",dir_name)
                # print("Hs_load-suffix",suffix)
                if ldr_j.is_dir() == 1:                           # 跳过文件夹
                    print("【Hs_load】跳过0：", dir_name)
                    continue
                elif str(peizhun_file) in  dir_name and suffix == ".npy":      #找到名字中含有
                    Hs = np.load(ldr_j)
                    print(f'加载{dir_name}')
                    # return Hs
                    Hs_set.append(Hs)
        return Hs_set


def get_tif_xyz_size(file_path):
    with tifffile.TiffFile(file_path) as tif:
    # 假设图像是多页的（三维），获取第一页以获得尺寸信息
    # 这里我们获取第一个图像系列的形状，多数情况下对应于整个文件
        if tif.pages:
            # 获取第一页的尺寸信息
            page = tif.pages[0]
            x_size, y_size = page.shape

            # 对于深度，我们计算文件中的页数
            z_size = len(tif.pages)
            # print(f'x_size: {x_size}, y_size: {y_size}, Depth: {z_size}')
        else:
            print("No pages found in TIF file.")

    return z_size,x_size, y_size



def count_elements(lst):
    count = 0
    for element in lst:
        if isinstance(element, list):
            # 如果元素是列表，则递归计数
            count += count_elements(element)
        else:
            # 如果元素不是列表，直接计数
            count += 1
    return count

#统计每个信号的nz为多少
def tongji(path):
    import pickle
    #先找pkl 文件，没有再重新加载数据统计
    p1 = Path(str(path)).iterdir()
    ##############################优先找含有pkl的文件#######
    for ldr_j in p1:
        dir_name = ldr_j.name
        suffix = ldr_j.suffix
        # print(dir_name)
        # print(suffix)
        if ldr_j.is_dir() == 1:                           # 跳过文件夹
            print("【查找PKL文件】跳过文件夹：", dir_name)
            continue
        elif 'nz_set' in  dir_name and suffix == ".pkl":      #找到名字中含有
            with open(ldr_j, 'rb') as f:
                nz_set = pickle.load(f)
            print(f'加载{dir_name}')
            return nz_set
    ########################################
    i=0
    dirct = {}
    for p1_i in p1:   #遍历不同数据01/02/03
        if p1_i.is_file():  # 忽略文件
            continue
        #如果找到list 数据就跳过
        p1_i_name = p1_i.name
        if "MP" not in p1_i_name:
            continue
        print(f'{p1_i_name}加载完成')
        dirct_2= {}
        i+=1
        p2 = Path(str(p1_i)).iterdir()   #遍历01文件夹下面的cam1 cam。 cam2，cam。

        for p2_i in p2:
            if not p2_i.is_file():  # 忽略文件夹
                continue
            p2_i_name = p2_i.name

            #访问IFDs
            #打开TIF文件以访问其IFDs
            # print(p2_i_name)
            z_size,x_size, y_size = get_tif_xyz_size(p2_i)
            # print(f"Dimensions (ZXY): {z_size} x {x_size} x {y_size}  ")
            dirct_2[p2_i_name] = [ z_size,x_size, y_size ]
            # hist, bins = np.histogram(data1, bins=256, range=[min,max])
        # print()
        dirct[p1_i_name] = dirct_2

    #输出结果
    nz_set = []
    for outer_key, inner_dict in dirct.items():
        print(f"filename: {outer_key}")                     #输出文件title，eg. 03、MP_eb3_3
        # 确保内层也是字典，然后遍历
        sum_z = []
        if isinstance(inner_dict, dict):
            for inner_key, value in inner_dict.items():
                print(f"\tcam: {inner_key}, Value: {value}")  #输出cam: eg. cam1_eb3_3_MMStack_Default.ome.tif, Value: [2037, 512, 2048]
                sum_z.append(value[0])
        nz_set.append(sum_z)
        sum_z = np.array(sum_z)
                                                              #统计 cam1 =  3000   point： 200.0
        print('\tcam1 = ',np.sum(sum_z[:len(sum_z)//2 ]), "  point：",np.sum(sum_z[:len(sum_z)//2 ])/15 )
        print('\tcam2 = ',np.sum(sum_z[len(sum_z)//2: ]), "  point：",np.sum(sum_z[len(sum_z)//2: ])/15)

    #保存sum_z数据
    date = Path(path).name
    with open(path+"\\99、nz_set-"+date+'.pkl', 'wb') as f:
        pickle.dump(nz_set, f)

    print(f"成功保存nz_set文件")
    return nz_set


#【数据移动】
path_input = os.environ["MP_SIM_RAW_ROOT"]  # mark
path_output = path_input
Data_PreProcessing_MP.main(path_input,path_output)
print('############数据移动完成#############')

#统计文件帧数   [2037, 963, 2037, 963]
# nz_set = tongji(path_input)

#【数据配准】
print('\n\
        #################开始计算Rough——Affine配准#################')#找到MP_peizhun文件 就计算，如果有多个文件 只会计算第一个文件夹内的内容
para = dict(
    data_name='',
    dir_name='',
    region_n=8,  # 需要处理的区域  MP为8 2d为1
    stand = 4,
    version  = 2.7, #版本控制
    AVGG=[1],  # 需要平均的帧数IE
    BF_kongzhi =1,   #BF控

    #15 与9
    zhenshu_sep=15,#15,  # 拍图时的相位（读入时以它为基础）               #~~~~~~~~~2D or MP 修改处~~~~~~~~~~~~~
    zhenshu=15,#15,#21,  # 需要提取的帧数                 #~~~~~~~~~2D or MP 修改处~~~~~~~~~~~~~

    #控制时间点
    points_name = 1,#控制nth_total 从1开始
    points =1,  #控制这个文件nth开始位点 从1开始 #以15为单位
    point=10,  #132,#200,#134,  ## 2037-134   从开头取！！！    #MP55 一定>2个时间点
               # dual >4      # (1,2),(3,4)
               #时间点（15为单位）
    volume = 300,#150,#3610,#2037,#2037,#3610, #帧数
    # W_order = [488,561], #points从1开始
    # W_order = [561,488], #points从3开始
    W_order = [488,488], #单色 #tag

    peizhun_file = "peizhun",   #mark
    FANWEI = [1], #ta8g    #mark  # 01、mp1（sorted 1-based under 731）
    # 0=只写 A(亮度矫正+配准→01-488)；1=额外写 B(仅配准→01-488_regOnly) 供 SIMcheck；recon 只用 A
    save_reg_only = 0,  #mark
    errorpoint = [],  #从1开始计，默认为[] 空 #tag

    # FANWEI = (np.array(range(1,11))).tolist(),
    subname = '', #0-1980这种情况
    # subname = '',

    x34_just = 0,#0.4,#0.4 ,   #左+   右-
    y34_just = -0,#-0.6,#-0.3   ,   #上+   下-

    x_just = -0.0,#0.4,#0.4 ,   #左+   右-
    y_just = -0.0,#-0.6,#-0.3   ,   #上+   下-

    # cell = 'FIX',
)

func_f = 123#3015#123#123#3015#123#212152#123#212152#123#212152#123#212152 #2121

# if para['point']<=40:
#     para['W_order'][0] = str(para['W_order'][0])+"-01"
#     para['W_order'][1] = str(para['W_order'][1])+"-01"

# FANWEI = (np.array(range(6,11))).tolist() #跑多个文件
Registration = 3   #3表示SIFT+直接移动   1表示互相关+45微调
longlong = 0   #长时间点 ，切割完cam后 一起跑

# import time
# time.sleep(3600*2)

if Registration==3:
    if longlong == 0:
        Hs = Data_Registration_MP.Rough_Affine(path_output,para,Registration = Registration)
        # Hs = np.load(r'F:\code-F\MP-SIM\real\dataOut\1.4\07、MP_er_5+2\Hs_base-base.npy') #直接加载

        for i in range(Hs.shape[0]):
            if i <3:
                Hs[i,0,2] = Hs[i,0,2] + para['x34_just']#(0.48) #水平 + 左
                Hs[i,1,2] = Hs[i,1,2] + para['y34_just']#(-1)#上下+ 上

            if i <4:
                Hs[i,0,2] = Hs[i,0,2] + para['x_just']#(0.48) #水平 + 左
                Hs[i,1,2] = Hs[i,1,2] + para['y_just']#(-1)#上下+ 上

        print('###################Rough_Affine 计算完成#####################')
        print(Hs.round(2))

        #调节初相差
        print('\n\
                #################开始应用配准#################')
        Data_Registration_MP.Fine_Affine_3(path_output,Hs,para,Registration = 3,func_f=func_f)  #计算45之前的差异
        print(Hs.round(2))
