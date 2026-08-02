import numpy as np
from tifffile import imread
import Utils as _U
from pathlib import Path

# #给叠加好的stack tif文件，平均
# data = imread(r'G:\4TB-G\24.3.14\MP-raw data\time lapse\12、MP_er_1-p2\Dr\2.tif')

# print(data.shape)
# data = data.reshape(11,120,210,430)

# data_aver = np.mean(data,axis=0)

# _U.save_tif(r'data_avger2.tif',data_aver)


#给一个文件夹，自动读取文件（间隔奇数，偶数）构成stack，平均
#MP55模式数据



def OneChannal55AVG(path,target_numer=10):
    "一个通道内控制奇数偶数"
    "遍历文件夹"
    p1 = Path(str(path)).iterdir() 
    data_odd = []
    data_even = []  
    # target_numer = 10  #控制平均数量
    numer=0
    for p1_i in p1:
        if not p1_i.is_file():           #忽略文件夹
            continue
        
        #跳过异常数据
        numer+=1

        file_name = str(Path(p1_i).name[-7:-4]) 
        data = imread(p1_i)
        print(file_name)
        if int(file_name)%2!=1:          #偶数据
            print("加入到偶数")
            data_even.append(data)
            
        elif int(file_name)%2==1:        #奇数据
            print("加入到奇数")
            data_odd.append(data)
        
                                #统计目前加了多少文件
        if numer == 2*target_numer:
            break
        
    "平均文件"
    data_even = np.array(data_even)
    data_odd = np.array(data_odd)
    # print(data_even.shape)
    
    data_even_aver = np.mean(data_even,axis=0)
    data_odd_aver = np.mean(data_odd,axis=0)
    return data_odd_aver,data_even_aver


"读取文件夹"

def main(path, target_numer=10):
    dir_name = str(Path(path).name)

    data_odd_aver,data_even_aver = OneChannal55AVG(path,target_numer)
    print("dir_name:",dir_name)

    _U.save_tif_self2((Path(path).parent),sub_filename=dir_name+'-AVG',filename=f'1、data-{dir_name}-aver-135.tif',data=data_odd_aver)
    _U.save_tif_self2((Path(path).parent),sub_filename=dir_name+'-AVG',filename=f'2、data-{dir_name}-aver-246.tif',data=data_even_aver)

# path = r'D:\GW\code\3D-SIM\3D-real\dataOut\F3-20240523(3)-rab\01、C3DSIM_Rab(1)\Dr'
# main(path,10)


# FileName = _U.save_path("", para)
# _AD.main(FileName/'Dr'/'01-488')
# _AD.main(FileName/'Dr'/'01-561')