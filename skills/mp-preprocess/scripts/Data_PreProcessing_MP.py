'''
此文件用于调整多层数据文件夹
将cam1、cma2的文件移动到一起
'''

from pathlib import Path
# from tifffile import imread
import shutil

def Make_Dir(FileName, file_dir,k):
    File1 = FileName / file_dir
    if not File1.exists():
        File1.mkdir()
        print(f" maker new folder :{File1}")
        k+=1
    return File1,k


def main(path_input,path_output):
    '主函数'
    
    #input 数据路径
    # path_input = r'E:\4TB图片\python 实验数据\仪器配准实验数据\2023\10.20'           #【每次新拍的数据要更改】
    p1 = Path(str(path_input)).iterdir()  # 仅遍历当前目录下内容
    
    #output 目标路径
    # path_output = r'E:\4TB图片\python 实验数据\仪器配准实验数据\2023\10.20'#path_input#r'E:\4TB图片\python 实验数据\仪器配准实验数据\2023\10.16'
    p_o = Path(str(path_output))
    
    
    data_name = Path(str(path_input)).name # 提取当日数据日期（用于区分不同日期）
    
    # #传入bg文件[先没有这一步]
    # bg = imread(r'E:\BaiduSyncdisk\My_code-7\AVG_cam1_background.tif')     
    # bg = bg.astype(np.float32)
    
    #遍历文件夹
    i= 0    #挑选数据
    warning = 0
    j = 0       #cam1 cam2中tif的数量
    k = 0   #新建文件夹数量
    for p1_i in p1:   #遍历不同数据
        i+=1
        
        # if i not in range(1,60):   # 挑选数据_________________________________________________________
        # if i not in [7,8,28,29]:   # 挑选数据
        #     print('【Loop-FILE】跳过',p1_i)
        #     continue
            
        if p1_i.is_dir() != 1:  #不是文件夹就跳过
            print('【Loop-FILE】跳过',p1_i)
            continue
        
        dir_name =  p1_i.name      #只要cam1 cam2
        if dir_name[:4] != 'cam1' and dir_name[:4] != 'cam2':  
            warning +=1
            print(f'跳过非3D-SIM数据,{dir_name},warning = {warning}')
            continue
        

        #在目标路径下建立 对应的文件夹
        dir_name_new = "MP_"+dir_name[5:]
        p_o_dir,k=Make_Dir(p_o,dir_name_new,k)
        
        
        #遍历.tif .txt 文件
        p2 = Path(str(p1_i)).iterdir()   #遍历一个文件夹下每个文件（应该有四个文件）
        for p2_i in p2:
            if not p2_i.is_file():  # 忽略文件夹
                continue
                
            path2=str(p2_i)
            file_suffix = Path(path2).suffix #具体文件后缀
            # file_stem = Path(path2).stem #具体文件主干名
            
            if file_suffix == '.tif':      #只要.tif文件
                #复制文件
                # shutil.copy(path2 , str(p_o_dir/Path(path2).name))
                shutil.move(path2 , str(p_o_dir/Path(path2).name))

                j+=1

    
                
                #【用于筛选数据】
                #SIMcheck里面看看
    print('\n##############################')
    print(f'成功完成数据移动操作，移动{j}个tif文件---->{k}个新文件夹中')
