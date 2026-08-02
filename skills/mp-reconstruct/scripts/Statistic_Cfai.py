import numpy as np
import Utils as _U
from tifffile import imread

def get_cfai(savepath0,num_js=3,Dr_sub=''):
    #遍历cfai
    from pathlib import Path
    import matplotlib.image as mpig
    from skimage import color
    '''
    思路：
    1、能遍历不同文件夹 A/B/C
    2、A-cfai里 先确定子文件数量（总共数量/除阶数/3=子区域数量）
    3、子文件数量、阶数确定后，三个for 循环（区域、方向、阶数）
    4、列数列组合，方向行组合，区域Z组合--保存文件（cfai父目录）
    5、当A/B/C所有文件组合完毕后，也输出保存文件------cfai父目录的父目录
    '''
    
    loop_dir = Path(str(savepath0)).iterdir() #用于遍历某文件夹
    All_file_cfai = [] #全部文件的cfai
    
    for ldr_i in loop_dir:  #开始遍历不同数据A/B/C
        
        if ldr_i.is_dir() != 1:  #不是文件夹就跳过
            print('【Loop-FILE】跳过非文件夹',ldr_i)
            continue
            
        #进入cfai文件夹
        cfainame = ldr_i.name
        ldr_i_cfai = ldr_i/'cfai'
        path_FF = str(Path(str(ldr_i_cfai)).parent.parent)#cfai的父父目录保存5
        path_F = str(Path(str(ldr_i_cfai)).parent)#cfai的父目录-保存4
        
        p1 = Path(str(ldr_i_cfai)).iterdir() #仅遍历当前目录下的cfai图
        
        data_raw=[]
        ' 2、A-cfai里 先确定子文件数量（总共数量/除阶数/3=子区域数量）'+'读入数据'
        for p1_ii in p1:     #遍历cfai文件
            if not p1_ii.is_file():           #忽略文件夹
                continue
            '读入数据，变为灰度'     #读入数据
            data = mpig.imread(p1_ii)
            data = color.rgb2gray(color.colorconv.rgba2rgb(data))[np.newaxis,:,:]
            data_raw.append(data)
        
        data_raw = np.array(data_raw)  #(12,nx,ny) 1-2(1) 1-2(2) 1-1(1) 1-1(2) ;10_2-2 10_2-1;10_3-2 10_3-1 
        data_raw = _U.image_reshape(data_raw,dim=3)
        num_all,nx,ny = data_raw.shape
        num_sub = num_all/num_js/3
        if num_all % (num_js*3)!=0:
            print(f'共{num_all}cfai图，每方向有{num_js}阶,有{num_sub}子数据')
            raise(fr'共{num_all}cfai图，每方向有{num_js}阶,有{num_sub}子数据')
        else:
            print(f'共{num_all}cfai图，每方向有{num_js}阶,有{num_sub}子数据')
            num_sub = int(num_sub)
            
        
        '3、子文件数量、阶数确定后，三个for 循环（子区域、方向、阶数）'  #读入文件顺序为子区域1,2,3,4  、其他阶1234,。。。其他方向 
        #(12,nx,ny) - >(num_sub,nx*num_js,3*ny)
        one_file_cfai = np.zeros((int(num_sub),int(nx*num_js),int(3*ny)))
        
        for sub_i in range(num_sub):   #z
            for num_js_i in range(num_js): #x
                for ori_i in range(3): #y
                    one_file_cfai[sub_i,num_js_i*nx:(num_js_i+1)*nx,ori_i*ny:(ori_i+1)*ny] = data_raw[(ori_i*num_sub*num_js)+(num_js_i*num_sub)+sub_i]
        '4、列数列组合，方向行组合，区域Z组合--保存文件（cfai父目录）'
        _U.save_tif_self((path_F)+f'\\参数统计_{cfainame}-{Dr_sub}.tif',one_file_cfai) 
        
        
        All_file_cfai.append(one_file_cfai)
    '5、当A/B/C所有文件组合完毕后，也输出保存文件------cfai父目录的父目录'
    
    All_file_cfai = np.array(All_file_cfai)
    _U.save_tif_self((path_FF)+f'\\参数统计_all-{Dr_sub}.tif',All_file_cfai) 
    
    return 0 


# path = r'F:\code-F\MP-SIM\reconV3.2版本\MPV3.2版本末-即将改3.3\dataOut\MP-SIM'
# get_cfai(path,num_js=3)




def get_P(savepath0,Dr_sub=''):
    #遍历cfai
    from pathlib import Path
    import matplotlib.image as mpig
    from skimage import color
    '''
    思路：
    1、能遍历不同文件夹 A/B/C
    2、A-P里 读取所有P文件，提取第二帧max出来，保存one_Ps,----P父目录
    3、当A/B/C所有文件组合完毕后，也输出保存文件--All_file_Ps----P父目录的父目录
    '''
    
    loop_dir = Path(str(savepath0)).iterdir() #用于遍历某文件夹
    All_file_Ps = [] #全部文件的P
    
    for ldr_i in loop_dir:  #开始遍历不同数据A/B/C
        if ldr_i.is_dir() != 1:  #不是文件夹就跳过
            print('【Loop-FILE】跳过非文件夹',ldr_i)
            continue
            
        #进入P文件夹
        cfainame = ldr_i.name
        ldr_i_P = ldr_i/'P'
        path_FF = str(Path(str(ldr_i_P)).parent.parent)#P的父父目录保存5
        path_F = str(Path(str(ldr_i_P)).parent)#P的父目录-保存4
        
        p1 = Path(str(ldr_i_P)).iterdir() #仅遍历当前目录下的P图
        
        #读取这个文件的全部P文件
        one_data_all = []    #保存一个文件的所有P文件
        one_file_Ps=[]       
        for p1_ii in p1:     #遍历P文件，
            if not p1_ii.is_file():           #忽略文件夹
                continue
            '读入数据，变为灰度'      # 读入数据
            data = imread(p1_ii)     #（5,nx,ny）
            one_data_all.append(data)  # (1) (2)
        
        one_data_all = np.array(one_data_all)        #(2,5,nx,ny)
        _,nz_one,_,_ = one_data_all.shape              #每个文件的nz层数
        one_data_all = _U.image_reshape(one_data_all,dim=3) #(10,nx,ny)
        nz,_,_ = one_data_all.shape

        #提取
        one_data_2P = []   #提取一个文件所有P文件的max层
        for z_i in range(nz//nz_one):
            one_data_2P.append(one_data_all[z_i*nz_one+1])
        #保存A文件的P
        one_data_2P = np.array(one_data_2P)
        _U.save_tif_self((path_F)+f'\\参数P统计_{cfainame}-{Dr_sub}.tif',one_data_2P) 

        
        #保存A,B,C
        All_file_Ps.append(one_data_2P)
        
    '5、当A/B/C所有文件组合完毕后，也输出保存文件------cfai父目录的父目录'
    All_file_Ps = np.array(All_file_Ps)
    _U.save_tif_self((path_FF)+f'\\参数P统计_all-{Dr_sub}.tif',All_file_Ps) 
    
    return 0 
