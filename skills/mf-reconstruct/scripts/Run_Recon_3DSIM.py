import importlib
from pathlib import Path
# from loguru import logger
import argparse
import numpy as np

import Utils as _U
import Main_recon_3DSIM
import Statistic_Cfai as _SCfai
from tifffile import imread

# V3.2 在这个基础上增加文件夹迭代 与MPSIM重建程序一致
# V3.3 增加notch 遍历（只在FZ添加notch） ，notch命名方式改为 {0阶1阶旁，2阶-1阶}--还没同步到MPSIM

#v3.4 增加空域重建方法
#v3.5 发现APO切趾有问题
#V3.51 OTF移动增加real   取real时会将不对称的OTF变为对称的0.5，无real时 会保留一侧的mask
#V3.6 在XC版本中，给频域是实数的变量 逆变换时都增加real（特别是用到GPU的地方），否则精度会有问题
#v3.7 padding 固定为2，重建用XC方式，去掉real   ---这个版本首尾和中间伪影都挺少
#V3.8 加入光漂白校正，4B的0阶notch
#V3.9  修改一个2D版本 2024-02-04
#V4.0  24.3.24 仿真数据
#V4.1  OTF插值条件改变（zpadding的 和 本来不匹配的）-增加【Run_3D】文件
#V4.12-4.14 更改文件命名方式,缩减名字
#V4.15 加载OTF 在层间距相同情况下，自己转变到PSF裁剪为相同深度

def save_parameter(opt,run_parameter):
    #保存运行参数（无需for循环）
    from datetime import datetime
    now = datetime.now().strftime("%H-%M-%S")
    
    opt_dic = vars(opt)
    merged_dic = opt_dic|run_parameter
    # print(merged_dic)
    txt_path = _U.save_path(f'opt_run_parameter-{now}.txt',run_parameter)
    _U.save_txt(txt_path,merged_dic)
    
    return 0

def main(opt,run_parameter,savepath0,OTFone,FANWEI):
    # 传入三个文件路径 Dr，OTF，parameter
    # Dr目录下会自动遍历所有文件，需要与OTF下文件数一致
    isdebug = [0, 0, 1, 1, 1, 0, 0, 0]
    """
    #0,   ; 1 加载数据 Dr；   #2 分离分量 Ck ；   3 其他数据处理 ;4 参数估计
            5 GetMsum
    """

    #######################################【V2.51遍历文件夹】
    opt.version ='V4.16'
    run_parameter['ver']=opt.version
    
    if run_parameter['mear']==1:  
        opt.Apoz=600
        opt.fcxs = 1.7     #制作Apo时的横向长度
        opt.plongxs = 1.7  #制作Apo时的横向长度
        run_parameter['note'] = 'ma60' 
    elif run_parameter['mear']==2:
        opt.Apoz=830
        opt.fcxs = 1.2     #制作Apo时的横向长度
        opt.plongxs = 1.2 
        run_parameter['note'] = 'Cell'
        
    else:
        opt.Apoz=750
        opt.fcxs = 1.2     #制作Apo时的横向长度
        opt.plongxs = 1.2 
        run_parameter['note'] = 'simul'
    
    loop_dir = Path(str(savepath0)).iterdir()  # 用于遍历某个文件夹
    
    # loop_dir = (Path(__file__).parent / savepath0.lstrip('\\')).resolve().iterdir()
    
    data_name = str(Path(str(savepath0)).name)  # 提取当日数据名称，eg 7.1 MP-SIM超多数据
    run_parameter["data_name"] = data_name
    print(data_name)

    i = 0  # 从1开始数
    for ldr_i in loop_dir:
        i += 1
        
        if i not in FANWEI:  # 不在这里面就跳过 ~~~~~~~~~~~指定文件夹~~~~~~~~~~~~~~~
            print(f"【Loop-FILE】跳过1,i={i}", ldr_i)

            continue

        if ldr_i.is_dir() != 1:  # 不是文件夹就跳过
            print("【Loop-FILE】跳过2", ldr_i)
            continue
                #
        savepath1 = ldr_i/'Dr/'/run_parameter['Dr_sub']
        # savepath2 = (ldr_i / "OTF\\")                            ###########~~~~OTF选择~~~~~~~~~
        savepath2 = OTFone        #
        savepath3 = ldr_i / "parameter\\"

        print("【Loop-DIR】处理", ldr_i)
        # dir_name = ldr_i.stem  # 获取文件夹名
        dir_name = ldr_i.parts[-1]  # 获取文件夹名
        run_parameter["dir_name"] = dir_name
        run_parameter["log"] = f"./dataOut/{data_name}/{dir_name}-{opt.version}/【{dir_name}】.log"
        # _U.LOG(run_parameter)
        # logger.info(f"#############RUN START【{dir_name}】############")

        # logger.info(f"savepath1=-【{savepath1}】")
        # logger.info(f"savepath2=-【{savepath2}】")
        # logger.info(f"savepath3=-【{savepath3}】")
        save_parameter(opt,run_parameter) 

        # 完整
        if run_parameter["just_cfai"] == 0:
                # opt,cfai,PGxy,PGz,plong,finalr,finalk,same_region_fai,same_region_pxy,Msum,Msum_mask = Main_recon_3DSIM.main(
                # opt, savepath1, savepath2, savepath3, run_parameter, isdebug=isdebug            )
                opt,cfai,PGxy,PGz,plong,finalr,finalk,same_region_fai,same_region_pxy = Main_recon_3DSIM.main(
                opt, savepath1, savepath2, savepath3, run_parameter, isdebug=isdebug            )
                

        if run_parameter["just_cfai"] == 1:
            cfai, PGxy, PGz, plong, same_region_fai, same_region_pxy, Irtest = Main_recon_3DSIM.main(
                opt, savepath1, savepath2, savepath3, run_parameter, isdebug=isdebug
            )
            
    #统计cfai图
    path = _U.save_path(r'',run_parameter).parent
    print(f'遍历文件夹进行cfai文件统计：{path}')
    # allori_image_set=_SCfai.get_cfai(str(path),num_j=1,num_n=2) #4个区域 每个区域计算了3阶的初相（2,1,-1）
    # _U.save_tif_self(str(path)+f'\\3D-SIM初相统计集合-{data_name}.tif',allori_image_set)
    # _U.save_tif_self((savepath0)+f'\\3D-SIM初相统计集合-{data_name}.tif',allori_image_set)
    try:
        _SCfai.get_cfai(str(path),num_js=2)
    except:
        pass
    
    #统计P
    # Dr_sub = run_parameter['Dr_sub']
    # path = _U.save_path(r'',run_parameter).parent
    # print(f'遍历文件夹进行P文件统计：{path}')
    # _SCfai.get_P(str(path)) #传入阶数