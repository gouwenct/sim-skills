import Utils as _U
import numpy as np
from pathlib import Path
import argparse
from loguru import logger

import Main_recon_MP as _MMP
import Statistic_Cfai as _SCfai
# import save_stack as _ss
from tifffile import imread

'''
$V2$ 版本把stack3D slice3D加入  
最大的不同就是OTF，参数估计只求Pxy  
v2.51 增加文件遍历操作，在将MP数据预处理之后，自动遍历指定文件夹，并且将数据保存到这个文件夹下  
**save_tif_v21(filename2,filename,data)**

V2.6 增加open-3DSIM的notch、filter；pz的估计可以通过一阶来计算，无需真的去估计
采用MCNR方法对2Dpattern进行求P，计算p2p1pz，然后在三维上估计C phase

V2.8 增加XC-1 XC-2 方式重建-需要调试
V2.9 发现APO不对称问题 注释掉WF里面的  apodiz = apodiz#*zATT
V2.91 发现OTF移动不对称令其对称（real），PE参数估计对称化 parameter_estimation1_2
V3.0  OTFmask生成用8层方法，然后移动；200nm分辨率问题出在切趾问题上，更换APO文件得到解决
      还有一个频谱微小的不对称问题，由于1阶左右频谱不一样在经过pz移动后整体会有一点小偏差
      解决思路，要么将8层变为9层（能做到，但可能稍微花点时间，且不一定有较大提升）；要么将另一侧多出来的频谱去掉，损失分辨率
V3.2 修改OTFmask为对称的8层，APO轴向衰减用opt.Apoz来控制，增加XC的方法， ------------这个版本后频谱应该是对称的了
V3.3 将ck分离出来的8层变为9层，将OTF变为9层，将OTFmask变为9层；再考虑Zpadding的事
      OTF，OTFmask可以直接对称过去
      ck，0阶上下位共轭对称、
V3.4 没有做9层的情况，仿真实验差不多，用XC的方式，去掉real
V3.41 增加pz遍历
V3.42 每个方向单独重建
V3.43 增加XYD-4(4B的方式),在这个里面增加0阶notch
V3.44 增加了初相间隔遍历，增加其他的分离方式- 52 34 7pi  5pi 
V3.5  另外一个分支 两个时间点联立求解（见V3.5版本）
V3.6  添加CP求逆那里 添加 Tikhonov 正则项
V3.7  整理代码，将opt.Nshift 拆分为opt.Nshift（分离之前） 与 opt.order（用于重建）;添加opt.sequence 控制不同时序分离
      取消8 9 层的nz改动，Zpadding用0零的方式
V3.8  7张图，用5x5,2x2方式求解2阶与1阶（用52的数据）sequence=5252(证明与直接7X7方式一样)
V3.9  5x5分离矩阵 使用matlab 做稍微的变化,看看是否有提升（没做）
V3.9  joint deconvolution ， 使用55模式，将两个ck2都提取出来； 计算φ，C，在最后重建时，额外加上这个2阶
V4.0  完善55模式,先5X5分离出两个时间点的C0 C2 C1 ,各时间点计算并消去2阶横向初相,消去1阶横向初相(理论上是2阶的一半)。
      再2x2联立求出4个分离的1阶
V4.1  给XYD,XC加上联合解卷积方法,
V4.2   V4.1 精简版, 去掉5343 5352 52模式; 去掉remode=1234，只保留0情况;只求一次P(未做)
V4.3  在北医改过后，增加saveonly模式，Dr的mask做了调整，可以应对长方形情况
V4.4  save-onely ，padding+参数估计只算一次
V4.5  经过24.3.14结果验证，第一二时间点的初相没差太多，所以不再额外消除2阶初相（保留在ck中，后面提取Sk时再消除）
V4.51 保存文件增加name在上面
V4.52 一些小BUG修改（XC方式可以work），给4B加上0阶notch
V4.53 增加复数函数（插值、归一化、wiener，但重建有问题），还是使用abs方式（在OTF ）（感觉最主要的贡献为OTF，0阶notch的影响，但目前的问题是分辨率下不来）
V4.54 新增文件【】Run2.py 可一次性先参数估计+重建数据
V4.55 初相使用自己匹配的初相，OTF层间距为200
V4.57-591 修改参数模式（0,1,2,3,4,5），保存文件命名模式；命名修改；缩短,
V4.6  #修改nxny430-462,OTF空域变形
V4.61 #7X7模式
V4.62 微调名字命名
V4.7  #将4.5废除的2阶初相修改回来,并考虑如何加回来 运行速度不会受损
V4.8  #第二个自己算
V4.9  #【notch的对冲会影响背景,通过notch=0 1控制】; 增加cfai_save cfai_02_save,保存2阶初相,但后续初相为0 ; OTF 加载时变换去除np.real
V4.91 #保存P fai para 增加对双色的保存
V4.92 #缩减名字，用uint16保存在only里面
V4.93 save_stack自动保存
V4.94 增加fast（重建一个时间点）/save参数（保存ck和最终频谱）
V4.95 增加插值操作
V6.0    每个频域分量单独加mask（不要加在OTF上）
# V6.2  打算保证输入输出的结果差不多--返回V6.0 只要OTF做替换，与归一化到0-1就行
V6.3
V6.4 0padding+mask
V6.5 保存V22 自动调整对比度,Apo只用一次
'''


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
    #传入三个文件路径 Dr，OTF，parameter
    #Dr目录下会自动遍历所有文件，需要与OTF下文件数一致
    
    # isdebug=[0,0,1,1,1,0,0,0]
    isdebug = run_parameter['DEBUG']   
    """
    #0,   ; 1 加载数据 Dr；   #2 分离分量 Ck ；   3 其他数据处理 ;4 参数估计
            5 GetMsum
    """
    ##############################【parameter】extra debug parameter
    opt.version ='V6.5'
    run_parameter['ver']=opt.version
    if run_parameter['mear'] == 1:
        opt.Apoz = 600
        opt.fcxs = 1.7#1.4                  # Axial range of the apodization function 1
        opt.plongxs = 1.7#opt.fcxs#
        run_parameter['note']= 'ma60'
    elif run_parameter['mear'] == 0:
        opt.Apoz = 750
        opt.fcxs = 1.2#1.4                  # Axial range of the apodization function 1
        opt.plongxs = 1.2#opt.fcxs#1.5 
        run_parameter['note']= 'simu'
    elif run_parameter['mear'] == 2:
        opt.Apoz = 830
        opt.fcxs = 1.2                  # Axial range of the apodization function 1
        opt.plongxs = 1.2#opt.fcxs#1.5 
        run_parameter['note']= 'cell'
    ##################################【run_parameter】
    
    #######################################【V2.51遍历文件夹】

    loop_dir = sorted(
        (path for path in Path(str(savepath0)).iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    source_data_name = str(Path(str(savepath0)).name)  #提取当日数据名称，eg 7.1 MP-SIM超多数据
    data_name = run_parameter.get('output_data_name', source_data_name)
    run_parameter['data_name'] = data_name 
    print(data_name)
    
    for i in FANWEI:
        if not 1 <= i <= len(loop_dir):
            print(f'【Loop-FILE】跳过,i={i}超出文件夹范围1-{len(loop_dir)}')
            continue
        
        ldr_i = loop_dir[i-1]
        dir_name = ldr_i.parts[-1]  # 获取文件夹名
        
        savepath1 = ldr_i/'Dr/'/run_parameter['Dr_sub']
        savepath2 = OTFone #+'/1\\' #ldr_i/'OTF\\'      #1真实OTF     3 from 3DOTF
        
        savepath3 = ldr_i/'parameter/'/run_parameter['W_order'] #
        if not savepath3.exists():
            savepath3.mkdir()
        
        print('【Loop-FILE】处理',dir_name)
        run_parameter['dir_name'] = dir_name
        # run_parameter['log'] = f'./dataOut/{data_name}/{dir_name}-{opt.version}/【{dir_name}】.log'
        save_parameter(opt,run_parameter) 
        cfais7 = []
        pgxy7 = []
        PGz7 = []
        
        # for pgzz in run_parameter['pzzzs'] :  #遍历PZ 
        #     run_parameter['pzzz'] = pgzz
        
        #完整
        if run_parameter['just_cfai']==0:
            opt,cfai,PGxy,PGz,plong,finalr,finalk,same_region_fai,same_region_pxy = _MMP.main(opt,savepath1,savepath2,savepath3,run_parameter,isdebug=isdebug)

        if run_parameter['just_cfai']==1:
            cfai,PGxy,PGz,plong,same_region_fai,same_region_pxy,Irtest = _MMP.main(opt,savepath1,savepath2,savepath3,run_parameter,isdebug=isdebug)

            # logger.remove(handler_id=None)              #停止这个log对象
        
        # pathhh = _U.save_tif_v22_path(run_parameter) #得到only那个文件夹
        # _ss.loop_pz_or_channal(pathhh,chazhi=run_parameter['chazhi'])#保存文件
        
            #统计cfai图
            # Dr_sub = run_parameter['Dr_sub']
            # path = _U.save_path(r'',run_parameter).parent
            # print(f'遍历文件夹进行cfai文件统计：{path}')
            # try:
            #     _ SCfai.get_cfai(str(path),num_js=1,Dr_sub=Dr_sub+f'-{pgzz}') #传入阶数
            # except:
            #     pass
            
            # try:
            #     _SCfai.get_cfai(str(path),num_js=3,Dr_sub=Dr_sub+f'-{pgzz}') #传入阶数
            # except:
            #     pass
            
            # try:
            #     _SCfai.get_cfai(str(path),num_js=5,Dr_sub=Dr_sub+f'-{pgzz}')
            # except:
            #     pass
            
            # # #统计P
            # if run_parameter['loadflag']==0:
            #     Dr_sub = run_parameter['Dr_sub']
            #     path = _U.save_path(r'',run_parameter).parent
            #     print(f'遍历文件夹进行P文件统计：{path}')
            #     _SCfai.get_P(str(path)) #传入阶数
                
            
            # cfais7.append(same_region_fai)
            # pgxy7.append(same_region_pxy)
            # PGz7.append(PGz)

            # cfais_n = np.array(cfais7)
            # pgxy_n = np.array(pgxy7)
            # PGz_n = np.array(PGz7)
            # #保存 
            # np.save(str(savepath3)+rf'\\para\\pgxy_n-{i}-.npy',pgxy_n)    #跳转到重新组合cfais_n，pgxy_n
            # np.save(str(savepath3)+rf'\\para\\cfais_n-{i}-.npy',cfais_n)


# if __name__ == '__main__':
#     main()
