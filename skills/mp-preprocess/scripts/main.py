import numpy as np
from tifffile import imread
from pathlib import Path
from datetime import datetime
import sys 
# sys.path.append('../')
import Utils as _U
import DrawOTF as _D
import CalRegistration as _C
# import Register as _R


BACKGROUND_ZERO_FRACTION_LIMIT = 0.20
BACKGROUND_STATS_FILENAME = 'background-subtraction-stats.md'


def _apply_run_quality_markers(report_path, para):
    markers = [] if para is None else para.get('quality_report_markers', [])
    return _U.apply_quality_report_markers(report_path, markers)


def _background_stats_path(para, pathfiname=None):
    if para is not None and para.get('background_stats_path'):
        return _apply_run_quality_markers(para['background_stats_path'], para)
    if (
        para is not None
        and pathfiname is not None
        and 'peizhun' in str(pathfiname).lower()
        and para.get('peizhun_file')
    ):
        return _U.sift_analysis_report_path(para)
    if para is not None:
        output_dir = _U.save_path('', para)
        report_path = output_dir / f'{output_dir.name}-{BACKGROUND_STATS_FILENAME}'
        return _apply_run_quality_markers(report_path, para)

    output_dir = Path.cwd() / 'dataOut'
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f'{output_dir.name}-{BACKGROUND_STATS_FILENAME}'
    return _apply_run_quality_markers(report_path, para)


def _append_background_stats(stats_path, stats, shared_sift_report=False):
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = stats_path.read_text(encoding='utf-8') if stats_path.exists() else ''
    if shared_sift_report and not report_text:
        report_text = f"# SIFT分析报告 - {stats['peizhun_file']}\n\n"
    if '## 背景扣除统计' not in report_text and '# 背景扣除统计' not in report_text:
        heading = '## 背景扣除统计' if shared_sift_report else '# 背景扣除统计'
        section_text = (
            f'{heading}\n\n'
            '> 默认规则：扣背景后新增零值比例大于20%时，记录质量告警并继续运行。\n\n'
            '| 时间 | 数据 | 文件 | 相机 | 原始均值 | 背景均值 | '
            '扣背景前零值 | 扣背景后零值 | 增加比例 | 是否超过20% |\n'
            '|---|---|---|---|---:|---:|---:|---:|---:|---|\n'
        )
        if shared_sift_report:
            title_end = report_text.find('\n') + 1
            report_text = (
                report_text[:title_end].rstrip() + '\n\n' + section_text + '\n'
                + report_text[title_end:].lstrip()
            )
        else:
            report_text += section_text
    values = [
        stats['timestamp'],
        stats['dataset'],
        stats['file'],
        stats['camera'],
        f"{stats['raw_mean']:.3f}",
        f"{stats['background_mean']:.3f}",
        f"{stats['zero_before']:.2%}",
        f"{stats['zero_after']:.2%}",
        f"{stats['zero_increase']:.2%}",
        '是（继续）' if stats['exceeds_limit'] else '否',
    ]
    row = '| ' + ' | '.join(str(value).replace('|', r'\|') for value in values) + ' |\n'
    if shared_sift_report:
        section_start = report_text.index('## 背景扣除统计')
        next_section = report_text.find('\n## ', section_start + 1)
        if next_section == -1:
            report_text = report_text.rstrip() + '\n' + row
        else:
            report_text = (
                report_text[:next_section].rstrip() + '\n' + row + '\n'
                + report_text[next_section + 1:]
            )
        stats_path.write_text(report_text, encoding='utf-8')
    else:
        if not stats_path.exists():
            stats_path.write_text(report_text, encoding='utf-8')
        with stats_path.open('a', encoding='utf-8') as file:
            file.write(row)


def _subtract_camera_background(Dr_read, background, pathfiname, camera, para=None):
    nx, ny = Dr_read.shape[-2:]
    background_used = np.asarray(background[:nx, :ny], dtype=np.float32)
    if background_used.shape != (nx, ny):
        raise ValueError(
            f'{camera}背景尺寸{background.shape}小于数据尺寸{(nx, ny)}'
        )

    zero_before = float(np.mean(Dr_read == 0))
    Dr_background_subtracted = Dr_read.astype(np.float32) - background_used
    np.maximum(Dr_background_subtracted, 0, out=Dr_background_subtracted)
    zero_after = float(np.mean(Dr_background_subtracted == 0))
    zero_increase = zero_after - zero_before
    exceeds_limit = zero_increase > BACKGROUND_ZERO_FRACTION_LIMIT

    pathfiname = Path(pathfiname)
    shared_sift_report = bool(
        para is not None
        and not para.get('background_stats_path')
        and 'peizhun' in str(pathfiname).lower()
        and para.get('peizhun_file')
    )
    stats_path = _background_stats_path(para, pathfiname)
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': pathfiname.parent.name,
        'file': pathfiname.name,
        'camera': camera,
        'raw_mean': float(np.mean(Dr_read, dtype=np.float64)),
        'background_mean': float(np.mean(background_used, dtype=np.float64)),
        'zero_before': zero_before,
        'zero_after': zero_after,
        'zero_increase': zero_increase,
        'exceeds_limit': exceeds_limit,
        'peizhun_file': para.get('peizhun_file') if para is not None else '',
    }
    if exceeds_limit:
        stats_path = _U.mark_quality_report_path(stats_path, 'BG！')
    _append_background_stats(stats_path, stats, shared_sift_report)
    print(
        f"{camera}背景扣除: raw_mean={stats['raw_mean']:.3f}, "
        f"background_mean={stats['background_mean']:.3f}, "
        f'zero_before={zero_before:.2%}, zero_after={zero_after:.2%}, '
        f'zero_increase={zero_increase:.2%}'
    )
    print(f'背景扣除统计已保存: {stats_path}')

    if exceeds_limit:
        _U.record_quality_warning(
            para,
            f'{pathfiname.parent.name}/{pathfiname.name} {camera}背景扣除后'
            f'新增零值比例{zero_increase:.2%}超过20%；'
            f'流程已继续，统计见{stats_path}',
        )
    return Dr_background_subtracted


#######################################################################
def pre_process(pathfiname,AVG=1,CHANNEL=1,TURN=1,camer_name='',para=None):
    "对4x1 or 1x1 的raw I，处理"
    "1.read_rawdata 这里减了一个背景, 3 这里是没做平均的 "
    '故出来的图像应该只是少了背景而已'
    #1.读入一个cam的图像：减背景,置0
    Dr_read = read_rawdata(pathfiname,para=para) #output：3D数据 
    assert len(Dr_read.shape)==3
    print('①Dr_read',Dr_read.shape)
    print('①Dr_read.dtype',Dr_read.dtype)
    
    #2.分离channel(只有一个的没有影响) #间隔变聚集(nc*nz,x,4y) (2*90,x,y)
    #这里需要手动选择一下
    if TURN==1:
        Dr_splitnc = splitChannel(Dr_read,CHANNEL=CHANNEL) #output：4d（2,100，X,Y）1212
    else:
        Dr_splitnc = splitChannel_2(Dr_read,CHANNEL=CHANNEL,AVG=AVG) #output：4d（2,100，X,Y） 1122
    print('②Dr_splitnc：',Dr_splitnc.shape) #(2,90,x,4y)
    del Dr_read

    #3.nt上做平均
    Dr_avg = make_avg(Dr_splitnc,AVG=AVG) #在每个nc上 每个nz上 拍了nt的时间点，可做平均
    print('③Dr_avg：',Dr_avg.shape)        #output：4d
    del Dr_splitnc

    #切分图像(放外面)
    # Dr_splitNto1,region_n= splitNto1(Dr_avg) #output：4d  （2,10*4，X,Y）,（2,10*1，X,Y）
    # print('④Dr_splitNto1：',Dr_splitNto1.shape) #需要知道每个cam 有几个区域-自动计算
    # del Dr_avg

    #翻转图像
    # camer_name = pathfiname.name[:4]
    # if camer_name=='cam2':
    #     print('将cam2镜像对称的图片翻转')
    #     Dr_out=fanzhuan(Dr_avg)
    # else:
    #     Dr_out = Dr_avg
    return Dr_avg,4        #返回5D数据，以及区域信息


def read_rawdata(pathfiname,bg1_path='',bg2_path='',para=None):  ##读入raw data，并减背景，outpu为3D
    #读入图像
    if para==None or ('peizhun' in str(pathfiname)) :
        Dr_read = imread(pathfiname)    #读进来应该是uint8 or u2
    else:
        start = para["start"] #(para['points']-1)*para['zhenshu_sep']
        final = para["final"] #(para['points']-1+para['point'])*para['zhenshu_sep']
        Dr_read = imread(pathfiname,key=range(start,final))    #读进来应该是uint8 or u2
        
    # dtypee = Dr_read.dtype
    camer_name = pathfiname.name[:4]
    
    #减背景 2D的适用，3D需要额外手动减点背景
    if bg1_path!='':
        bg1 = imread(bg1_path)
    else:
        # bg1 = imread(r'F:\code-F\AVG_cam1_background.tif')
        bg1 = imread(r'../../AVG_cam1_background.tif')

    if bg2_path!='':
        bg2 = imread(bg2_path)
    else:
        # bg2 = imread(r'F:\code-F\AVG_cam2_background.tif')
        bg2 = imread(r'../../AVG_cam2_background.tif')

    
    if camer_name == 'cam1':
        Dr_read = _subtract_camera_background(
            Dr_read, bg1, pathfiname, camer_name, para
        )
    elif camer_name == 'cam2':
        Dr_read = _subtract_camera_background(
            Dr_read, bg2, pathfiname, camer_name, para
        )
    
    #统一变为3维数据
    Dr_read = _U.image_reshape(Dr_read,dim=3)
    
    return Dr_read.astype(np.float16)


def splitChannel(Dr_read,CHANNEL=1):
    #1212
    print('1212')
    Dr_out=[]
    for i in range(CHANNEL): #    这里分为了2，nz，x，y
        Dr_out.append(Dr_read[i::CHANNEL])  # 这里增加了维度
    return np.array(Dr_out,dtype=np.float16)       #output 4D   #分离原本为1212的数据，output为（nc，nz/nt，nx，ny）
def splitChannel_2(Dr_read,CHANNEL=1,AVG=1):
    #1122
    print('1122')
    nz,nx,ny = Dr_read.shape
    Dr_out=[]
    for i in range(CHANNEL):
        Dr_out.append(Dr_read.reshape(nz//AVG,-1)[i::CHANNEL].reshape(nz//CHANNEL,nx,ny))
    return np.array(Dr_out,dtype=np.float16)#分离原本为1122的数据，output为（nc，nz/nt，nx，ny）

def make_avg(Dr_read,AVG=1): #在每个nc上 每个nz上 拍了nt的时间点，可做平均
    Dr_out=[]
    nc_items= Dr_read.shape[0]
    nz_items = Dr_read.shape[1]//AVG  #20//5  0//5
    for nc_i in range(nc_items):
        temp_list = []               #中间加一个list 保持维度不变
        temp = Dr_read[nc_i]         #降维
        for nz_i in range(nz_items): # 0 1 2 3
            start = nz_i*AVG
            end = (nz_i+1)*AVG
            temp_list.append( np.mean(temp[start:end],axis=0) ) #降维 升维
        Dr_out.append(temp_list)                                #升维
    return np.array(Dr_out,dtype=np.float16) #4D

def splitNto1(Dr_read):    
    '增加region_n，可处理单区域图像 '
    if len(Dr_read.shape)==2:
        nx,ny4 = Dr_read.shape  #
        nc ,nz = 1,1
    elif len(Dr_read.shape)==3:
        nz,nx,ny4 = Dr_read.shape
        nc = 1
    elif len(Dr_read.shape)==4:
        nc,nz,nx,ny4 = Dr_read.shape
    else:
        raise('error Dr_read dim')
    
    Dr_read.resize(nc,nz,nx,ny4)   #统一为4维度
    # region_n = ny4//nx     #2048 / 512 的情况
    region_n = 4      #目前MP情况都为4
    ny = ny4//region_n
    Dr_out = []
    for nc_i in range(nc): #拿出channel
        temp_channel = Dr_read[nc_i]   #down dim
        temp_list = []
        for ny_i in range(region_n):  
            temp_list.append(temp_channel[:,:,(ny_i*ny):((ny_i+1)*ny)]) #down  up dim
        Dr_out.append(temp_list)     #up dim
    Dr_out = np.array(Dr_out,dtype=np.float16).reshape(nc,-1,nx,ny)  #直接reshape 一下最简单
    return Dr_out       #4D

def fanzhuan(arr):
    if len(arr.shape) == 3:
        return arr[:,:,::-1]
    if len(arr.shape) == 4:
        return arr[:,:,:,::-1]


##################################################################
def cam12_5dTo4d(data5d,region_n):
    '5D (cam,nc,nt*nz,nx,ny) 数据变为 '
    '4D（region_n，nt/nz，nx，ny） ）'   #注意循序需要后面再调
    '还具有调节各个区域分光的作用'
    cam,nc,ntnz,nx,ny = data5d.shape
    assert nc==1  # channel 通道为1

    region_n_1 = int(region_n//cam)        #每个cam中有几个区域
    ntnz_1 = int(ntnz//region_n_1)         #每个区域下有多少ntnz

    data4d = np.zeros((region_n,ntnz_1,nx,ny),dtype=np.float32)

    # c = np.array([1.2,1.08,1.08,1,1,1.18,1.06,1.3])

    # c0 = np.sum(data5d[0,0])
    
    for c_i in range(cam): #遍历cam            #0 ,1
        stack_3D = data5d[c_i,0] #4321 5678
        for r_n in range(region_n_1):  #0,1,2,3
            s = r_n*ntnz_1
            e = (r_n+1)*ntnz_1
            data4d[c_i*region_n_1+r_n] = stack_3D[s:e]   #这里不做分光校正，

            # if c_i==0 and r_n==0:#计算第一个相机，第一个区域的均值
            #     c0 = np.mean(data4d[c_i*region_n_1+r_n])
            # ctemp = np.mean(data4d[c_i*region_n_1+r_n]) #计算每层的均值
            # print(f'{r_n}区域整体均值为{ctemp.round(2)},用{data4d[c_i*region_n_1+r_n].shape}来计算的',)
            # c = ctemp/c0
            # print('校正8层分光c=',c)
            # data4d[c_i*region_n_1+r_n] = data4d[c_i*region_n_1+r_n]/c #[c_i*region_n_1+r_n]
            
    return data4d

def prism_correct(data4D,margin=50):
    assert len(data4D.shape)==4
    nc,nz,nx,ny = data4D.shape
    if nx-margin<=0:
        raise('error! nx<50,修改下面限制范围')
    # data4D_430 = _U.pad_or_cut(data4D,x=nx-margin,y=ny-margin,z=nz) #用来限制计算区域
    data4D_430 = data4D[:,:,40:nx-10, 15:ny-35]
    
    for r_n in range(nc):
        if r_n == 0:
            c0 = np.mean(data4D_430[r_n])
        ctemp = np.mean(data4D_430[r_n]) #计算每层的均值
        print(f'{r_n}区域整体均值为{ctemp.round(2)},用{data4D_430[r_n].shape}来计算的',)
        c = c0/ctemp
        print('校正8层分光c=',c)
        data4D[r_n] = data4D[r_n]*c
    return data4D    


def prism_correct_2(data4D,margin=50):
    #不同区域，分光矫正采用两通道单独矫正
    assert len(data4D.shape)==4
    nc,nz,nx,ny = data4D.shape
    # if (nx-margin)<=0:
    #     raise('error! nx<50,修改下面限制范围')
    # data4D_430 = _U.pad_or_cut(data4D,x=nx-margin,y=ny-margin,z=nz) #用来限制计算区域

    data4D_430 = data4D[:,:,40:nx-10, 15:ny-35]
    _,_,nx2,ny2= data4D_430.shape
    #区分不同通道 
    data5D = data4D.reshape(8,nz//30,30,nx,ny) #理解：一行30个，（488-1，488-2）；（561-1，561-2） 
    data5D_430 = data4D_430.reshape(8,nz//30,30,nx2,ny2)  # (8,20,30,472,472)
    del data4D,data4D_430

    #统计区域4的两通道强度均值
    c3_0 = np.mean(data5D_430[3,0::2,:,:,:])  #统计3区域1通道均值
    c3_1 = np.mean(data5D_430[3,1::2,:,:,:])
    for r_n in range(nc):
        # if r_n == 0: #
        #     c0 = np.mean(data4D_430[r_n])
        ctemp_0 = np.mean(data5D_430[r_n,0::2]) #统计每个区域1通道均值
        ctemp_1 = np.mean(data5D_430[r_n,1::2]) #统计每个区域2通道均值
        print(f'{r_n+1}区域_1通道整体均值为{ctemp_0.round(2)},用{data5D_430[r_n,0::2].shape}来计算的',)
        # c = ctemp_0/c3_0
        data5D[r_n,0::2] = data5D[r_n,0::2]*(c3_0/ctemp_0)
        print('校正8层分光c=',(c3_0/ctemp_0))
        
        print(f'{r_n}区域_通道2整体均值为{ctemp_1.round(2)},用{data5D_430[r_n,1::2].shape}来计算的',)
        # c = ctemp_1/c3_1
        data5D[r_n,1::2] = data5D[r_n,1::2]*(c3_1/ctemp_1)
        print('校正8层分光c=',(c3_0/ctemp_0))

    
    #额外手动分光系数
    # xishu  = [1.467603351,1.285559018,1.149709986,1.036047972,1,1.005231873,1.061455943,1.147432051]
    # for i in range(8):
    #     data5D[i,1::2] = data5D[i,1::2]* xishu[i]
    # data5D[1,1::2] = data5D[1,1::2]* 1.491
    # data5D[2,1::2] = data5D[2,1::2]* 1.239
    # data5D[3,1::2] = data5D[3,1::2]* 1.054
    # data5D[4,1::2] = data5D[4,1::2]* 1.000
    # data5D[5,1::2] = data5D[5,1::2]* 1.008
    # data5D[6,1::2] = data5D[6,1::2]* 1.094
    # data5D[7,1::2] = data5D[7,1::2]* 1.235

    return data5D.reshape(8,-1,nx,ny)    


def ori_correct_2(data4D,margin=40):
    #仍然分方向矫正
    assert len(data4D.shape)==4
    nc,nz,nx,ny = data4D.shape
    if (nx-margin)<=0:
        raise('error! nx<40,修改下面限制范围')
    # data4D_430 = _U.pad_or_cut(data4D,x=nx-margin,y=ny-margin,z=nz) #用来限制计算区域
    data4D_430 = data4D[:,:,40:nx-10, 15:ny-35]
    _,_,nx2,ny2= data4D_430.shape
    
    # data5D_430 = data4D_430.reshape(8,15,nz//15,nx-margin,ny-margin)
    # data5D_430_swap =np.swapaxes(data5D_430 ,1,2) 
    data5D = data4D.reshape(8,nz//30,30,nx,ny)  #一行30个，（488-1，488-2）；（561-1，561-2） 
    data5D_430 = data4D_430.reshape(8,nz//30,30,nx2,ny2)
    del data4D,data4D_430

    #统计方向1的强度均值--通道1--通道2
    c1_1 = np.mean(data5D_430[:,0::2,0:5,:,:])   #统计所有区域，所有1通道-1时间点， 方向1 均值
    c1_2 = np.mean(data5D_430[:,0::2,5:10,:,:])  #                               方向2 均值
    c1_3 = np.mean(data5D_430[:,0::2,10:15,:,:]) #                               方向3 均值
    
    c1_4 = np.mean(data5D_430[:,0::2,15:20,:,:])  #统计所有区域，所有1通道-2时间点，方向1 均值
    c1_5 = np.mean(data5D_430[:,0::2,20:25,:,:])  #                              方向2 均值
    c1_6 = np.mean(data5D_430[:,0::2,25:30,:,:])  #                              方向3 均值

    c2_1 = np.mean(data5D_430[:,1::2,0:5,:,:])    #统计所有区域，所有2通道-1时间点，方向1 均值
    c2_2 = np.mean(data5D_430[:,1::2,5:10,:,:])
    c2_3 = np.mean(data5D_430[:,1::2,10:15,:,:]) 

    c2_4 = np.mean(data5D_430[:,1::2,15:20,:,:])  #统计所有区域，所有2通道-2时间点，方向1 均值
    c2_5 = np.mean(data5D_430[:,1::2,20:25,:,:])
    c2_6 = np.mean(data5D_430[:,1::2,25:30,:,:]) 

    #分方向矫正
    # data5D[:,0::2,0:5] = data5D[:,0::2,0:5]*(c1_1/c1_1)
    data5D[:,0::2,5:10] = data5D[:,0::2,5:10]*(c1_1/c1_2)
    data5D[:,0::2,10:15] = data5D[:,0::2,10:15]*(c1_1/c1_3)

    data5D[:,0::2,15:20] = data5D[:,0::2,15:20]*(c1_1/c1_4)
    data5D[:,0::2,20:25] = data5D[:,0::2,20:25]*(c1_1/c1_5)
    data5D[:,0::2,25:30] = data5D[:,0::2,25:30]*(c1_1/c1_6)

    #两个时间点的矫正
    C1_7 = np.mean(data5D_430[:,0::2,0:15,:,:])
    C1_8 = np.mean(data5D_430[:,0::2,15:30,:,:])
    data5D[:,0::2,15:30] = data5D[:,0::2,15:30]*(C1_7/C1_8)


    # data5D[:,1::2,0:5] = data5D[:,1::2,0:5]*(c2_1/c2_1)
    data5D[:,1::2,5:10] = data5D[:,1::2,5:10]*(c2_1/c2_2)
    data5D[:,1::2,10:15] = data5D[:,1::2,10:15]*(c2_1/c2_3)

    data5D[:,1::2,15:20] = data5D[:,1::2,15:20]*(c2_1/c2_4)
    data5D[:,1::2,20:25] = data5D[:,1::2,20:25]*(c2_1/c2_5)
    data5D[:,1::2,25:30] = data5D[:,1::2,25:30]*(c2_1/c2_6)

    #两个时间点的矫正
    C2_7 = np.mean(data5D_430[:,1::2,0:15,:,:])
    C2_8 = np.mean(data5D_430[:,1::2,15:30,:,:])
    data5D[:,1::2,15:30] = data5D[:,1::2,15:30]*(C2_7/C2_8)

    return data5D.reshape(8,-1,nx,ny) 


def phase_correct_2(data4D,margin=40):
    #仍然分方向矫正
    assert len(data4D.shape)==4
    nc,nz,nx,ny = data4D.shape
    if (nx-margin)<=0:
        raise('error! nx<40,修改下面限制范围')
    # data4D_430 = _U.pad_or_cut(data4D,x=nx-margin,y=ny-margin,z=nz) #用来限制计算区域
    data4D_430 = data4D[:,:,40:nx-10, 15:ny-35]
    _,_,nx2,ny2= data4D_430.shape
    

    # data5D_430 = data4D_430.reshape(8,15,nz//15,nx-margin,ny-margin)
    # data5D_430_swap =np.swapaxes(data5D_430 ,1,2) 
    data5D = data4D.reshape(8,nz//30,30,nx,ny)  #一行30个，（488-1，488-2）；（561-1，561-2） 
    data5D_430 = data4D_430.reshape(8,nz//30,30,nx2,ny2)
    del data4D,data4D_430

    #统计方向1的强度均值--通道1--通道2
    C1 = []
    C0 = []
    for i in range(30):
        if i == 0:
            C0.append(np.mean(data5D_430[:,0::2,0:5,:,:]))
        elif i == 15:
            C0.append(np.mean(data5D_430[:,0::2,15:20,:,:]))
        
        if i <15: #1时间点
            C1.append(np.mean(data5D_430[:,0::2,i,:,:]))   #统计所有区域，所有1通道-1时间点，方向1 相位1均值
        elif i >= 15:
            C1.append(np.mean(data5D_430[:,1::2,i,:,:]))    #统计所有区域，所有2通道-1时间点，方向1 均值

    #分相位矫正
    for i in range(30):
        if i <15: #1时间点
            data5D[:,0::2,i] = data5D[:,0::2,i]*(C0[0]/C1[i])
        # elif i >= 15:
        #     data5D[:,1::2,i] = data5D[:,1::2,i]*(C0[15]/C1[i])

    #暴力 两个通道各自拉到一致
    # for k in range(nz//30):
    data4D =  data5D.reshape(8,-1,nx,ny) 
    


    return data5D.reshape(8,-1,nx,ny) 
#######################################################################














#*************************************************************************************************#
#                                        main                                                     #
#*************************************************************************************************#
def iter_tif_files(path):
    for item in Path(str(path)).iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() not in ('.tif', '.tiff'):
            continue
        yield item


def main(savepath1,pro_meth=None,mode=2,Registration=1,pgxy=0,theta=0,scale=0,Hs=0,file_suffix=None,stand=4,ratio_thresh=1,para=None):
    pro_meth = _C.get_registration_projection_method(pro_meth, Registration)
    p = iter_tif_files(savepath1) #仅遍历当前目录下tif文件
    
    #公共，读取raw 为一个5D数据（cam,nc,nt/nz,nx,ny）
    cam12_5d = []
    # cam1_4d = []
    cam1_4d = []
    cam2_4d = []

    region_n = 0
    cam1_i = 0 
    cam2_i = 0
    point_re1 = para["point"]*15  #剩余帧数
    point_re2 = para["point"]*15
    for ii in p:  #遍历两个raw cam相机
        if not ii.is_file(): #不是文件-->跳过
            continue 
        # print(ii)
        #如果是cam1 
        if "cam1" in ii.name and point_re1!=0:  #正常第一组数据
            # camer_name = ii.name[:4]             #这里要求文件名字前缀为cam1 cam2
            # print(camer_name+'.......'+ii.name[4:] )
            cam1_i += 1
            #当前文件开始和结束区间-#两个选择 2037 和 3610
            
            if point_re1 >= para['volume']  :  #剩下时间点大于一个文件
                para['start'] = 0
                para['final'] = para['volume'] #直接全部读取完
                point_re1-=para['volume']        
            elif point_re1 < para['volume'] : #剩下时间点在一个文件以内
                para['start'] = 0#4*15 
                para['final'] = point_re1#(para["point"]+4)*15 #point_re1     #- para['zhenshu_sep']
                point_re1=0
            
            #设置开始时间点
            if cam1_i==1 and para['points']!=1 : #第一次选择读入的开始
                para['start'] = (para['points'] -1)*para['zhenshu_sep']#para['points_name']-1

            cam1_4d_temp,_ = pre_process(ii,AVG=1,CHANNEL=1,camer_name='cam1',para=para) #  eg 读取前140个时间点  0-2037 + 0+63
                
            cam1_4d.append(cam1_4d_temp)   #2037，63
            if cam1_i==1:
                region_n += 4      #保存几个区域信息

        elif 'cam2' in ii.name and point_re2!=0:
            cam2_i += 1      
            if point_re2 >= para['volume']  :    #两个选择 2037 和 3610
                para['start'] = 0#(para['points']-1)*para['zhenshu_sep']
                para['final'] = para['volume']#(para['points']-1+para['point'])*para['zhenshu_sep']
                point_re2-=para['volume']
            elif point_re2 < para['volume'] :
                para['start'] = 0
                para['final'] = point_re2 #- para['zhenshu_sep']
                point_re2=0
            
            #设置开始时间点
            if cam2_i==1 and para['points'] !=1 : #第一次选择读入的开始
                para['start'] = (para['points'] -1)*para['zhenshu_sep']#para['points_name']-1

            cam2_4d_temp,_ = pre_process(ii,AVG=1,CHANNEL=1,camer_name='cam2',para=para)
                
            cam2_4d.append(cam2_4d_temp)
            if cam2_i==1:
                region_n += 4      #保存几个区域信息
            
        else:  #跳过
            continue

    assert cam2_i == cam1_i #两者要相等  
    
    #将2037 963 轴向叠加起来，组合得到cam12-5D-temp
    result  = cam1_4d[0]  #1，2037，512，2048
    for arr in cam1_4d[1:]:
        result = np.concatenate((result, arr), axis=1)
    del cam1_4d
    
    result2  = cam2_4d[0]
    for arr in cam2_4d[1:]:
        result2 = np.concatenate((result2, arr), axis=1)
    del cam2_4d
    
    
    #切分图像(放外面)
    result_splitNto1 = splitNto1(result) #output：4d  （2,10*4，X,Y）,（2,10*1，X,Y）
    print('cam1:',result_splitNto1.shape) #需要知道每个cam 有几个区域-自动计算
    del result

    result2_splitNto1 = splitNto1(result2) #output：4d  （2,10*4，X,Y）,（2,10*1，X,Y）
    result2_splitNto1_F=fanzhuan(result2_splitNto1) #添加一个翻转？
    print('cam2',result2_splitNto1_F.shape) #需要知道每个cam 有几个区域-自动计算
    del result2,result2_splitNto1

    cam12_5d.append(result_splitNto1) # [4D,4D] 
    cam12_5d.append(result2_splitNto1_F) # [4D,4D]
                
    del result_splitNto1,result2_splitNto1_F
    
    
    print(f'有{len(cam12_5d)}个数据，{region_n}个区域')
    print()
    
    cam12_5d = np.array(cam12_5d)     #(cam,nc,nt*nz,nx,ny)
    
    # _U.save_tif(r'only sub BG cam12_5d.tif',cam12_5d) #验证过 是的
    


    #数据重排列为4D （region_n，nt/nz，nx，ny）
    cam12_4d = cam12_5dTo4d(cam12_5d,region_n)
    
    nzz = cam12_4d.shape[1]
    del cam12_5d
    
    if mode ==2:   #可以删除第一种
        cam12_4d = cam12_4d[:,1:]   #删除所有区域的第一帧
        print(f'删除了所有区域的第一帧,{nzz}变为{nzz-1}')
    else:
        print(f'不删除帧,有{nzz}')
        
    _U.save_tif(r'cam12_4d.tif',cam12_4d)  #这个文件会很大
    
    #提取OTF
    if mode==1:           #drawOTF_f == 1:
        ROIS_1,ROIS_2,ROIS_3,ROIS_4,ROIS_5,ROIS_6,ROIS_7,ROIS_8 = _D.drawROIs(cam12_4d)
        #未完成
        pass


    #计算配准
    if mode==2:           
        if Registration==1: 
            pgxy = np.zeros((8,2))
            theta = np.zeros((8))
            scale = np.ones((8))
            coordinate_beads,coordinate_theta,coordinate_scale,imag_5 = _C.calRegistra(cam12_4d,pro_meth=pro_meth,Registration=1,pgxy=pgxy,theta=theta,scale=scale,file_suffix = file_suffix,stand=stand,para=para) #
            return   coordinate_beads,coordinate_theta,coordinate_scale,imag_5

        elif Registration==3:
            Hs = _C.calRegistra(cam12_4d,pro_meth=pro_meth,Registration=3,pgxy=pgxy,theta=theta,scale=scale,file_suffix = file_suffix,stand=stand,ratio_thresh=ratio_thresh,para=para,focus_source_layer_offset=2) #
            return Hs
    #应用配准
    if mode==3:           #Register_f == 1:
        if Registration==1:
            print('pgxy[0]:',pgxy[0])
            imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = _C.calRegistra2(cam12_4d,pro_meth='mean',pgxy=pgxy,theta=theta,scale=scale,Regster_f=1,file_suffix=file_suffix) # #配准的那个量也需要
            # imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = _C.calRegistra2(cam12_4d,pro_meth='max',pgxy=pgxy,theta=theta,scale=scale,Regster_f=1,file_suffix=file_suffix) # #配准的那个量也需要

            # imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = _C.calRegistra(cam12_4d,pro_meth='mean',pgxy=pgxy,theta=theta,scale=scale,Regster_f=1,file_suffix=file_suffix)
            # imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = _C.calRegistra(cam12_4d,pro_meth='mean',pgxy=pgxy,theta=theta,scale=scale,Regster_f=1,file_suffix=file_suffix)

            #直接移动
            # print('直接移动')
            # imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = _C.DirectRegistra(cam12_4d,pro_meth=pro_meth,pgxy=pgxy,theta=theta,scale=scale,file_suffix=file_suffix,para=para)
            return   np.array([imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M])
        

        elif Registration==3:

            print('直接移动')
            imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = _C.DirectRegistra(cam12_4d,pro_meth=pro_meth,Hs=Hs,file_suffix = file_suffix,para=para)

            return   np.array([imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M])


    if mode == 4: #先施加仿射变换，再次计算45区域的变化，作用到pgxy上面，
        print('！！！一、先直接移动Hs_base')
        imag_4_M,imag_5_M= _C.DirectRegistra45(cam12_4d,pro_meth=pro_meth,pgxy=pgxy,theta=theta,scale=scale,file_suffix=file_suffix,para=para)


        print('！！二、再配准一遍45区域')
        #方法1
        # Hs2 = np.copy(Hs)
        pgxy45,theta45,scale45 = _C.calRegistra45_3_3(imag_4_M,imag_5_M,pro_meth=pro_meth,Registration=1,file_suffix = file_suffix,stand=1,ratio_thresh=ratio_thresh,para=para) #

        ##########更新pgxy###
        # print("原始pgxy = ",pgxy.round(2))
        # print("pgxy45 = ",pgxy45.round(2))
        # print("原始theta = ",theta.round(2))
        # print("theta45 = ",theta45.round(2))
        pgxy2  = np.copy(pgxy) 
        theta2 = np.copy(theta)
        scale2 = np.copy(scale)


        pgxy2[0:4] = pgxy[0:4] + pgxy45[0]
        theta2[0:4] = theta[0:4] + theta45[0]
        # print("改变后pgxy2 = ",pgxy2.round(2))
        # print("改变后theta = ",theta2.round(2))
        print('！！三、最后配准一遍')
        imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M  = _C.DirectRegistra(cam12_4d,pro_meth=pro_meth,pgxy=pgxy2,theta=theta2,scale=scale2,file_suffix=file_suffix,para=para)

        return   np.array([imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M])


    if mode ==6:
        # print('直接移动')cam12_4d 4D （region_n，nt/nz，nx，ny）
        #用信息直接配准
        if Registration==1: 
            print('mode=6,方法①互相关直接移动')
            imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = _C.DirectRegistra(cam12_4d,pro_meth=pro_meth,pgxy=pgxy,theta=theta,scale=scale,file_suffix=file_suffix,para=para)
            return   np.array([imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M])

        elif Registration==3:
            print('mode=6,方法③SIFT直接移动')
            imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = _C.DirectRegistra(cam12_4d,pro_meth=pro_meth,Hs=Hs,file_suffix = file_suffix,para=para)
            return   np.array([imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M])

        else:
            raise(f'Registration error,{Registration=:}')
