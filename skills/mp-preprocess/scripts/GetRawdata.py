# import BaseFunctions_np_7_1 as n
import sys 
sys.path.append('../')
import Utils as _U

import Cam_process as ca 
# import pocpy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


BRIGHTNESS_WARNING_FRACTION = 0.20
BACKGROUND_STATS_FILENAME = 'background-subtraction-stats.md'


def _background_stats_path(para):
    output_dir = _U.save_path('', para)
    return output_dir / f'{output_dir.name}-{BACKGROUND_STATS_FILENAME}'


def get_final_cropped_data(data, para):
    _, nx, ny = data.shape
    top, bottom, left, right = para.get(
        'final_crop_margins', (65, 39, 54, 50)
    )
    if min(top, bottom, left, right) < 0 or top + bottom >= nx or left + right >= ny:
        raise ValueError(
            f'final_crop_margins={(top, bottom, left, right)} '
            f'与数据尺寸{(nx, ny)}不兼容'
        )
    return data[:, top:nx - bottom, left:ny - right]


def _get_sorting_function(func_f):
    functions = {
        2121: sorting21_21,
        212152: sorting21_21_52,
        2115: sorting21_15,
        219: sorting21_9,
        123: sorting_series,
        3015: sorting30_52,
    }
    if func_f not in functions:
        raise ValueError(f'不支持的func_f={func_f}')
    return functions[func_f]


def calculate_brightness_matrix(data, region_n, zhenshu):
    if zhenshu % 3 != 0:
        raise ValueError(f'zhenshu={zhenshu}不能均分为3个方向')
    expected_frames = region_n * zhenshu
    if data.shape[0] != expected_frames:
        raise ValueError(
            f'亮度分析需要{expected_frames}帧，实际为{data.shape[0]}帧'
        )
    phase_n = zhenshu // 3
    shaped = np.asarray(data, dtype=np.float32).reshape(
        region_n, 3, phase_n, data.shape[-2], data.shape[-1]
    )
    return np.mean(shaped, axis=(2, 3, 4), dtype=np.float64)


def _brightness_record(data, para, stage, avgg, wavelength, time_start, time_end):
    return {
        'stage': stage,
        'avgg': int(avgg),
        'wavelength': wavelength,
        'time_start': int(time_start),
        'time_end': int(time_end),
        'crop_shape': tuple(int(value) for value in data.shape[-2:]),
        'phase_n': int(para['zhenshu'] // 3),
        'brightness': calculate_brightness_matrix(
            data, para['region_n'], para['zhenshu']
        ),
    }


def collect_brightness_records(data, para, func_f, stage):
    func = _get_sorting_function(func_f)
    records = []
    for avgg in para['AVGG']:
        if para['point'] % avgg != 0:
            raise ValueError('point 与AVGG不是整数关系')
        averaged = []
        time_point = 0
        wavelength_index = 0
        index = 0
        while index < para['point']:
            stack = reorder_n(
                data,
                para['zhenshu_sep'],
                para['zhenshu'],
                nth=index,
                func=func,
                region_n=para['region_n'],
            )
            averaged.append(stack)
            if (index + 1) % avgg == 0:
                stack_mean = np.mean(np.asarray(averaged), axis=0)
                averaged = []
                np.maximum(stack_mean, 0, out=stack_mean)
                cropped = get_final_cropped_data(stack_mean, para)
                records.append(
                    _brightness_record(
                        cropped,
                        para,
                        stage,
                        avgg,
                        para['W_order'][wavelength_index],
                        para['points_name'] + index - avgg + 1,
                        para['points_name'] + index,
                    )
                )
                if (
                    para['zhenshu_sep'] == 15
                    and para['zhenshu'] == 15
                    and func_f == 123
                ):
                    time_point += 15
                    if time_point == 30:
                        wavelength_index = (wavelength_index + 1) % 2
                        time_point = 0
            if func_f == 3015:
                index += 1
            index += 1
    return records


def _safe_normalize(values, axis):
    denominator = np.mean(values, axis=axis, keepdims=True)
    return np.divide(
        values,
        denominator,
        out=np.zeros_like(values, dtype=np.float64),
        where=denominator != 0,
    )


def _plot_brightness_group(records, figure_path, title):
    values = np.stack([record['brightness'] for record in records])
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)
    region_normalized = _safe_normalize(values, axis=1)
    direction_normalized = _safe_normalize(values, axis=2)
    region_norm_mean = np.mean(region_normalized, axis=0)
    region_norm_std = np.std(region_normalized, axis=0)
    direction_norm_mean = np.mean(direction_normalized, axis=0)
    direction_norm_std = np.std(direction_normalized, axis=0)

    region_x = np.arange(1, 9)
    direction_x = np.arange(1, 4)
    colors = plt.get_cmap('tab10').colors
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    for direction in range(3):
        color = colors[direction]
        axes[0, 0].plot(
            region_x, mean_values[:, direction], marker='o', color=color,
            label=f'Direction {direction + 1}',
        )
        axes[0, 0].fill_between(
            region_x,
            mean_values[:, direction] - std_values[:, direction],
            mean_values[:, direction] + std_values[:, direction],
            color=color,
            alpha=0.15,
        )
        axes[1, 0].plot(
            region_x, region_norm_mean[:, direction], marker='o', color=color,
            label=f'Direction {direction + 1}',
        )
        axes[1, 0].fill_between(
            region_x,
            region_norm_mean[:, direction] - region_norm_std[:, direction],
            region_norm_mean[:, direction] + region_norm_std[:, direction],
            color=color,
            alpha=0.15,
        )
        abnormal = np.abs(region_norm_mean[:, direction] - 1) > BRIGHTNESS_WARNING_FRACTION
        axes[1, 0].scatter(
            region_x[abnormal], region_norm_mean[abnormal, direction],
            color='red', zorder=5,
        )

    for region in range(8):
        color = colors[region]
        axes[0, 1].plot(
            direction_x, mean_values[region], marker='o', color=color,
            label=f'R{region + 1}',
        )
        axes[0, 1].fill_between(
            direction_x,
            mean_values[region] - std_values[region],
            mean_values[region] + std_values[region],
            color=color,
            alpha=0.08,
        )
        axes[1, 1].plot(
            direction_x, direction_norm_mean[region], marker='o', color=color,
            label=f'R{region + 1}',
        )
        axes[1, 1].fill_between(
            direction_x,
            direction_norm_mean[region] - direction_norm_std[region],
            direction_norm_mean[region] + direction_norm_std[region],
            color=color,
            alpha=0.08,
        )
        abnormal = np.abs(direction_norm_mean[region] - 1) > BRIGHTNESS_WARNING_FRACTION
        axes[1, 1].scatter(
            direction_x[abnormal], direction_norm_mean[region, abnormal],
            color='red', zorder=5,
        )

    axes[0, 0].set_title('Raw brightness by region')
    axes[0, 1].set_title('Raw brightness by direction')
    axes[1, 0].set_title('Region-normalized brightness')
    axes[1, 1].set_title('Direction-normalized brightness')
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel('Region' if axis in axes[:, 0] else 'Direction')
        axis.set_ylabel('Mean intensity' if axis in axes[0] else 'Relative brightness')
    axes[1, 0].axhspan(0.8, 1.2, color='green', alpha=0.08)
    axes[1, 1].axhspan(0.8, 1.2, color='green', alpha=0.08)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 1].legend(ncol=2, fontsize=8)
    axes[1, 0].legend(ncol=3, fontsize=8)
    axes[1, 1].legend(ncol=2, fontsize=8)
    figure.suptitle(title, fontsize=11)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    return {
        'mean': mean_values,
        'std': std_values,
        'region_normalized_mean': region_norm_mean,
        'direction_normalized_mean': direction_norm_mean,
    }


def _append_brightness_markdown(report_path, records, figure_path, summary, title):
    report_path = Path(report_path)
    if not report_path.exists():
        report_path.write_text('# 背景扣除与亮度分析\n', encoding='utf-8')
    relative_figure = figure_path.relative_to(report_path.parent).as_posix()
    first = records[0]
    time_start = min(record['time_start'] for record in records)
    time_end = max(record['time_end'] for record in records)
    crop_shapes = sorted(set(record['crop_shape'] for record in records))
    lines = [
        '',
        f'## {title}',
        '',
        f"- 记录时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据来源：{'几何配准后、亮度校正前' if first['stage'] == 'before-correction' else '最终保存裁剪结果'}",
        f"- 时间范围：{time_start}–{time_end}，统计样本数：{len(records)}",
        f"- 实际裁剪尺寸：{crop_shapes}",
        f"- 方向/相位映射：3方向 × {first['phase_n']}相位",
        '',
        '### 原始平均灰度（时间均值 ± 标准差）',
        '',
        '| 区域 | 方向1 | 方向2 | 方向3 |',
        '|---|---:|---:|---:|',
    ]
    for region in range(8):
        cells = [
            f"{summary['mean'][region, direction]:.3f} ± "
            f"{summary['std'][region, direction]:.3f}"
            for direction in range(3)
        ]
        lines.append(f"| R{region + 1} | {' | '.join(cells)} |")

    lines.extend([
        '',
        '### 每个方向内的区域归一化亮度',
        '',
        '| 区域 | 方向1 | 方向2 | 方向3 |',
        '|---|---:|---:|---:|',
    ])
    region_warnings = []
    for region in range(8):
        cells = []
        for direction in range(3):
            value = summary['region_normalized_mean'][region, direction]
            cells.append(f'{value:.3f}')
            if abs(value - 1) > BRIGHTNESS_WARNING_FRACTION:
                region_warnings.append(
                    f'R{region + 1}/方向{direction + 1}: {(value - 1):+.1%}'
                )
        lines.append(f"| R{region + 1} | {' | '.join(cells)} |")

    direction_means = np.mean(summary['mean'], axis=0)
    direction_relative = direction_means / np.mean(direction_means)
    direction_warnings = [
        f'方向{direction + 1}: {(value - 1):+.1%}'
        for direction, value in enumerate(direction_relative)
        if abs(value - 1) > BRIGHTNESS_WARNING_FRACTION
    ]
    warnings = region_warnings + direction_warnings
    lines.extend([
        '',
        '### 异常检查',
        '',
        '- 判据：相对平均亮度偏离超过 ±20%，仅警告、不停止。',
        f"- 结果：{'; '.join(warnings) if warnings else '未发现超过阈值的亮度差异。'}",
        '',
        f'![{title}]({relative_figure})',
        '',
    ])
    with report_path.open('a', encoding='utf-8') as file:
        file.write('\n'.join(lines))


def save_brightness_analysis(before_records, after_records, para):
    report_path = _background_stats_path(para)
    figure_dir = report_path.parent / 'brightness-analysis'
    figure_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    all_records = before_records + after_records
    groups = {}
    for record in all_records:
        key = (record['stage'], record['avgg'], str(record['wavelength']))
        groups.setdefault(key, []).append(record)

    for (stage, avgg, wavelength), records in sorted(groups.items()):
        crop_shapes = sorted(set(record['crop_shape'] for record in records))
        title = (
            f"{para['data_name']} / {para['dir_name']} | {stage} | "
            f'wavelength={wavelength} nm | AVGG={avgg} | '
            f'timepoints={len(records)} | crop={crop_shapes} | '
            f"8 regions x 3 directions x {records[0]['phase_n']} phases | "
            f'run={run_id}'
        )
        figure_path = figure_dir / (
            f'brightness-{run_id}-{stage}-{wavelength}nm-AVGG{int(avgg):02d}.png'
        )
        summary = _plot_brightness_group(records, figure_path, title)
        _append_brightness_markdown(
            report_path, records, figure_path, summary, title
        )
    print(f'亮度分析已保存: {report_path}')


#样本后处理
#对配准好的图片进行后续处理
def get_rawdata(data,para,func_f,file_suffix='',brightness_before_records=None):
    '''
    zhenshu_sep: 从多少相位里
    zhenshu: 提取出多少帧正常情况是一样的

    '''
    for jj in para['AVGG']:
        if para['point']%jj!=0:  #不是整数倍
            raise ValueError("point 与AVGG不是整数关系")
    if func_f ==2121: #4+3
        print('21帧-提取')
        print('zhenshu  21 == ',para['zhenshu'])
        func = sorting21_21
    if func_f ==212152:#5+2
        print('21帧-5-2提取')
        print('zhenshu  21 == ',para['zhenshu'])
        func = sorting21_21_52
    elif func_f==2115:
        print('15帧-提取')
        # print('zhenshu_sep=',zhenshu_sep)
        print('zhenshu  15 == ',para['zhenshu'])
        func = sorting21_15
    elif func_f==219:
        func = sorting21_9
    elif func_f == 123:
        func = sorting_series
    elif func_f == 3015:
        func = sorting30_52
    para['func_f']=func_f
    post_brightness_records = []

    Dr_stack_set = []
    _,nz,_,_ = data.shape
    dtypee = data.dtype

    print(nz)
    point = (nz)//para['zhenshu_sep']  #全部的时间点
    print('all point=',point)


    # #从开头取point个时间点
    # if point!=para['point']:
    #     points = para['point']
    #     s = 0                           #int(np.ceil(point-para['point'])/2)*para['zhenshu_sep']  #优先向上取整    #从中间取point个时间点  -不行
    #     e = points*para['zhenshu_sep'] #s+para['point']*para['zhenshu_sep']
    #     print(f"s={s},e={e}")
    #     print(f'取{points} point')
    #     imagS_M2 = data[:,s:e]
    # else:
    # print('取全部时间点')
    # imagS_M2 = data
    # del data
    # _U.save_tif('imagS_M2.tif',imagS_M2)
    # print(f'imagS_M2.shape={imagS_M2.shape}')

    for jj in para['AVGG']:     #[1,10]
        #AVG or not avg
        AVG_contan = []
        para ['nth_total'] =  para['points_name'] #记录这个数据的时间点 #开始从1开始

        # for ii in range(para['point']):                 #控制取多少个时间点
        ii = 0
        time_point = 0 # 0 15 30 记录 MP模式 大时间点
        w_i = 0 # [488,561]    0,1,0,1  #记录波长名字
        while(ii<para['point']):
            print("II=",ii)
            #把不同区域 ii时间点的raw data 按照 func 规则提取出来，并组合
            temp = reorder_n(data,para['zhenshu_sep'],para['zhenshu'],nth= ii,func=func,region_n=para['region_n']) #每个时间点

            ######时间点之间的漂白校正处理###### 不进行校正,放在了重建之后
            # if ii==0:         ###首个时间点
            #     # temp0 = np.copy(temp)
            #     recoder = Bleachcorrection_0(temp)
            # else: 
            #     temp = Bleach_main(temp,recoder) #漂白校正功能
            ################################
        
            ######不同方向之间的校正######## 放在了外面
            # temp = Ori3_adjust(temp) #7,7,7 / 7,7,7
                
            
            AVG_contan.append(temp)     #放入这个时间点，用来后面平均用
            
            if (ii+1)%jj == 0:   #可以平均了
                AVG_data = np.array(AVG_contan,dtype=dtypee)
                AVG_contan = []
                nc,nz,nx,ny = AVG_data.shape
                Dr_stack_mean = np.mean( AVG_data , axis=0 )  #时间点上的平均
                del AVG_data
                Dr_stack_mean[Dr_stack_mean<0] = 0

                Dr_stack_cai = save_Dr(Dr_stack_mean,para,AVGG=jj,file_suffix=file_suffix,w_i=w_i)
                post_brightness_records.append(
                    _brightness_record(
                        Dr_stack_cai,
                        para,
                        'after-correction',
                        jj,
                        para['W_order'][w_i],
                        para['points_name'] + ii - jj + 1,
                        para['points_name'] + ii,
                    )
                )
                
                #多一个双色名字命名，
                if para['zhenshu_sep']==15 and para['zhenshu']==15 and  para['func_f']==123:   #MP 5+5 123模式
                    time_point +=15  #
                    if time_point==30:
                        w_i = (w_i+1)%2  # 
                        time_point = 0  #重新置0 
                # break

            #保存WF
            temp[temp<0]=0
            save_WF(temp,para)
            para ['nth_total'] +=1
            
            if func_f == 3015:  #
                ii+=1
            ii+=1
    save_brightness_analysis(
        brightness_before_records or [], post_brightness_records, para
    )
    return Dr_stack_cai #返回最后一个时间点
    



def save_WF(temp,para):
    import Data_Registration_MP
    #裁剪
    nz,nx,ny = temp.shape
    # width_nx = para['nxny']//2   #450//2
    # width_ny = para['nyny']//2
    # Dr_stack_cai = temp[:,nx//2-width_nx:nx//2+width_nx,ny//2-width_ny:ny//2+width_ny]
    Dr_stack_cai = get_final_cropped_data(temp, para)
    nth = para['nth_total']
    nth3 = str(nth).zfill(3)
    data_name = para['data_name']
    dir_name = para['dir_name']
    
    WF,WF2_rcai,WF3 = Data_Registration_MP.getWF(Dr_stack_cai,para)
    _U.save_tif_v21(f'WF1-R-{WF.shape}-{nth3}.tif',WF,para,sub_filename="WF",subsub_filename =str(data_name)+'_'+str(dir_name)+"_"+str(WF.shape) )
    if nth%10==0 or nth==1:
        _U.save_tif_v21(f'WF2-R-{WF2_rcai.shape}-{nth3}.tif',WF2_rcai,para,sub_filename="WF",subsub_filename =str(data_name)+'_'+str(dir_name)+"_"+str(WF2_rcai.shape)+'-Frequence')
        _U.save_tif_v21(f'WF3-R-{WF2_rcai.shape}-{nth3}.tif',WF3,para,sub_filename="WF",subsub_filename =str(data_name)+'_'+str(dir_name)+"_"+str(WF2_rcai.shape)+'-real')
    # if para['cell']=='FIX':
    #     _U.save_tif_v21(f'WF4-R-{WF4_rcai.shape}-{nth3}.tif',WF4_rcai,para,sub_filename="WF",subsub_filename =str(data_name)+'_'+str(dir_name)+"_"+str(WF4_rcai.shape))
    return
    

def save_Dr(temp,para,AVGG,file_suffix='',w_i=0):
    nz,nx,ny = temp.shape
    # width_nx = para['nxny']//2   #450//2
    # width_ny = para['nyny']//2
    # Dr_stack_cai = temp[:,nx//2-width_nx:nx//2+width_nx,ny//2-width_ny:ny//2+width_ny]#从中间取
    Dr_stack_cai = get_final_cropped_data(temp, para)
                                                                          #

    # AVGG = para['AVGG']
    data_name = para['data_name']
    dir_name = para['dir_name']
    zhenshu = para['zhenshu']
    nth = para['nth_total']
    func_f = para['func_f']
    BF = para['BF_kongzhi']
    W_order = para['W_order'][w_i]  #
    

    nth = str(nth).zfill(4) #时间点后缀，长时间的最好用三个长度记录
    AVGG= str(AVGG).zfill(2)#

    # if para['zhenshu'] == 21:
        
    #     _U.save_tif_v21(f'Dr_stack-{data_name}日{dir_name}-{func_f}-{zhenshu}帧-mp--[{AVGG}]_{nth}.tif', Dr_stack_cai ,para,f'Dr\\{AVGG}\\')
    
    # dr_name_suffix: '' → Dr/01-488；'_regOnly' → Dr/01-488_regOnly（仅配准轨）
    dr_suffix = str(para.get('dr_name_suffix', '') or '')
    dr_sub = f'{AVGG}-{W_order}{dr_suffix}'
    _U.save_tif_v21(
        f'Dr_stack-{data_name}日{dir_name}-{func_f}-{zhenshu}帧-mp-BF={BF}-[{AVGG}]_{nth}.tif',
        Dr_stack_cai,
        para,
        f'Dr\\{dr_sub}\\',
    )
    
    # elif para['zhenshu'] == 15:
    #     # _U.save_tif(f'Dr_stack-{pp_name}日-{AVGG}-{pp_name2}-{func_f}-{zhenshu}帧-sli-{nth}.tif',Dr_stack_cai )
    #     _U.save_tif_v21(f'Dr_stack-{pp_name}日-{AVGG}-{pp_name2}-{func_f}-{zhenshu}帧-sli-{nth}.tif',Dr_stack_cai ,para)

    # elif para['zhenshu'] == 9:
    #     for r_n in range(para['region_n']):    #2d的情况，按照区域保存
    #         # _U.save_tif(f'Dr_stack-{pp_name}日-{AVGG}-{pp_name2}-{func_f}-{zhenshu}帧-2d-{r_n+1}区域-{nth}.tif',Dr_stack_cai[r_n*zhenshu:(r_n+1)*zhenshu])
    #         _U.save_tif_v21(f'Dr_stack-{pp_name}日-{AVGG}-{pp_name2}-{func_f}-{zhenshu}帧-2d-{r_n+1}区域-{nth}.tif',Dr_stack_cai[r_n*zhenshu:(r_n+1)*zhenshu],para)

    #     # _U.save_tif(f'Dr_stack-{pp_name}日-{AVGG}-{pp_name2}-{func_f}-{zhenshu}帧-2d-{r_n+1}区域-all.tif',Dr_stack_cai)
    #     _U.save_tif_v21(f'Dr_stack-{pp_name}日-{AVGG}-{pp_name2}-{func_f}-{zhenshu}帧-2d-{r_n+1}区域-all.tif',Dr_stack_cai,para)

    return Dr_stack_cai

#公共函数
def reorder_n(imag_1_M,zhenshu_sep,zhenshu,nth,func,region_n=8):
    '''
    zhenshu_sep： 拍图时的间隔
    zhenshu：需要提取的帧数
    nth：不同时间点or位置帧
    '''
    _,_,nx,ny = imag_1_M.shape   #105,nx,ny
    dtypee = imag_1_M.dtype
    Dr_stack = np.zeros(( zhenshu * region_n,nx,ny ),dtype=dtypee)  #八个区域

    for i in range(region_n):
        Dr_stack[i*zhenshu:(i+1)*zhenshu] = func(imag_1_M[i],zhenshu_sep,zhenshu,nth)

    return Dr_stack


def sorting30_52(imag_i_M,zhenshu_sep,zhenshu,nth):
    '''
    zhenshu_sep = 15 
    zhenshu = 21
    '''
    print('01234 01234 01234 |01234 01234 01234  ->01234 02,')
    nz,nx,ny = imag_i_M.shape   #nz,nx,ny
    dtypee = imag_i_M.dtype
    Dr = np.zeros((zhenshu,nx,ny ),dtype=dtypee)  #需要多少张相位
        
    if nth==0:
        z_s = 0   #
    else:
        z_s = nth*zhenshu_sep #
    #1
    i=0
    Dr[i:i+5] = imag_i_M[z_s:z_s+5]
    Dr[i+5:i+7] = imag_i_M[[z_s+15,z_s+15+2]]
    #n.save_tif('Dr[4-7].tif',Dr)
    #2
    i=i+7
    Dr[i:i+5] = imag_i_M[z_s+5:z_s+5+5]
    Dr[i+5:i+7] = imag_i_M[[z_s+15+5,z_s+15+5+2]]
    #n.save_tif('Dr[11-14]',Dr)
    #3
    i=i+7
    Dr[i:i+5] = imag_i_M[z_s+10:z_s+10+5]
    Dr[i+5:i+7] = imag_i_M[[z_s+15+10,z_s+15+10+2]]
    #n.save_tif('Dr[18-21].tif',Dr)
    return Dr


def sorting21_21(imag_i_M,zhenshu_sep,zhenshu,nth):
    '''
    三个方向，每个方向7帧 共21帧
    '''
    print('0123 0123 0123 ABCABCABC->0123ABC,')
    nz,nx,ny = imag_i_M.shape   #nz,nx,ny
    dtypee = imag_i_M.dtype
    Dr = np.zeros((zhenshu,nx,ny ),dtype=dtypee)  #需要多少张相位
        
    if nth==0:
        z_s = 0   #
    else:
        z_s = nth*zhenshu_sep #
    #1
    i=0
    Dr[i:i+4] = imag_i_M[z_s:z_s+4]
    Dr[i+4:i+7] = imag_i_M[z_s+12:z_s+12+3]
    #n.save_tif('Dr[4-7].tif',Dr)
    #2
    i=i+7
    Dr[i:i+4] = imag_i_M[z_s+4:z_s+4+4]
    Dr[i+4:i+7] = imag_i_M[z_s+12+3:z_s+12+3+3]
    #n.save_tif('Dr[11-14]',Dr)
    #3
    i=i+7
    Dr[i:i+4] = imag_i_M[z_s+8:z_s+8+4]
    Dr[i+4:i+7] = imag_i_M[z_s+12+6:z_s+12+6+3]
    #n.save_tif('Dr[18-21].tif',Dr)
    return Dr


def sorting21_21_52(imag_i_M,zhenshu_sep,zhenshu,nth):
    '''
    三个方向，每个方向7帧 共21帧
    '''
    print('01234 01234 01234 ABABAB->01234AB,')
    nz,nx,ny = imag_i_M.shape   #nz,nx,ny
    dtypee = imag_i_M.dtype
    Dr = np.zeros((zhenshu,nx,ny ),dtype=dtypee)  #需要多少张相位
        
    if nth==0:
        z_s = 0   #
    else:
        z_s = nth*zhenshu_sep #
    #1
    i=0
    Dr[i:i+5] = imag_i_M[z_s:z_s+5]
    Dr[i+5:i+7] = imag_i_M[z_s+15:z_s+15+2]
    #n.save_tif('Dr[4-7].tif',Dr)
    #2
    i=i+7
    Dr[i:i+5] = imag_i_M[z_s+5:z_s+5+5]
    Dr[i+5:i+7] = imag_i_M[z_s+15+2:z_s+15+2+2]
    #n.save_tif('Dr[11-14]',Dr)
    #3
    i=i+7
    Dr[i:i+5] = imag_i_M[z_s+10:z_s+10+5]
    Dr[i+5:i+7] = imag_i_M[z_s+15+4:z_s+15+4+2]
    #n.save_tif('Dr[18-21].tif',Dr)
    return Dr

def sorting21_15(imag_i_M,zhenshu_sep,zhenshu,nth):
    
    '''
    三个方向，每个方向5帧 共15帧   02461
    
    '''

    nz,nx,ny = imag_i_M.shape   #nz,nx,ny
    Dr = np.zeros((zhenshu,nx,ny ))  #需要多少张相位
        
    if nth==0:
        z_s = 0   #
    else:
        z_s = nth*zhenshu_sep #
    #1
    i=0
    Dr[i:i+4] = imag_i_M[z_s:z_s+4]
    Dr[i+4:i+5] = imag_i_M[z_s+12:z_s+12+1] #这里的12
    #n.save_tif('Dr[4-7].tif',Dr)
    #2
    i=i+5
    Dr[i:i+4] = imag_i_M[z_s+4:z_s+4+4]
    Dr[i+4:i+5] = imag_i_M[z_s+12+3:z_s+12+3+1] #这里的12 ，3 均为超参数
    #n.save_tif('Dr[11-14]',Dr)
    #3
    i=i+5
    Dr[i:i+4] = imag_i_M[z_s+8:z_s+8+4]
    Dr[i+4:i+5] = imag_i_M[z_s+12+6:z_s+12+6+1]
    #n.save_tif('Dr[18-21].tif',Dr)
    return Dr

def sorting21_9(imag_i_M,zhenshu_sep,zhenshu,nth):
    print('21帧中 2D提取 0 1 2')
    nz,nx,ny = imag_i_M.shape   #nz,nx,ny
    Dr = np.zeros((zhenshu,nx,ny ))  #需要多少张相位
    z_s = nth*zhenshu_sep            #拍图的时候
    print('z_s=',z_s)
    #1
    i=0
    Dr[i:i+3] = imag_i_M[z_s:z_s+3]
    #2
    i=i+3
    Dr[i:i+3] = imag_i_M[z_s+4:z_s+4+3]
    #3
    i=i+3
    Dr[i:i+3] = imag_i_M[z_s+8:z_s+8+3]
    return Dr

def sorting_series(imag_i_M,zhenshu_sep,zhenshu,nth):
    print('连续提取帧，请设置好参数')
    print('zhenshu_sep=',zhenshu_sep)
    print('zhenshu=',zhenshu)
    nz,nx,ny = imag_i_M.shape   #nz,nx,ny
    Dr = np.zeros((zhenshu,nx,ny ))  #需要多少张相位
    z_s = nth*zhenshu_sep            #拍图的时候
    print('z_s=',z_s)
    Dr = imag_i_M[z_s:z_s+zhenshu]
    return Dr


#####漂白功能########
def MaxMinN(data):
    minn = np.min(data)
    maxx = np.max(data)
    data = (data-minn)/(maxx-minn)
    return data

# def Bleachcorrection_0(data):
#     recoder = []
#     for i in range(data.shape[0]):
#         temp = np.sum(data[i])
#         recoder.append(temp)     #记录第一个时间点的每个nc value
#     return recoder,np.array(data)

def Bleachcorrection_0(data):
    assert len(data.shape)==3          #(21,nx,ny)
    recoder = np.mean(data,axis=(1,2)) #记录第一个时间点的每个nc value
    return recoder 

# def Bleachcorrection(data,recoder):
#     data_correc = np.zeros_like(data)
#     for i in range(data.shape[0]):
#         temp  = np.sum(data[i])
#         rate = recoder[i]/temp
#         data_correc[i] = data[i] * rate
#     return data_correc

def Bleachcorrection(data,recoder):
    data_correc = np.zeros_like(data)
    recoder2 = np.mean(data,axis=(1,2))
    rate = recoder/recoder2           #首/后续
    rate = rate.reshape(-1,1,1)
    data_correc = data*rate 
    return data_correc
    
    # for i in range(data.shape[0]):
    #     temp  = np.sum(data[i])
    #     rate = recoder[i]/temp
    #     data_correc[i] = data[i] * rate
    # return data_correc

def Bleach_main(temp,recoder):
    #增加漂白校正功能
    temp_Bleachcorrec = Bleachcorrection(temp,recoder) #然后校正其他数据
    return temp_Bleachcorrec
