# import BaseFunctions_np_7_1 as n
import sys 
import re
# sys.path.append('../')  #添加上一层目录

import Utils as _U
import Cam_process as ca 
# import pocpy
import poc as pocpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

def get_registration_projection_method(pro_meth, registration):
    if pro_meth is None:
        return 'tenengrad' if registration == 3 else 'mean'
    return pro_meth

def decentralise(cam12_4d):
    #
    region_n,nz,nx,ny = cam12_4d.shape #(8,nz,nx,ny)
    assert region_n==8
    imag_4 = cam12_4d[0]
    imag_3 = cam12_4d[1]
    imag_2 = cam12_4d[2]
    imag_1 = cam12_4d[3]
    
    imag_5 = cam12_4d[4]
    imag_6 = cam12_4d[5]
    imag_7 = cam12_4d[6]
    imag_8 = cam12_4d[7]

    return np.array([imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8])

def decentralise45(cam12_4d):
    #
    region_n,nz,nx,ny = cam12_4d.shape #(8,nz,nx,ny)
    assert region_n==8
    imag_4 = cam12_4d[0]
    # imag_2 = cam12_4d[1]
    # imag_3 = cam12_4d[2]
    # imag_4 = cam12_4d[3]
    
    imag_5 = cam12_4d[4]
    # imag_6 = cam12_4d[5]
    # imag_7 = cam12_4d[6]
    # imag_8 = cam12_4d[7]

    return imag_4,imag_5


# ==================== mean scale ====================
def _scale_project_mean_to_base(project, base):
    project_scale = project.astype(np.float32, copy=True)
    project_mean = np.mean(project_scale, axis=(1, 2))
    base_mean = project_mean[base]
    scale = np.ones(project_mean.shape, dtype=np.float32)

    if base_mean == 0:
        print('base区域均值为0，不做均值缩放')
        return project_scale, project_mean, scale

    for i in range(project_scale.shape[0]):
        if project_mean[i] != 0:
            scale[i] = base_mean / project_mean[i]
            project_scale[i] = project_scale[i] * scale[i]

    print(f'base区域: {base}, base_mean: {base_mean:.4f}')
    print(f'缩放前各区域均值: {project_mean.round(4)}')
    print(f'各区域缩放系数: {scale.round(4)}')
    print(f'缩放后各区域均值: {np.mean(project_scale, axis=(1, 2)).round(4)}')
    return project_scale, project_mean, scale


def _save_focus_spacing_markdown(focus_record, spacing, file_suffix, para):
    report_path = _U.sift_analysis_report_path(para)
    source_layers = focus_record['source_imagej_layer_1based']
    spacing_limit = spacing['spacing_limit_layers']
    invalid_pairs = set(spacing['invalid_pair_indices_0based'])

    lines = [
        '## Tenengrad焦面层间隔检查',
        '',
        f"- 记录时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 输入栈层数：{spacing['stack_layers']}",
        f"- 焦面层（原始ImageJ 1-based）：{source_layers}",
        f"- 状态：{spacing['status']}",
        '',
        '| 区域对 | 前一区域焦面 | 后一区域焦面 | 间隔层数 | 允许上限 | 结果 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for pair_index, gap in enumerate(spacing['adjacent_gaps_layers']):
        result = '异常' if pair_index in invalid_pairs else '正常'
        limit_text = str(spacing_limit) if spacing_limit is not None else '不支持'
        lines.append(
            f'| R{pair_index + 1}→R{pair_index + 2} | '
            f'{source_layers[pair_index]} | {source_layers[pair_index + 1]} | '
            f'{gap} | {limit_text} | {result} |'
        )
    if report_path.exists():
        report_text = report_path.read_text(encoding='utf-8')
    else:
        report_text = f"# SIFT分析报告 - {para['peizhun_file']}\n"
    report_text = re.sub(
        r'\n## Tenengrad焦面层间隔检查\n.*?(?=\n## |\Z)',
        '',
        report_text,
        flags=re.DOTALL,
    )
    with report_path.open('w', encoding='utf-8') as file:
        file.write(report_text.rstrip() + '\n\n')
        file.write('\n'.join(lines) + '\n')
    print(f'Tenengrad焦面层间隔报告已保存: {report_path}')
    return report_path


def _record_focus_spacing_warning(spacing, focus_path, para):
    if spacing['status'] != 'invalid_spacing':
        return None
    focus_path = _U.mark_quality_report_path(focus_path, 'foc！')
    _U.record_quality_report_marker(para, 'foc！')
    invalid_pairs = [
        f'R{index + 1}→R{index + 2}'
        for index in spacing['invalid_pair_indices_0based']
    ]
    return _U.record_quality_warning(
        para,
        f'Tenengrad焦面层间隔异常: {invalid_pairs}，'
        f"允许上限为{spacing['spacing_limit_layers']}层；"
        f'流程已继续，报告见{focus_path}',
    )


def calRegistra(cam12_4d,pro_meth=None,Registration=1, pgxy=0,theta =0,scale =0,Regster_f=0,Hs=0,file_suffix='_',stand=4,ratio_thresh=1,para=None,focus_source_layer_offset=1):
    Debug= np.array([0,0,0,0,0,0,0,0,0])
    pro_meth = get_registration_projection_method(pro_meth, Registration)
    file_suffix = str(file_suffix)+pro_meth+'_1'
    imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8 = decentralise(cam12_4d)
    # imag_1,imag_5 = decentralise(cam1_4d[0],cam2_4d[0])
    # return imag_1,imag_5
    

    print(imag_1.shape)
    
    nx = imag_1.shape[1]
    #555⑤ 投影得到project(8,nx,ny)
    # project = ca.get_project(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,method='max',Debug=Debug[5])
    
    if Registration==0:
        ###########################################################################
        #配准方法一
        #5566得到project后，进行互相关,计算坐标差值
        coordinate =  ca.cal_corrlate(project,base=4) # 4这个表示以谁为基准
        coordinate3 = (coordinate-512+1)
        #coordinate3[0] = 0
        print('配准为:',coordinate3)

        #将project 做一个移动 得到project_match
        project_match = ca.project_pipei(project,coordinate3)  #!这是个很重要的配准信息， 进行了一个整数级的配准 

        #保存下来看看（验证）
        _U.save_tif_v21(f'5566 第一种配准法project_match-【1~8】.tif',project_match,para)
        ###########################################################################
        #利用配准后的再来做一次拟合 验证
        coordinate2 =  ca.cal_corrlate(project_match)
        coordinate2 = coordinate2-512+1
        coordinate2[0] = 0
        print('再做一次配准coordinate2=',coordinate2) 
    ##############################################

    if Registration==1:
            #配准方法二
        print('【第二种配准方法】')
        print(f'input:\n {pgxy.round(3)}')
        print(f'input:\n {theta.round(3)}')
        print(f'input:\n {scale.round(3)}')
        for loop in range(2):
            #1.根据参数，移动得到，M （注意一定是要image 通过pgxy 移动）
            imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,theta,scale,pgxy)

            #2.根据M得到投影projection 
            project_M = ca.get_project( imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,method=pro_meth,Debug=Debug[5])
            nz,nx,ny = project_M.shape
            width = 512//2#430//2#512//2
            project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
            _U.save_tif_v21(f'project-8区域-{loop+1}次迭代配准-{file_suffix}.tif',project_M_c,para)

            #3.根据投影计算当前需要移动的坐标
            pgxy_sub ,theta_sub,scale_sub = Method_pocpy(project_M ,stand=stand)

            c=np.sum(abs(pgxy_sub)>=1)
            print(f'【{loop+1}】pgxy_sub=\n{pgxy_sub.round(2)}')
            
            #pgxy 可累加，其他不变
            if pgxy.any() ==0:
                print('初始化')
                pgxy,theta,scale =  pgxy_sub ,theta_sub,scale_sub
            elif c< 16 :
                #选择性更新 其中小于1的亚像素移动
                yuzhi = 5
                index = np.where(abs(pgxy_sub)<yuzhi) #只更新其中<4的情况
                array12 = np.bincount(index[0])   #统计一行中 xy都小于1的情况
                index2 = np.where(array12==2)
                print(f'只更新其中<{yuzhi}的情况',index2)
                for index3 in index2:
                    print(f'{pgxy_sub[index3].round(2)}')
                    pgxy[index3] +=pgxy_sub[index3]
                # print('theta 更新',theta_sub)
                # theta = theta_sub
                # print('scale 更新',scale_sub)
                # scale = scale_sub

            else:
                print('不更新')
                pass


        if Regster_f==0:
            return pgxy,theta,scale,imag_5
        else:
            #到了这一步 前后两帧之间差距很小了，且对于MP来说，前后两帧信息量更多
            #增加一个前后帧对齐
            print('scale',scale)
            print('theta',theta)
            print()
            print(f'‘pgxy修正后’：{pgxy[index3].round(3)}')
            #缩小范围
            nz,nx,ny = project_M.shape  # 8,nx,ny
            width = 430//2#512//2
            project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]

            wavelet = [0,1,2,3]   #离中心的距离   
            for w in wavelet:
                print(f'{w}距离为中心，更新其他')
                centro_r = stand+w
                centro_l = stand-w
                #重新定stand
                yuzhi = 2
                #right
                if centro_r<=6:
                    pgxy_sub ,theta_sub,scale_sub = Method_pocpy(project_M_c,quyuR=[centro_r,centro_r+1],stand=centro_r)
                    print(f'【{centro_r+1}】pgxy_sub=\n{pgxy_sub[1].round(3)}')
                    #最新最后一组
                    if abs(pgxy_sub[1][0])<yuzhi and abs(pgxy_sub[1][1])<yuzhi: 
                        pgxy[centro_r+1] = pgxy[centro_r+1]+pgxy_sub[1]
                    else:
                        print('不更新')

                #left
                if centro_l>=1:
                    pgxy_sub ,theta_sub,scale_sub = Method_pocpy(project_M_c,quyuR=[centro_l,centro_l-1],stand=centro_l)
                    print(f'【{centro_l-1}】pgxy_sub=\n{pgxy_sub[1].round(3)}')
                    if abs(pgxy_sub[1][0])<yuzhi and abs(pgxy_sub[1][1])<yuzhi: 
                        pgxy[centro_l-1] = pgxy[centro_l-1]+pgxy_sub[1]
                    else:
                        print('不更新')

                #利用更新后的pgxy 重新移动
                imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,theta,scale,pgxy)
                
                #查看投影
                project_M = ca.get_project( imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,method=pro_meth,Debug=Debug[5])
                project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
                # _U.save_tif(f'project_M-8区域-{w}-wave配准-{file_suffix}.tif',project_M_c)
            _U.save_tif_v21(f'project_M-8区域-{w}-wave配准-{file_suffix}.tif',project_M_c,para)
            return imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M

    #################################################
    
    if Registration == 3:   #使用SIFT进行配准
        #  计算特征点提取&生成描述时间
        sift = cv2.SIFT_create()
        #  使用SIFT查找关键点key points和描述符descriptors
        project_8bit = []
        kp_list = []
        des_list = []
        # imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,theta,scale,pgxy)
        # project = ca.get_project( imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,method=pro_meth,Debug=Debug[5])
        images = np.array([imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8])
        if pro_meth == 'tenengrad':
            project, focus_indices, focus_scores = ca.get_tenengrad_project(images)
            focus_record = {
                'method': pro_meth,
                'code_region_order': 'region 1..8',
                'input_stack_index_0based': focus_indices.tolist(),
                'input_stack_layer_1based': (focus_indices + 1).tolist(),
                'source_imagej_layer_1based': (
                    focus_indices + focus_source_layer_offset
                ).tolist(),
                'peak_scores': focus_scores[np.arange(8), focus_indices].round(6).tolist(),
            }
            print(f"Tenengrad焦面层(输入栈1-based): {focus_record['input_stack_layer_1based']}")
            print(
                'Tenengrad焦面层(原始ImageJ 1-based，按输入偏移换算): '
                f"{focus_record['source_imagej_layer_1based']}"
            )
            spacing = ca.get_focus_spacing_analysis(focus_indices, images.shape[1])
            focus_path = _save_focus_spacing_markdown(
                focus_record, spacing, file_suffix, para
            )
            if spacing['status'] == 'unsupported_stack_layers':
                raise ValueError(
                    f"Tenengrad焦面检查不支持{spacing['stack_layers']}层输入；"
                    f'已记录到{focus_path}'
                )
            _record_focus_spacing_warning(spacing, focus_path, para)
        else:
            project = ca.get_project(
                imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,
                method=pro_meth,Debug=Debug[5]
            )
        _U.save_tif_v21(f'project-origin-{file_suffix}.tif',project,para)
        project_sift, project_mean, project_scale = _scale_project_mean_to_base(project, base=stand)
        _U.save_tif_v21(f'project-mean-scale-{file_suffix}.tif',project_sift,para)
        nz,nx,ny = project.shape

        for i in range(8):
            project_8bit.append(cv2.normalize(project_sift[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
            kp1, des1 = sift.detectAndCompute(project_8bit[i], None)
            kp_list.append(kp1)
            des_list.append(des1)

        project_8bit = np.array(project_8bit)
        kp = np.array(kp_list,dtype=object)
        des = np.array(des_list,dtype=object)
        # print(kp)
        # print(des)
        #画出特征点图像
        # kp_image = []
        # for i in range(8):
        #     kp_image.append(cv2.drawKeypoints(project_8bit[i], kp[i], None))
        #     plt.figure()
        #     plt.imshow(kp_image[i])
            #plt.savefig(f'kp_image{i}.png', dpi = 340) #保存图像

        #特征点匹配 向区域4匹配
        Hs = []
        base= stand
        match_dir = _U.save_path('', para) / f"SIFT匹配-{para['peizhun_file']}"
        match_dir.mkdir(parents=True, exist_ok=True)
        for i in range(8):
            matches_imag,good_matches = ca.matcher(project_8bit[base],project_8bit[i],kp[base],kp[i],des[base],des[i],'BF','SIFT',ratio_thresh=ratio_thresh)
            #过滤特征值点，设定斜率阈值和长度阈值
            # filter_matcher = ca.filtermatcher(good_matches,kp[base],kp[i],nx = nx, slope_threshold=0.1,length_threshold=5)
            filter_matcher = np.copy(good_matches)
            filter_matcher_imag = cv2.drawMatchesKnn(
                    img1 = project_8bit[base],
                    keypoints1 = kp[base],
                    img2 = project_8bit[i],
                    keypoints2 = kp[i],
                    matches1to2 = filter_matcher,
                    outImg = None,
                    flags = 2)
            match_figure = plt.figure()
            plt.imshow(filter_matcher_imag)
            plt.title(
                f'SIFT matches: region {base + 1} -> region {i + 1} '
                f'(n={len(filter_matcher)})'
            )
            plt.axis('off')
            match_path = (
                match_dir
                / f'SIFT匹配_基准区域{base + 1}_目标区域{i + 1}.png'
            )
            match_figure.savefig(match_path, dpi=340, bbox_inches='tight')
            plt.close(match_figure)
            print(f'SIFT匹配图已保存: {match_path}')

            if len(filter_matcher) > 4:
                ptsA= np.float32([kp[base][m[0].queryIdx].pt for m in filter_matcher]).reshape(-1, 1, 2)
                ptsB = np.float32([kp[i][m[0].trainIdx].pt for m in filter_matcher]).reshape(-1, 1, 2)
                ransacReprojThreshold = 2  #
             #单应性矩阵可以将一张图通过旋转、变换等方式与另一张图对齐
                H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
                Hs.append(H)
                
                
            # if len(good_matches) > 4:
            #     ptsA= np.float32([kp[base][m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            #     ptsB = np.float32([kp[i][m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            #     ransacReprojThreshold = 5
            #     plt.figure()
            #     plt.imshow(matches_imag)
            #     #plt.savefig('matches_imag.png', dpi = 340)
            #  #单应性矩阵可以将一张图通过旋转、变换等方式与另一张图对齐
            #     H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
            #     Hs.append(H)

        #################################################
        #根据Hs  根据投影 验证
        #SIFT处理  八个区域‘投影’根据Hs 变回来
        imag_8_stack_new=[]
        for i in range(8):
            H = Hs[i]
            imgOut_2D = cv2.warpPerspective(project_sift[i], H, (ny,nx),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                
            imag_8_stack_new.append(np.array(imgOut_2D).reshape(nx,ny))

        imag_8_stack_new = np.array(imag_8_stack_new)
        #保存下来看看（验证）
        _U.save_tif_v21(f'SIFT-project_{file_suffix}.tif',imag_8_stack_new,para)

        return np.array(Hs)


# def calRegistra2(cam12_4d,pro_meth='mean',Registration=1, pgxy=0,theta =0,scale =0,Regster_f=0,file_suffix = '_' ):
#     "样品计算这 每一次先WAVE 在loop"
#     Debug= np.array([0,0,0,0,0,0,0,0,0])
#     file_suffix = str(file_suffix)+pro_meth+'_2'

#     imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8 = decentralise(cam12_4d)
#     # imag_1,imag_5 = decentralise(cam1_4d[0],cam2_4d[0])
#     # return imag_1,imag_5
    

#     print(imag_1.shape)
    
#     nx = imag_1.shape[1]
#     #555⑤ 投影得到project(8,nx,ny)
#     # project = ca.get_project(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,method='max',Debug=Debug[5])
    
#     if Registration==0:
#         ###########################################################################
#         #配准方法一
#         #5566得到project后，进行互相关,计算坐标差值
#         coordinate =  ca.cal_corrlate(project,base=4) # 4这个表示以谁为基准
#         coordinate3 = (coordinate-512+1)
#         #coordinate3[0] = 0
#         print('配准为:',coordinate3)

#         #将project 做一个移动 得到project_match
#         project_match = ca.project_pipei(project,coordinate3)  #!这是个很重要的配准信息， 进行了一个整数级的配准 

#         #保存下来看看（验证）
#         _U.save_tif(f'5566 第一种配准法project_match-【1~8】.tif',project_match)
#         ###########################################################################
#         #利用配准后的再来做一次拟合 验证
#         coordinate2 =  ca.cal_corrlate(project_match)
#         coordinate2 = coordinate2-512+1
#         coordinate2[0] = 0
#         print('再做一次配准coordinate2=',coordinate2) 
#     ##############################################

#     if Registration==1:
#             #配准方法二
#         print('【第二种配准方法】')
#         print(f'input:\n {pgxy.round(3)}')
#         print(f'input:\n {theta.round(3)}')
#         print(f'input:\n {scale.round(3)}')
#         loop = 0
#         while(1):
#             #1.根据参数，移动得到，M （注意一定是要image 通过pgxy 移动）
#             imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,theta,scale,pgxy)

#             #2.根据M得到投影projection 
#             project_M = ca.get_project( imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,method=pro_meth,Debug=Debug[5])
#             # _U.save_tif(f'project-8区域-{loop+1}次迭代配准-{file_suffix}.tif',project_M)

#             #缩小范围
#             nz,nx,ny = project_M.shape  # 8,nx,ny
#             width = 512//2
#             project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]

#             wavelet = [0,1,2,3]   #离中心的距离
#             maxx=0
#             for w in wavelet:
#                 print(f'{w}距离为中心，更新其他')
#                 centro_r = 4+w
#                 centro_l = 4-w
#                 #重新定stand
#                 #right
#                 if centro_r<=6:
#                     pgxy_sub ,theta_sub,scale_sub = Method_pocpy(project_M_c,quyuR=[centro_r,centro_r+1],stand=centro_r)
#                     print(f'【{centro_r+1}】pgxy_sub=\n{pgxy_sub[1].round(3)}')
#                     #最新最后一组
#                     if np.all( abs(pgxy_sub[1])<10):
#                         pgxy[centro_r+1] = pgxy[centro_r+1]+pgxy_sub[1]

#                 #left
#                 if centro_l>=1:
#                     pgxy_sub ,thqeta_sub,scale_sub = Method_pocpy(project_M_c,quyuR=[centro_l,centro_l-1],stand=centro_l)
#                     print(f'【{centro_l-1}】pgxy_sub=\n{pgxy_sub[1].round(3)}')
#                     if np.all( abs(pgxy_sub[1])<10):
#                         pgxy[centro_l-1] = pgxy[centro_l-1]+pgxy_sub[1]
#                 tempp = max(abs(pgxy_sub[1][0]),abs(pgxy_sub[1][1]) )
#                 maxx = max(maxx,tempp)
#                 #利用更新后的pgxy 重新移动
#                 imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,theta,scale,pgxy)
                
#                 #查看投影
#                 project_M = ca.get_project( imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,method=pro_meth,Debug=Debug[5])
#                 project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
#                 # _U.save_tif(f'project_M-8区域-{w}-wave配准.tif',project_M_c)
#             loop = loop + 1
#             if maxx > 0.1:    #变化大于0.1时启动
#                 continue       
#             else:            #变化小于1时停止
#                 break            

#         #对投影截图一部分再次计算
#         # nz,nx,ny = project_M.shape  # 8,nx,ny
#         # width = 450//2
#         # project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
#         # _U.save_tif(f'project_M_c-8区域-最后次迭代配准.tif',project_M_c)

#         # pgxy_sub ,theta_sub,scale_sub = Method_pocpy(project_M_c)
#         # print(f'【最后次】pgxy_sub=\n{pgxy_sub.round(3)}')

#         # print(f'【{loop+1}】pgxy=\n{pgxy.round(3)}')
#         # pgxy +=pgxy_sub

#         if Regster_f==0:
#             return pgxy,theta,scale,imag_5
#         else:
#             #到了这一步 前后两帧之间差距很小了，且对于MP来说，前后两帧信息量更多
#             #增加一个前后帧对齐
#             # print()
#             # print(f'‘pgxy修正后’：{pgxy[index3].round(3)}')
#             # #缩小范围
#             # nz,nx,ny = project_M.shape  # 8,nx,ny
#             # width = 512//2
#             # project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]

#             # wavelet = [0,1,2,3]   #离中心的距离   
#             # for w in wavelet:
#             #     print(f'{w}距离为中心，更新其他')
#             #     centro_r = 4+w
#             #     centro_l = 4-w
#             #     #重新定stand
#             #     #right
#             #     if centro_r<=6:
#             #         pgxy_sub ,theta_sub,scale_sub = Method_pocpy(project_M_c,quyuR=[centro_r,centro_r+1],stand=centro_r)
#             #         print(f'【{centro_r+1}】pgxy_sub=\n{pgxy_sub[1].round(3)}')
#             #         #最新最后一组
#             #         pgxy[centro_r+1] = pgxy[centro_r+1]+pgxy_sub[1]

#             #     #left
#             #     if centro_l>=1:
#             #         pgxy_sub ,theta_sub,scale_sub = Method_pocpy(project_M_c,quyuR=[centro_l,centro_l-1],stand=centro_l)
#             #         print(f'【{centro_l-1}】pgxy_sub=\n{pgxy_sub[1].round(3)}')
#             #         pgxy[centro_l-1] = pgxy[centro_l-1]+pgxy_sub[1]

#             #     #利用更新后的pgxy 重新移动
#             #     imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,theta,scale,pgxy)
                
#             #     #查看投影
#             #     project_M = ca.get_project( imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,method='max',Debug=Debug[5])
#             #     project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
#             #     _U.save_tif(f'project_M-8区域-{w}-wave配准.tif',project_M_c)
#             project_M = ca.get_project( imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,method='mean',Debug=Debug[5])
#             _U.save_tif_v21(f'project-8区域-{loop+1}次迭代配准-{file_suffix}.tif',project_M,para)

#             return imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M



#         # return pocpy_set,pocpy_theta_set,pocpy_scale_set,imag_5
#     #################################################
    
#     if Registration == 3:   #使用SIFT进行配准
#         #  计算特征点提取&生成描述时间
#         sift = cv2.SIFT_create()
#         #  使用SIFT查找关键点key points和描述符descriptors
#         project_8bit = []
#         kp_list = []
#         des_list = []
#         for i in range(8):
#             project_8bit.append(cv2.normalize(project[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
#             kp1, des1 = sift.detectAndCompute(project_8bit[i], None)
#             kp_list.append(kp1)
#             des_list.append(des1)

#         project_8bit = np.array(project_8bit)
#         kp = np.array(kp_list)
#         des = np.array(des_list)

#         #画出特征点图像
#         kp_image = []
#         for i in range(8):
#             kp_image.append(cv2.drawKeypoints(project_8bit[i], kp[i], None))
#             plt.figure()
#             plt.imshow(kp_image[i])
#             #plt.savefig(f'kp_image{i}.png', dpi = 340) #保存图像

#         #特征点匹配 向区域4匹配
#         Hs = []
#         for i in range(8):
#             matches_imag,good_matches = ca.matcher(project_8bit[4],project_8bit[i],kp[4],kp[i],des[4],des[i],'BF','SIFT')
            
#             if len(good_matches) > 4:
#                 ptsA= np.float32([kp[4][m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#                 ptsB = np.float32([kp[i][m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#                 ransacReprojThreshold = 5
#                 plt.figure()
#                 plt.imshow(matches_imag)
#                 #plt.savefig('matches_imag.png', dpi = 340)
#              #单应性矩阵可以将一张图通过旋转、变换等方式与另一张图对齐
#                 H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
#                 Hs.append(H)

#         #################################################
#         #根据Hs  根据投影 验证
#         #SIFT处理  八个区域‘投影’根据Hs 变回来
#         imag_8_stack_new=[]
#         for i in range(8):
#             H = Hs[i]
#             imgOut_2D = cv2.warpPerspective(project[i], H, (ny,nx),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                
#             imag_8_stack_new.append(np.array(imgOut_2D).reshape(nx,ny))

#         imag_8_stack_new = np.array(imag_8_stack_new)
#         #保存下来看看（验证）
#         _U.save_tif(f'第三种配准法SIFT-project_match-【1~8】.tif',imag_8_stack_new)

#         return Hs


# def calRegistra45(imag_4_M,imag_5_M,pro_meth='mean',Registration=1, pgxy=0,theta =0,scale =0,Regster_f=0,file_suffix='_',stand=1,ratio_thresh=1,para=None):
#     Debug= np.array([0,0,0,0,0,0,0,0,0])
#     file_suffix = str(file_suffix)+pro_meth+'_1'
#     # _,_,_,imag_4,imag_5,_,_,_, = decentralise(cam12_4d)
#     # imag_1,imag_5 = decentralise(cam1_4d[0],cam2_4d[0])
#     # return imag_1,imag_5
    

#     print(imag_4_M.shape)
    
#     nx = imag_4_M.shape[1]

#     if Registration==1:
#         pgxy = np.zeros((2,2))
#         theta = np.zeros((2))
#         scale = np.ones((2))
#         #利用3x3  3个方向x3帧 计算
#         for loop in range(2):
#             #1.根据参数，移动得到，M （注意一定是要image 通过pgxy 移动）
#             imag_4_M,imag_5_M = ca.one_setp45(imag_4_M,imag_5_M,theta,scale,pgxy)
#             print(imag_4_M.shape,imag_4_M.shape)
#             #2.根据M得到投影projection 

#             project_M = ca.get_slice45(imag_4_M,imag_5_M,method=pro_meth,Debug=Debug[5])
#             nz,nx,ny = project_M.shape
#             # _U.save_tif_v21(f'project454545-8区域-{loop+1}次迭代配准-{file_suffix}.tif',project_M_c,para)

#             width = 430//2 #430//2#512//2
#             project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
#             _U.save_tif_v21(f'project454545-8区域-{loop+1}次迭代配准-{file_suffix}.tif',project_M_c,para)

#             #3.根据slice计算当前需要移动的坐标
#             pgxy_sub ,theta_sub,scale_sub = Method_pocpy(project_M,quyuR=[0,1],stand=1)

#             c=np.sum(abs(pgxy_sub)>=1)
#             print(f'【{loop+1}】pgxy_sub=\n{pgxy_sub.round(2)}')
            
#             #pgxy 可累加，其他不变
#             if pgxy.any() ==0:
#                 print('初始化')
#                 pgxy,theta,scale =  pgxy_sub ,theta_sub,scale_sub
#             elif c< 3 :
#                 #选择性更新 其中小于1的亚像素移动
#                 yuzhi = 3
#                 index = np.where(abs(pgxy_sub)<yuzhi) #只更新其中<4的情况
#                 array12 = np.bincount(index[0])   #统计一行中 xy都小于1的情况
#                 index2 = np.where(array12==2)
#                 print(f'只更新其中<{yuzhi}的情况',index2)
#                 for index3 in index2:
#                     print(f'{pgxy_sub[index3].round(2)}')
#                     pgxy[index3] +=pgxy_sub[index3]

#             else:
#                 print('不更新')
#                 pass



#         return pgxy,theta,scale




#     if Registration == 3:   #使用SIFT进行配准
#         #  计算特征点提取&生成描述时间
#         sift = cv2.SIFT_create()
#         #  使用SIFT查找关键点key points和描述符descriptors
#         project_8bit = []
#         kp_list = []
#         des_list = []
#         # imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,theta,scale,pgxy)
#         # project = ca.get_project( imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,method=pro_meth,Debug=Debug[5])
#         project45 = ca.get_project45(imag_4_M,imag_5_M,method=pro_meth,Debug=Debug[5])
#         _U.save_tif_v21(f'project45-origin-{file_suffix}.tif',project45,para)
#         nz,nx,ny = project45.shape

#         for i in range(2):
#             project_8bit.append(cv2.normalize(project45[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
#             kp1, des1 = sift.detectAndCompute(project_8bit[i], None)
#             kp_list.append(kp1)
#             des_list.append(des1)

#         project_8bit = np.array(project_8bit)
#         kp = np.array(kp_list,dtype=object)
#         des = np.array(des_list,dtype=object)
#         # print(kp)
#         # print(des)
#         #画出特征点图像
#         kp_image = []
#         for i in range(2):
#             kp_image.append(cv2.drawKeypoints(project_8bit[i], kp[i], None))
#             plt.figure()
#             plt.imshow(kp_image[i])
#             #plt.savefig(f'kp_image{i}.png', dpi = 340) #保存图像

#         #特征点匹配 向区域4匹配
#         Hs = []
#         base= 1
#         for i in range(2):
#             matches_imag,good_matches = ca.matcher(project_8bit[base],project_8bit[i],kp[base],kp[i],des[base],des[i],'BF','SIFT',ratio_thresh=ratio_thresh)
#             if len(good_matches) > 4:
#                 ptsA= np.float32([kp[base][m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#                 ptsB = np.float32([kp[i][m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#                 ransacReprojThreshold = 5
#                 plt.figure()
#                 plt.imshow(matches_imag)
#                 #plt.savefig('matches_imag.png', dpi = 340)
#              #单应性矩阵可以将一张图通过旋转、变换等方式与另一张图对齐
#                 H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
#                 Hs.append(H)

#         #################################################
#         #根据Hs  根据投影 验证
#         #SIFT处理  八个区域‘投影’根据Hs 变回来
#         imag_8_stack_new=[]
#         for i in range(2):
#             H = Hs[i]
#             imgOut_2D = cv2.warpPerspective(project45[i], H, (ny,nx),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                
#             imag_8_stack_new.append(np.array(imgOut_2D).reshape(nx,ny))

#         imag_8_stack_new = np.array(imag_8_stack_new)
#         #保存下来看看（验证）
#         _U.save_tif_v21(f'SIFT-project45_{file_suffix}.tif',imag_8_stack_new,para)

#         return np.array(Hs)


def calRegistra45_3_3(imag_4_M,imag_5_M,pro_meth='mean',Hs=0,Registration=1,Regster_f=0,file_suffix='_',stand=1,ratio_thresh=1,para=None):
    "利用条纹信息，再次计算45区域"
    Debug= np.array([0,0,0,0,0,0,0,0,0])
    file_suffix = str(file_suffix)+pro_meth+'_1'
    # _,_,_,imag_4,imag_5,_,_,_, = decentralise(cam12_4d)
    # imag_1,imag_5 = decentralise(cam1_4d[0],cam2_4d[0])
    # return imag_1,imag_5
    

    print(imag_4_M.shape)
    nx = imag_4_M.shape[1]

    if Registration==1:
        pgxy = np.zeros((2,2))
        theta = np.zeros((2))
        scale = np.ones((2))
        #利用3x3  3个方向x3帧 计算

        for i in range(3):  #3个方向
            print(f'##########遍历第{i+1}方向')
            #每个方向更新一次pgxy
            #1.根据参数，移动得到，M2（注意一定是要image 通过pgxy 移动）
            imag_4_M2,imag_5_M2 = ca.one_setp45(imag_4_M,imag_5_M,theta,scale,pgxy)

            z = np.array([4*i+1, 4*i+3, 12+3*i+1]) # 1: 0123, 2:4567,3:89 10 11 ; 
                                        # 1: 12 13 14 , 2: 15 16 17
            z = z+int(21*(para['point']/10)+1)   #10: 就从第二个时间点取，90：就从10个时间点取
            print('选取的slice',z)
            pgxy_sub_set = []
            for z_j in range(3):
                #2.根据M2得到slice（利用上条纹信息）！
                slice45_z = ca.get_slice45(imag_4_M2,imag_5_M2,z=z[z_j],method=pro_meth,Debug=Debug[5])
                # slice45_z.append(ca.get_slice45(imag_4_M2,imag_5_M2,z=z[j],method=pro_meth,Debug=Debug[5])) 
                nz,nx,ny = slice45_z.shape

                width = 430//2
                slice45_z_c = slice45_z[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
                # _U.save_tif_v21(f'slice45{i,z_j}-{file_suffix}.tif',slice45_z_c,para)
                
                #3.根据slice计算当前需要移动的坐标
                pgxy_sub ,theta,scale = Method_pocpy(slice45_z_c,quyuR=[0,1],stand=1)

                #记录3次
                pgxy_sub_set.append(pgxy_sub)

            #计算平均值
            print("pgxy_sub_set=",np.array(pgxy_sub_set).round(2))
            pgxy_sub_mean = np.mean(np.array(pgxy_sub_set),axis=0)

            c=np.sum(abs(pgxy_sub_mean)>=3)
            print(f'【{i=:}-pgxy_sub_mean=\n{pgxy_sub_mean.round(2)}')
            print(f"i:{i}, c={c}")
            #pgxy 可累加，其他不变
            # if pgxy.any() ==0:
            #     print('初始化')
            #     pgxy,theta,scale =  pgxy_sub ,theta_sub,scale_sub
            if c<= 3 :
                #选择性更新 其中小于1的亚像素移动
                yuzhi = 3
                index = np.where(abs(pgxy_sub_mean)<yuzhi) #只更新其中<4的情况
                array12 = np.bincount(index[0])   #统计一行中 xy都小于1的情况
                index2 = np.where(array12==2)
                print(f'pgxy只更新其中<{yuzhi}的情况',index2)
                for index3 in index2:
                    print(f'{pgxy_sub_mean[index3].round(2)}')
                    pgxy[index3] +=pgxy_sub_mean[index3]
                print(f"{pgxy=:}")
            else:
                print('pgxy变化太大不更新')
                pass


        prpject45_z = ca.get_project45(imag_4_M2,imag_5_M2,method=pro_meth,Debug=Debug[5])

        # _U.save_tif_v21(f'prpject45_z-{file_suffix}.tif',prpject45_z,para)


        return pgxy,theta,scale




    if Registration == 3:   #使用SIFT进行配准--这个不太好
        
        temp = np.diag([1,1,1])
        Hs_233 = np.array([temp,temp],dtype=np.float32)  #记录了这次的更新变化
        print("Hs_233:",Hs_233.shape)
        print(Hs_233)
        # Hs_233 = np.zeros_like(Hs[[3,4]]) # np.zeros((2,3,3))
        #利用3x3  3个方向x3帧 计算
        for i in range(3):  #3个方向
            print(f'##########遍历第{i+1}方向')
            #每个方向更新一次pgxy
            #1.根据参数，移动得到，M2（注意一定是要image 通过pgxy 移动）
            imag_4_M2,imag_5_M2,= ca.one_setp_Hs45(imag_4_M,imag_5_M,Hs_233)

            z = np.array([4*i+1, 4*i+3, 12+3*i+1]) # 1: 0123, 2:4567,3:89 10 11 ; 
                                        # 1: 12 13 14 , 2: 15 16 17
            z = z+21*2
            Hs_sub_set = []
            for z_j in range(3):
                #2.根据M2得到slice（利用上条纹信息）！
                slice45_z = ca.get_slice45(imag_4_M2,imag_5_M2,z=z[z_j],method=pro_meth,Debug=Debug[5])
                # slice45_z.append(ca.get_slice45(imag_4_M2,imag_5_M2,z=z[j],method=pro_meth,Debug=Debug[5])) 
                nz,nx,ny = slice45_z.shape

                width = 430//2
                slice45_z_c = slice45_z[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
                _U.save_tif_v21(f'slice45{i,z_j}-{file_suffix}.tif',slice45_z_c,para)
                
                #3.根据slice计算当前需要移动的坐标
                Hs_sub,_ = Method_pocpy_SIFT(slice45_z_c,stand=1,ratio_thresh=ratio_thresh)
                # Hs_sub = Method_pocpy(slice45_z_c,quyuR=[0,1],stand=1)

                #记录3次
                Hs_sub_set.append(Hs_sub)

            #计算平均值
            print("Hs_sub_set=",np.array(Hs_sub_set).round(2))
            Hs_sub_mean = np.mean(np.array(Hs_sub_set),axis=0)

            # c=np.sum(abs(Hs_sub_mean)>=3)
            # print(f'【{i=:}-Hs_sub_mean=\n{Hs_sub_mean.round(2)}')
            # print(f"{i=:}=c={c}")
            #pgxy 可累加，其他不变
            # if pgxy.any() ==0:
            #     print('初始化')
            #     pgxy,theta,scale =  pgxy_sub ,theta_sub,scale_sub

            print(f"mean:{Hs_sub_mean.round(2)=:}")
            #更新Hs_233
            if Hs_sub_mean.any()<6:  #小于6像素的才更新
                # Hs2_233 = np.copy(Hs[3,4])

                # Hs2_233 与 Hs_sub_mean 在第四个区域的差距
                print(f"{Hs_233.round(2)=:}")
                Hs_233[0,0,-1] += Hs_sub_mean[0,0,-1]
                Hs_233[0,1,-1] += Hs_sub_mean[0,1,-1]
                # print(f"{sub1_amend_1=:}")
                # print(f"{sub1_amend_2=:}")
                # = sub1_amend_1
                # = sub1_amend_2
                print(f"更新后{Hs_233.round(2)=:}")
            # if c<= 3 :
            #     #选择性更新 其中小于3的亚像素移动
            #     yuzhi = 3
            #     index = np.where(abs(pgxy_sub_mean)<yuzhi) #只更新其中<4的情况
            #     array12 = np.bincount(index[0])   #统计一行中 xy都小于1的情况
            #     index2 = np.where(array12==2)
            #     print(f'pgxy只更新其中<{yuzhi}的情况',index2)
            #     for index3 in index2:
            #         print(f'{pgxy_sub_mean[index3].round(2)}')
            #         pgxy[index3] +=pgxy_sub_mean[index3]
            #     print(f"{pgxy=:}")
            else:
                
                print(f'本轮{i+1}方向，Hs_233 不更新')
                pass

        #统一更新Hs
        Hs2 = np.copy(Hs)


        sub1 = Hs_233[0]  # Hs_base 与 Hs_sub 在第四个区域的差距
        print(f"{sub1.round(2)=:}")
        sub1_amend_1 = sub1[0,-1]
        sub1_amend_2 = sub1[1,-1]

        print(f"{sub1_amend_1=:}")
        print(f"{sub1_amend_2=:}")
        print(f"{Hs2[[3,4]].round(2)=:}")
        Hs2[0:4,0,-1] = Hs[0:4,0,-1]-sub1_amend_1
        Hs2[0:4,1,-1] = Hs[0:4,1,-1]-sub1_amend_2
        print(f"更新后{Hs2[[3,4]].round(2)=:}")


        prpject45_z = ca.get_project45(imag_4_M2,imag_5_M2,method=pro_meth,Debug=Debug[5])
        _U.save_tif_v21(f'SIFI-更新45后-投影{file_suffix}.tif',prpject45_z,para)

        return Hs2



def DirectRegistra(cam12_4d,pro_meth='mean',Hs=np.zeros(1),pgxy=np.zeros(1),theta =np.zeros(1),scale =np.zeros(1),file_suffix = '_',para=None):
    #传入矩阵信息，直接移动
    file_suffix = str(file_suffix)+'_1'
   # imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8 = decentralise(cam12_4d)    
    cam12=decentralise(cam12_4d) #重排列区域
    nc,_,_,_ = cam12.shape
    del cam12_4d
    imag_M = []
    if  pgxy.any()==0:
        print('SIFT')
        # imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp_Hs(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,Hs)
        # imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp_Hs(cam12,Hs)
        for i in range(nc):
            imag_M.append(ca.one_setp_Hs0(cam12[i],Hs[i]))
        strt = 'SIFT'
        
    elif Hs.any()==0 :
        print('pgxy 配准')
        # imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.one_setp(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,theta,scale,pgxy)
        
        for i in range(nc):
            imag_M.append(ca.one_setp0(cam12[i],theta[i],scale[i],pgxy[i]))
        strt = 'Affine'

    del cam12
    imag_1_M = imag_M[0]
    imag_2_M = imag_M[1]
    imag_3_M = imag_M[2]
    imag_4_M = imag_M[3]
    imag_5_M = imag_M[4]
    imag_6_M = imag_M[5]
    imag_7_M = imag_M[6]
    imag_8_M = imag_M[7]
    del imag_M
    # del imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8
    
    #查看投影-更改为每个通道
    zhenshu = para['zhenshu']
    project_M = ca.get_project( imag_1_M[0:zhenshu],imag_2_M[0:zhenshu],imag_3_M[0:zhenshu],imag_4_M[0:zhenshu],imag_5_M[0:zhenshu],imag_6_M[0:zhenshu],imag_7_M[0:zhenshu],imag_8_M[0:zhenshu],method='mean')
    nz,nx,ny = project_M.shape  # 8,nx,ny
    # width = 512//2#430//2#512//2
    # project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
    project_M_c = project_M[:,40:nx-10, 15:ny-35] #一边取 #40为第二行负数最小值，10为第二行正数最大值

    peizhun_file = para['peizhun_file']
    W_order = para['W_order'][0]
    _U.save_tif_v21(f'③project-SAMPLE-mean_-Direct-{strt}-{file_suffix}-{peizhun_file}-{W_order}.tif',project_M_c,para)


    project_M = ca.get_project( imag_1_M[2*zhenshu:3*zhenshu],imag_2_M[2*zhenshu:3*zhenshu],imag_3_M[2*zhenshu:3*zhenshu],imag_4_M[2*zhenshu:3*zhenshu],imag_5_M[2*zhenshu:3*zhenshu],imag_6_M[2*zhenshu:3*zhenshu],imag_7_M[2*zhenshu:3*zhenshu],imag_8_M[2*zhenshu:3*zhenshu],method='mean')

    nz,nx,ny = project_M.shape  # 8,nx,ny
    # width = 512//2#430//2#512//2
    # project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
    project_M_c = project_M[:,40:nx-10, 15:ny-35] #一边取 #40为第二行负数最小值，10为第二行正数最大值
    peizhun_file = para['peizhun_file']
    W_order = para['W_order'][1]
    _U.save_tif_v21(f'③project-SAMPLE-mean_-Direct-{strt}-{file_suffix}-{peizhun_file}-{W_order}.tif',project_M_c,para)

    return imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M

def DirectRegistra45(cam12_4d,pro_meth='mean',Hs=np.zeros(1),pgxy=np.zeros(1),theta =np.zeros(1),scale =np.zeros(1),file_suffix = '_',para=None):
    #传入矩阵信息，直接移动
    file_suffix = str(file_suffix)+'_1'
    imag_4,imag_5 = decentralise45(cam12_4d)

    if  pgxy.any()==0:
        print('SIFT')
        imag_4_M,imag_5_M = ca.one_setp_Hs45(imag_4,imag_5,Hs[[3,4]])# 传入{3,4}不一定正确
        strt = 'SIFT'
    elif Hs.any()==0 :
        print('pgxy 配准')
        imag_4_M,imag_5_M = ca.one_setp45(imag_4,imag_5,theta[[3,4]],scale[[3,4]],pgxy[[3,4]])
        strt = 'Affine'

    #查看投影
    project_M = ca.get_project45( imag_4_M,imag_5_M,method=pro_meth)
    nz,nx,ny = project_M.shape  # 8,nx,ny
    width = 512//2#430//2#512//2
    project_M_c = project_M[:,nx//2-width:nx//2+width,ny//2-width:ny//2+width]
    _U.save_tif_v21(f'①project45-{pro_meth}_{strt}-{file_suffix}.tif',project_M_c,para)
    return imag_4_M,imag_5_M   


def Method_pocpy(projectM,quyuR=[0,1,2,3,4,5,6,7],stand=4):
    #根据投影计算
    pgxy0 = []    #8个区域的 x,y平均值 
    theta0 = []   #8个区域 theta平均值
    scale0 = []
    for i in quyuR:   #以三作为标准
        #每个区域多次计算，取均值
        pgxy_set1 = []
        theta_set1 = []
        scale_set1 = []

        for j in range(1,10):
            t2,theta2,scale2 = pocpy.ripoc(projectM[stand],projectM[i],fitting_shape=(9, 9),M=j)   #!!!!!!!!!!!!!!
            pgxy_set1.append([t2[1],t2[2]])
            theta_set1.append(theta2)
            scale_set1.append(scale2)

        pgxy_meann = np.mean(np.array(pgxy_set1),axis=0)
        theta_mean = np.mean(np.array(theta_set1))
        scale_mean = np.mean(np.array(scale_set1))

        pgxy0.append(pgxy_meann)
        theta0.append(theta_mean)
        scale0.append(scale_mean)

    pgxy0=np.array(pgxy0)#.round(3)
    theta0 = np.array(theta0)
    scale0 = np.array(scale0)
    return pgxy0,theta0,scale0

def Method_pocpy_SIFT(data3D,stand=1,ratio_thresh=1):
    '''
    data3D 为 nc,nx,ny
    '''
    #  计算特征点提取&生成描述时间
    sift = cv2.SIFT_create()

    project_8bit = []
    kp_list = []
    des_list = []
    nc,nx,ny = data3D.shape
    
    # 使用SIFT查找关键点key points和描述符descriptors
    for i in range(nc):
        project_8bit.append(cv2.normalize(data3D[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
        kp1, des1 = sift.detectAndCompute(project_8bit[i], None)
        kp_list.append(kp1)
        des_list.append(des1)

    project_8bit = np.array(project_8bit)
    kp = np.array(kp_list,dtype=object)
    des = np.array(des_list,dtype=object)
    # print(kp)
    # print(des)
    #画出特征点图像
    kp_image = []
    for i in range(nc):
        kp_image.append(cv2.drawKeypoints(project_8bit[i], kp[i], None))
        plt.figure()
        plt.imshow(kp_image[i])
        #plt.savefig(f'kp_image{i}.png', dpi = 340) #保存图像

    #特征点匹配 向区域4匹配
    Hs = []
    base= stand
    for i in range(nc):
        matches_imag,good_matches = ca.matcher(project_8bit[base],project_8bit[i],kp[base],kp[i],des[base],des[i],'BF','SIFT',ratio_thresh=ratio_thresh)
        if len(good_matches) > 4:
            ptsA= np.float32([kp[base][m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            ptsB = np.float32([kp[i][m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            ransacReprojThreshold = 5
            plt.figure()
            plt.imshow(matches_imag)
            #plt.savefig('matches_imag.png', dpi = 340)
         #单应性矩阵可以将一张图通过旋转、变换等方式与另一张图对齐
            H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
            Hs.append(H)
        else:
            raise("good_matches 不够，请调大")

    #################################################
    #根据Hs  根据投影 验证
    #SIFT处理  八个区域‘投影’根据Hs 变回来
    imag_8_stack_new=[]
    for i in range(nc):
        H = Hs[i]
        imgOut_2D = cv2.warpPerspective(data3D[i], H, (ny,nx),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
        imag_8_stack_new.append(np.array(imgOut_2D).reshape(nx,ny))

    imag_8_stack_new = np.array(imag_8_stack_new)
    #保存下来看看（验证）
    # _U.save_tif_v21(f'SIFT-project45_{file_suffix}.tif',imag_8_stack_new,para)

    return np.array(Hs),imag_8_stack_new
