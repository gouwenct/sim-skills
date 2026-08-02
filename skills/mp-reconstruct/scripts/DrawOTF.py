
def decentralise(cam12_4d):
    #
    region_n,nz,nx,ny = cam12_4d.shape #(8,nz,nx,ny)
    assert region_n==8
    imag_1 = cam12_4d[0]
    imag_2 = cam12_4d[1]
    imag_3 = cam12_4d[2]
    imag_4 = cam12_4d[3]
    
    imag_5 = cam12_4d[4]
    imag_6 = cam12_4d[5]
    imag_7 = cam12_4d[6]
    imag_8 = cam12_4d[7]

    return imag_4,imag_3,imag_2,imag_1,imag_5,imag_6,imag_7,imag_8



def drawROIs(cam12_4d):
    #修改ROI程序 返回最大值中心位置
    p = Path(str(savepath1)).iterdir() #仅遍历当前目录下内容
    nnn=0 #用于计算

    yuzhi= 20  #阈值
    suoxiao = 1.0
    size= 120     #截图大小
    r_julei = 25 #距离小于这个值的分为一类--点大就取大一点
    jugle_noise = 3  #忽略小于等于这个数量的聚类点
    MatchXY = 4 #提取出ROI值后 小于这个区域的分为同一类，如果抖动很厉害，那么这个值可以设置大
    
    """
    以上参数设定， 阈值可取大一点，这样宁缺毋滥，且留下的点区域变小，如果有两个挨在一起的点，也容易根据r_julei 判断是否需要舍弃
    可根据结果文件roi 来进行调整
    """
    Debug= np.array([0,0,1,0,1,0,0,0,1])     #3 输出区域1的拟合值，这个开启会慢， 4 输出所有区域上的拟合值
                                             #8 保存ROI区域，  5投影  2输出删除之后的对应坐标
                    #  1 2 3 4 5 6 7 8
    #111切割区域
#     iii = 1        #用于标记这是第几个cam相机
#     for ii in p:  #假设有两个cam 一共可切为8块
#         if ii.is_file():
#             camer_name = ii.name[:4]
#             if iii==1:   #445  591  483  667  = 2186
#                 print(camer_name)
#                 print("4321 camer_name=",camer_name)
#                 #这里面cam1_channel_12就包括 2x4幅imag_4,imag_3,imag_2,imag_1
#                 cam1_4d = pre_process(ii,AVG=1,CHANNEL=1,camer_name=camer_name) #预处理 AVG帧，
# #                 imag_4,imag_3,imag_2,imag_1,nx,ny,nz,bili1 = ca.load_RAW_DATA_1x4_2(ii,Debug=1,suoxiao=suoxiao,camer_name=camer_name)  # 1x4的图切分为4块，Debug=1 保存这四块图

#             elif iii==2:   #第一个相机的 586 475 527 437    =2025  _2 没有要这个分光棱镜系数
#                 print(camer_name)
#                 print("5678 camer_name=",camer_name)
#                 cam2_4d = pre_process(ii,AVG=1,CHANNEL=1,camer_name=camer_name)
# #                 imag_5,imag_6,imag_7,imag_8,nx,ny,nz,bili2 = ca.load_RAW_DATA_1x4_2(ii,Debug=1,camer_name=camer_name,suoxiao=suoxiao) 

#             iii+=1
    
    imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8 = decentralise(cam12_4d)
            
    #
    nx = imag_1.shape[1]
    
    #555⑤ 投影得到project(8,nx,ny)
    project = ca.get_project(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,method='max',Debug=Debug[5])

    ###########################################################################
    #配准方法一
    #5566得到project后，进行互相关,计算坐标差值
    coordinate =  ca.cal_corrlate(project,4)
    coordinate3 = (coordinate-512+1)
    print('配准为:',coordinate3)

    #将project 做一个移动 得到project_match
    project_match = ca.project_pipei(project,coordinate3)  #!这是个很重要的配准信息， 进行了一个整数级的配准 
    #保存下来看看（验证配准效果）
    
    n.save_tif(f'5566project_match-all.tif',project_match)
    ###########################################################################
    #利用配准后的再来做一次拟合 验证
    coordinate2 =  ca.cal_corrlate(project_match,4)
    coordinate2 = coordinate2-512+1
    print('再做一次配准coordinate2=',coordinate2) 
    ##############################################
    
    if Registration==1:
            #配准方法二
        print('【第二种配准方法】')
        pocpy_set = []    #8个区域的 x,y平均值 
        pocpy_theta_set = []   #8个区域 theta平均值
        pocpy_scale_set = []
        for i in [0,1,2,3,4,5,6,7]:   #以三作为标准
            #每个区域多次计算，取均值
            t_set = []
            theta_set = []
            scale_set = []

            for j in range(1,50):
                t,theta,scale = pocpy.ripoc(project[4],project[i],fitting_shape=(9, 9),M=j)   #!!!!!!!!!!!!!!
                t_set.append([t[1],t[2]])
                theta_set.append(theta)
                scale_set.append(scale)

    #         print(f'{i} t_set:',np.array(t_set),np.array(t_set).shape)
    #         print(f'{i} theta_set:',np.array(theta_set),np.array(theta_set).shape)
    #         print(f'{i} scale:',np.array(scale_set),np.array(scale_set).shape)
            t_mean = np.mean(np.array(t_set),axis=0)
            theta_mean = np.mean(np.array(theta_set))
            scale_mean = np.mean(np.array(scale_set))

    #         print('t_mean',t_mean)
            pocpy_set.append(t_mean)
            pocpy_theta_set.append(theta_mean)
            pocpy_scale_set.append(scale_mean)

        pocpy_set=np.array(pocpy_set).round(3)
        pocpy_theta_set = np.array(pocpy_theta_set)
        pocpy_scale_set = np.array(pocpy_scale_set)
        print(pocpy_set)
        print(pocpy_theta_set)
        print(pocpy_scale_set)
        return pocpy_set,pocpy_theta_set,pocpy_scale_set
    #################################################
    
    
    #6666⑥ 根据project进行模板匹配找index，
    #  返回XY_list (8,ROIs,2) #以阈值为标准，不断删除那些点 
    #聚类，删除挨着,删除noise ，删除边界点
    XY_lists = ca.find_beads_index(project_match,coordinate3,YZ1=yuzhi,JL_JL2=r_julei,jugle_noise3=jugle_noise,size=size,Debug=Debug[6],margin=size//2 )
    
    #777⑦得到匹配的2D对应下标(8,Rois,2)!!!!
#     print('XY_lists0=',XY_lists[0])
    
    right_list = ca.get_right_list(XY_lists,MatchXY=MatchXY,Debug = Debug[2])  #输出删除之后的对应坐标
    
    #10101010 根据最大值投影，查看函数
    ca.get_2D_project_match(right_list,project_match)#对配准后的图对齐  并标记出这些点出来+
    
    #根据coordinate 对image_stack做一个移动
    imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M = ca.match_image(imag_1,imag_2,imag_3,imag_4,imag_5,imag_6,imag_7,imag_8,coordinate)

    #888⑧ 根据对应2D下标，及其值 找到在3D图像中的位置，根据位置截取出ROI区域 (Rois,8,3)

    ROIS_1,ROIS_2,ROIS_3,ROIS_4,ROIS_5,ROIS_6,ROIS_7,ROIS_8,right_list_3d = ca.get_right_list3d(imag_1_M,imag_2_M,imag_3_M,imag_4_M,imag_5_M,imag_6_M,imag_7_M,imag_8_M,right_list,project_match,size=size,Debug =Debug[8])
    
    return ROIS_1,ROIS_2,ROIS_3,ROIS_4,ROIS_5,ROIS_6,ROIS_7,ROIS_8


def drawOTF(ROIS_1,ROIS_2,ROIS_3,ROIS_4,ROIS_5,ROIS_6,ROIS_7,ROIS_8):
	pass

def drawOTF_fromFre():
	pass



def drawOTF_fromKong():
	pass