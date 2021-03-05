import numpy as np
import open3d as o3d


def normalize(pcl,visutalization=False):

    pcl[:, :2] = pcl[:, :2] / 100
    pcl[:, 2] = pcl[:, 2] / 20
    pcl = zoom_in(pcl[:, :3])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
    pcd = remove_outlier(pcd)
    pcd = remove_ground(pcd)
    pcd = down_sample(pcd)
    pcd = down4096(pcd)
    
    pcl=np.asarray(pcd.points)

    cx=np.mean(pcl[:,0])
    cy=np.mean(pcl[:,1])
    cz=np.mean(pcl[:,2])
    new_pos=np.array([cx,cy,cz])
    if visutalization:
        display_pcl(pcl,'normalized')
    return pcl-new_pos,new_pos*np.array([100,100,20])


def zoom_in(pcl):
    num = pcl.shape[0]
    pcl_new = []
    for i in range(num):
        if abs(max(pcl[i, :])) < 1 and abs(min(pcl[i, :])) < 1:
            pcl_new.append(pcl[i, :])
    pcl_new = np.vstack(pcl_new).reshape(-1, 3)
    
    return pcl_new


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    # print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Open3D Removal Outlier', width=1920,
                                      height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)


def display_pcl(cloud,name='Open3D'):
    #     input [N, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd], window_name=name, width=1920,
                                      height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)

def add_noise_points(pcl,num,num_per_point,r_per_point,box,base,visutalization=False):
    if visutalization:
        display_pcl(pcl,'before noise')
    noise=[pcl]
    for i in range(num):
        noise_base=np.random.rand(3)*box+base
        for j in range(num_per_point):
            noise_point=noise_base+(np.random.rand(3)-0.5)*r_per_point
            noise.append(noise_point)
    pcl=np.vstack(noise)
    if visutalization:
        display_pcl(pcl,'after noise')
    return pcl

def add_noise_points2(pcl,num,num_per_point,r_per_point,box,base,visutalization=False):
    if visutalization:
        display_pcl(pcl,'before noise')
    noise=[pcl]
    if num<0:
        num=len(pcl)//100
    for i in range(num):
        noise_base=pcl[int(np.random.rand()*pcl.shape[0])]
        noise_points=noise_base+(np.random.rand(num_per_point,3)-0.5)*r_per_point
        noise.append(noise_points)
    pcl=np.vstack(noise)

    if visutalization:
        display_pcl(pcl,'after noise')
    return pcl

def select_points_not_in(pcl, center, box):
        cloud = []
        ground = []

        sel1=pcl[:,0]<center[0]+box[0]
        sel2=pcl[:,0]>center[0]-box[0]
        sel3=pcl[:,1]<center[1]+box[1]
        sel4=pcl[:,1]>center[1]-box[1]
        sel5=pcl[:,2]<center[2]+box[2]
        sel6=pcl[:,2]>center[2]-box[2]

        sel=np.logical_and(sel1,sel2)
        sel=np.logical_and(sel,sel3)
        sel=np.logical_and(sel,sel4)
        sel=np.logical_and(sel,sel5)
        sel=np.logical_and(sel,sel6)
        sel=np.logical_not(sel)
        cloud=pcl[sel]
        
        return cloud

def add_noise_points3(pcl,num,num_per_point,r_per_point,box,base,visutalization=False):
    if visutalization:
        display_pcl(pcl,'before noise')
    
    if num<0:
        num=len(pcl)//100
    for i in range(num):
        noise_index=int(np.random.rand()*pcl.shape[0])
        noise_base=pcl[noise_index]
        selected=select_points_not_in(pcl,noise_base,r_per_point)
        noise_num=pcl.shape[0]-selected.shape[0]
        noise_points=noise_base+2*(np.random.rand(noise_num,3)-0.5)*r_per_point

        pcl=np.vstack([selected,noise_points])
            
    if visutalization:
        display_pcl(pcl,'after noise')
    return pcl

def down4096(pcd):
    pcl = np.asarray(pcd.points)
    np.random.shuffle(pcl)
    if pcl.shape[0] >= 4096:
        pcl = pcl[:4096, :]
    else:
        compensate = 4096 - pcl.shape[0]
        idx = np.random.randint(0, pcl.shape[0], size=compensate)
        pcl_ = pcl[idx, :]
        pcl = np.vstack((pcl, pcl_))
    pcd.points = o3d.utility.Vector3dVector(pcl)

    return pcd


def down_sample(pcd, visualization=False):
    # 下采样
    # voxel_down_sample（把点云分配在三维的网格中，取平均值）
    # uniform_down_sample (可以通过收集每第n个点来对点云进行下采样)
    # select_down_sample (使用带二进制掩码的select_down_sample仅输出所选点。选定的点和未选定的点并可视化。）
    v_size = 0.02

    downpcd = pcd1 =  pcd.voxel_down_sample(voxel_size=v_size)
    while len(pcd1.points) < 4096 and v_size>0.001:
        downpcd = pcd1
        v_size -= 0.001
        pcd1 = pcd.voxel_down_sample(voxel_size=v_size)
    if visualization:
        print(' down sample to:',len(downpcd.points))
        o3d.visualization.draw_geometries([downpcd], window_name='Open3D downSample', width=1920, height=1080, left=50,
                                          top=50,
                                          point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

    return downpcd


def remove_outlier(pcd, visualization=False):
    # 离群点去除 【outlier removal】
    # 点云离群值去除 从扫描设备收集数据时，可能会出现点云包含要消除的噪声和伪影的情况。本教程介绍了Open3D的异常消除功能。 准备输入数据，使用降采样后的点云数据。
    # statistical_outlier_removal 【统计离群值移除】 删除与点云的平均值相比更远离其邻居的点。
    #          它带有两个输入参数：nb_neighbors 允许指定要考虑多少个邻居，以便计算给定点的平均距离
    #                           std_ratio 允许基于跨点云的平均距离的标准偏差来设置阈值级别。此数字越低，过滤器将越具有攻击性
    #
    # radius_outlier_removal 【半径离群值去除】  删除在给定球体中周围几乎没有邻居的点。
    #          两个参数可以调整以适应数据：nb_points 选择球体应包含的最小点数
    #                                  radius 定义将用于计算邻居的球体的半径
    # print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if visualization:
        display_inlier_outlier(pcd, ind)
    downpcd_inlier_cloud = pcd.select_by_index(ind)

    return downpcd_inlier_cloud


def remove_ground(pcd, visualization=False):
    # 平面分割 【Plane Segmentation】
    # Open3D还包含使用RANSAC从点云中分割几何图元的支持。要在点云中找到具有最大支持的平面，我们可以使用segement_plane。该方法具有三个参数。
    # distance_threshold定义一个点到一个估计平面的最大距离，该点可被视为一个不规则点； ransac_n定义随机采样的点数以估计一个平面； num_iterations定义对随机平面进行采样和验证的频率。
    # 函数然后将平面返回为（a，b，c，d） 这样，对于平面上的每个点（x，y，z），我们都有ax + by + cz + d = 0。该功能进一步调整内部点的索引列表。
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=5, num_iterations=100)
    [a, b, c, d] = plane_model

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    if visualization:
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Open3D Plane Model', width=1920,
                                          height=1080, left=50, top=50, point_show_normal=False,
                                          mesh_show_wireframe=False,
                                          mesh_show_back_face=False)
    return outlier_cloud
