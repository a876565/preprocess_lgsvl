import numpy as np
from glob import glob
from tqdm import tqdm
import os
import open3d as o3d
import pandas as pd
import shutil
from os.path import join
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import re
from normalize_pcl import normalize
import argparse
from scipy.spatial.transform import Rotation
import math

def visual_pcl(pcl1, pcl2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcl1[:,:3])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcl2[:,:3])
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name='Open3D Origin', width=1920, height=1080, left=50, top=50,
                                      point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

def print_pcl_info(pcl):
    print('normalized:\tx\t\ty\t\tz\n',
                '      max:\t%4f\t%4f\t%4f\n'%(np.max(pcl[:,0]),np.max(pcl[:,1]),np.max(pcl[:,2])),
                '      min:\t%4f\t%4f\t%4f\n'%(np.min(pcl[:,0]),np.min(pcl[:,1]),np.min(pcl[:,2])),
                '   median:\t%4f\t%4f\t%4f\n'%(np.median(pcl[:,0]),np.median(pcl[:,1]),np.median(pcl[:,2])),
                '     mean:\t%4f\t%4f\t%4f\n'%(np.mean(pcl[:,0]),np.mean(pcl[:,1]),np.mean(pcl[:,2]))
            )

def deal_kitti_pcl(pcl):
    pcl=pcl[pcl[:,2]>-1.45]
    return pcl
            

file1 = open('/media/mustar/DATA4/LGSVL/benchmark_datasets/kitti_new/00/000000.bin')
pcl1 = np.fromfile(file1, dtype=np.float32).reshape(-1, 4)
pcl1 = deal_kitti_pcl(pcl1[:,:3])
file2 = open('/media/mustar/DATA4/LGSVL/benchmark_datasets/kitti_new/00/000001.bin')
pcl2 = np.fromfile(file2, dtype=np.float32).reshape(-1, 4)
pcl2 = pcl2[:,:3]
pcl2 = deal_kitti_pcl(pcl2[:,:3])
#print(np.max(np.abs(pcl1), axis=0))
#print(np.max(np.abs(pcl2), axis=0))
#for i in range(3):
#    pcl1[:,i]-=np.mean(pcl1[:,i])
print_pcl_info(pcl1)
print_pcl_info(pcl2)
visual_pcl(pcl1, pcl2)