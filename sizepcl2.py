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
import pdb
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='../../newdata/00', type=str)
args = parser.parse_args()


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

def read_csv(fname):
    with open(fname) as f:
        records=[]
        for i in f.readlines()[1:]:

            s=i.split(',')
            record=[s[0]]
            for j in s[1:]:
                record.append(float(j))
            records.append(record)
        return records

def read_pcl(dir,record):
    print('read ',record[0])
    fname=os.path.join(dir,'pointcloud_20m',record[0]+'.bin')
    pcl=np.fromfile(fname,dtype=np.float64).reshape(-1, 3)
    print_pcl_info(pcl)
    if len(record)>=9:
        pos=np.array(record[1:4]).astype(np.float32)
        rot=Rotation.from_rotvec(np.array(record[4:7])).as_matrix()
        scale=np.diag(record[7:10])
        pcl=pcl +pos@scale
    return pcl

dir1='/media/mustar/DATA4/LGSVL/benchmark_datasets/lgsvl6/00'
dir2='/media/mustar/DATA4/LGSVL/benchmark_datasets/lgsvl6/01'

records1=read_csv(os.path.join(dir1,"pointcloud_locations_20m.csv"))
records2=read_csv(os.path.join(dir2,"pointcloud_locations_20m.csv"))

pcl1=read_pcl(dir1,records1[5])
pcl2=read_pcl(dir2,records2[6])

visual_pcl(pcl1, pcl2)