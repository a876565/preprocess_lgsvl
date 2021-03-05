import numpy as np
import math
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
from math import pi
from scipy.spatial.transform import Rotation
import pdb
import random

Degree2Rad = 1.0 * pi / 180.0
Rad2Degree = 180.0 / pi 


class Prepross:
    def __init__(self, source_dir, target_dir):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.mkdir_dirs()
        self.make_gt()
        tqdm.write('[OK] make_gt')
        self.generate_map()
        tqdm.write('[OK] generate_map')
        self.generate_samples()
        tqdm.write('[OK] generate_samples')
        if args.build_map:
            self.generate_pcls(self.tar_train_dir,self.timestamp_list_train,self.positions_train,[100,100,30],args.visual)
            self.generate_pcls(self.tar_test_dir,self.timestamp_list_test,self.positions_test,[100,100,30],args.visual)
        else:
            self.move_pcl()
        tqdm.write('[OK] move_pcl')
        self.save_samples_csv()
        tqdm.write('[OK] save_samples_csv')

    def mkdir_dirs(self):

        self.source_odometry_lidar = join(self.source_dir, "odometry_lidar")
        self.mkdir_safe(self.source_odometry_lidar)
        self.source_calib = join(self.source_dir, "calib.txt")
        self.source_velodyne = join(self.source_dir, "velodyne")
        self.seq_name = os.path.basename(self.source_dir)

        self.source_poses = join(self.source_dir, "../poses",self.seq_name+'.txt')

        self.mkdir_safe(self.target_dir)

        self.tar_train_loc = join(self.target_dir, "./pointcloud_locations_20m_10overlap.csv")
        self.tar_test_loc = join(self.target_dir, "./pointcloud_locations_20m.csv")
        self.tar_trans_loc = join(self.target_dir, "./pointcloud_transforms.csv")
        self.tar_train_dir = os.path.join(self.target_dir, 'pointcloud_20m_10overlap')
        self.mkdir_safe(self.tar_train_dir)
        self.tar_test_dir = os.path.join(self.target_dir, 'pointcloud_20m')
        self.mkdir_safe(self.tar_test_dir)

        self.seq_id=int(self.seq_name)
        self.base_map_x=args.base_map_x*self.seq_id
        self.base_map_y=args.base_map_y*self.seq_id

    @staticmethod
    def read_matrix(line):
        line=[float(f) for f in line.split()]
        mat=np.array(line).reshape(3,4)
        mat=np.vstack([mat,np.array([0.,0.,0.,1.])])
        return mat

    @staticmethod
    def print_pose(p):
        pos=p[:3,3]
        rot=p[:3,:3]
        rot=Rotation.from_matrix(rot).as_euler('xyz',degrees=True)
        print('Position:\n',pos)
        print('Rotations:\n',rot)

    def make_gt(self):
        
        self.positions=[]
        self.rotations=[]
        
        self.poses=[]
        with open(self.source_calib) as f:
            lines=f.readlines()
            P0=self.read_matrix(lines[0].replace('P0:',''))
            Tr=self.read_matrix(lines[4].replace('Tr:',''))
            invTr=np.linalg.inv(Tr)
            print('P0:\n',P0)
            self.print_pose(P0)
            print('Tr:\n',Tr)
            self.print_pose(Tr)

        with open(self.source_poses) as f:
            poses_line=f.readlines()
            init_T=None
            for line in poses_line:
                pose_f=self.read_matrix(line)

                pose=invTr@pose_f@Tr

                position=pose[:3,3]

                rotation=pose[:3,:3]

                self.positions.append(position)
                self.rotations.append(rotation)
                
                self.poses.append(pose)

                
        positions=np.vstack(self.positions)
        #rotations=np.vstack(self.rotations)

        self.print_pcl_info(positions,'positions')
        #self.print_pcl_info('rotations',rotations*Rad2Degree)
        pass

    @staticmethod
    def make_time_stamp(i):
        if isinstance(i,str):
            return '0'*(6-len(i))+i if len(i)<6 else i
        elif isinstance(i,int):
            return '%06d'%i
        else:
            return '000000'

    def generate_map(self):
        # for pcl_file in sorted(glob(os.path.join(self.source_velodyne, '*.bin'))):

        positions = [[0, 0, 0]]
        timestamp_list = []
        poses = []

        for i,p in enumerate(self.poses):
            position=self.positions[i]
            if self.findNearest(position, positions, 40):
                positions.append(position)
                timestamp_list.append(self.make_time_stamp(str(i)))
                poses.append(p)

        map = []
        for timestamp, pose in zip(timestamp_list, poses):
            pcl_file = os.path.join(self.source_velodyne, timestamp + '.bin')
            pcl = np.fromfile(pcl_file, dtype=np.float32).reshape(-1, 4)[:, :3]
            pcl = pcl @ pose[:3, :3].T + pose[:3, 3].reshape(1, 3)
            map.append(pcl)

        if False:
            mmap=map[0]
            for i in range(1,len(map)):
                self.visual_pcl(mmap,map[i])
                mmap=np.vstack([mmap,map[i]])

        map = np.concatenate(map, axis=0)
        self.print_pcl_info(map,'map')
        self.map = map
        if args.show_map:
            map_pcd = o3d.geometry.PointCloud()
            map_pcd.points = o3d.utility.Vector3dVector(self.map)
            map_pcd = map_pcd.voxel_down_sample(voxel_size=0.3)

            o3d.visualization.draw_geometries([map_pcd], window_name='Open3D Origin')


    def save_trans(self):
        trans = []
        timestamps = []
        gps_files = sorted(glob(os.path.join(self.source_gps, '*')))
        for gps_file in gps_files:
            timestamp = os.path.splitext(gps_file)[0].split('/')[-1]
            t = self.get_transform(gps_file)
            trans.append(t)
            timestamps.append(timestamp)

        trans = np.asarray(trans).reshape(-1, 6)
        # 字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'timestamp': timestamps,
                                  'x': trans[:, 0].reshape(-1),
                                  'y': trans[:, 1].reshape(-1),
                                  'z': trans[:, 2].reshape(-1),
                                  'rx': trans[:, 3].reshape(-1),
                                  'ry': trans[:, 4].reshape(-1),
                                  'rz': trans[:, 5].reshape(-1)})
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv(self.tar_trans_loc, index=False, sep=',')

    def generate_samples(self):
        self.positions_train = [[0, 0, 0]]
        self.positions_test = [[0, 0, 0]]

        self.timestamp_list_train = []
        self.timestamp_list_test = []
        
        base_map_move=np.array([self.base_map_x,self.base_map_y,0.0])
        for timestamp,position in enumerate(self.positions):
            
            position = position+base_map_move

            self.positions_train.append(position)
            self.timestamp_list_train.append(timestamp+100000)
            #if self.findNearest(position, self.positions_train, 10):
            #    self.positions_train.append(position)
            #    self.timestamp_list_train.append(timestamp+100000)

            if self.findNearest(position, self.positions_test, 40):
                self.positions_test.append(position)
                self.timestamp_list_test.append(timestamp+100000)

        # 去除第一个[0,0,0]
        self.positions_train = np.asarray(self.positions_train[1:]).reshape(-1, 3)
        self.positions_test = np.asarray(self.positions_test[1:]).reshape(-1, 3)

    def save_samples_csv(self):
        # 字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'timestamp': self.timestamp_list_train,
                                  'northing': self.positions_train[:, 0].reshape(-1),
                                  'easting': self.positions_train[:, 1].reshape(-1),
                                  'altitude': self.positions_train[:, 2].reshape(-1),
                                  })
                                  
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv(self.tar_train_loc, index=False, sep=',')

        dataframe = pd.DataFrame({'timestamp': self.timestamp_list_test,
                                  'northing': self.positions_test[:, 0].reshape(-1),
                                  'easting': self.positions_test[:, 1].reshape(-1),
                                  'altitude': self.positions_test[:, 2].reshape(-1),
                                  })
        dataframe.to_csv(self.tar_test_loc, index=False, sep=',')

    def move_pcl_data(self,ts,target_dir,positions):
        for i,pos in tqdm(zip(ts,positions),total=len(ts)):
            i=i if i<100000 else i-100000
            src_name=self.make_time_stamp(i)
            dst_name=self.make_time_stamp(i+100000)
            src_path=os.path.join(self.source_velodyne,src_name+'.bin')
            dst_path=os.path.join(target_dir,dst_name+'.bin')
            
            src_pcl=np.fromfile(src_path,dtype=np.float32).reshape(-1,4)[:,:3]
            src_pcl=src_pcl@self.poses[i][:3,:3].T
            
            dst_pcl,newpos=normalize(src_pcl,False)

            dst_pcl.astype(np.float64).tofile(dst_path)
            pos+=newpos

    def move_pcl(self):
        self.move_pcl_data(self.timestamp_list_test,self.tar_test_dir,self.positions_test)
        self.move_pcl_data(self.timestamp_list_train,self.tar_train_dir,self.positions_train)
        

    def select_points_in(self, center, box, keepground=True):
        cloud = []
        ground = []

        mapped=self.map-center
        sel1=mapped[:,0]<box[0]
        sel2=mapped[:,0]>-box[0]
        sel3=mapped[:,1]<box[1]
        sel4=mapped[:,1]>-box[1]
        sel5=mapped[:,2]<box[2]
        sel6=mapped[:,2]>-box[2]

        sel=np.logical_and(sel1,sel2)
        sel=np.logical_and(sel,sel3)
        sel=np.logical_and(sel,sel4)
        ground=mapped[sel]

        sel=np.logical_and(sel,sel5)
        sel=np.logical_and(sel,sel6)
        cloud=mapped[sel]
        
        return cloud, ground

    def generate_pcls(self, dirpath, timestamps, positions, box, loginfo=False):
        if loginfo:
            print(' map info:')
            self.print_pcl_info(self.map)
            
        for (ts, pos) in tqdm(zip(timestamps, positions), total=len(timestamps),desc=dirpath):
            if loginfo:
                print(" positon=", pos)
            if ts<100000:
                ts+=100000
            pcl, pcl_g = self.select_points_in(pos, box)

            if pcl is not None:
                pcl_norm,newpos = normalize(pcl,args.visual)
                pcl_file_new = os.path.join(dirpath, self.make_time_stamp(ts) + '.bin')

                pcl_norm.astype(np.float64).tofile(pcl_file_new)
                pos+=newpos

                if loginfo:
                    self.print_pcl_info(pcl_norm)
                    self.visual_pcl1(pcl_norm)
        print('\n')

    def check(self):

        pclfiles = sorted(glob(os.path.join(self.source_velodyne, '*.bin')))
        one = 1
        othor = one + 10
        pcl1 = np.fromfile(
            pclfiles[one], dtype=np.float32).reshape(-1, 4)[:, :3]
        pcl2 = np.fromfile(
            pclfiles[othor], dtype=np.float32).reshape(-1, 4)[:, :3]
        # self.draw_traj(self.odometry)
        pose1 = self.odometry[one]
        pose2 = self.odometry[othor]
        self.visual_pcl(pcl1, pcl2, np.eye(4), np.linalg.inv(pose1) @ pose2)

    def findNearest(self, pos, database, thresold=0.125):
        database = np.asarray(database).reshape(-1, 3)
        pos = np.asarray(pos).reshape(-1, 3)
        tree = KDTree(database)
        dist, ind = tree.query(pos, k=1)
        if dist[0] > thresold:
            return True
        return False

    @staticmethod
    def print_pcl_info(pcl,name='Point Cloud'):
        print(name+':')
        print('          :\tx\t\ty\t\tz\n',
              '      max:\t%4f\t%4f\t%4f\n' % (
                  np.max(pcl[:, 0]), np.max(pcl[:, 1]), np.max(pcl[:, 2])),
              '      min:\t%4f\t%4f\t%4f\n' % (
                  np.min(pcl[:, 0]), np.min(pcl[:, 1]), np.min(pcl[:, 2])),
              '   median:\t%4f\t%4f\t%4f\n' % (
                  np.median(pcl[:, 0]), np.median(pcl[:, 1]), np.median(pcl[:, 2])),
              '     mean:\t%4f\t%4f\t%4f\n' % (
                  np.mean(pcl[:, 0]), np.mean(pcl[:, 1]), np.mean(pcl[:, 2]))
              )

    def visual_pcl(self, pcl1, pcl2, T1=np.eye(4), T2=np.eye(4)):

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcl1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcl2)
        pcd1.paint_uniform_color([1, 0.706, 0])
        pcd2.paint_uniform_color([0, 0.651, 0.929])
        pcd1 = pcd1.transform(T1)
        pcd2 = pcd2.transform(T2)
        o3d.visualization.draw_geometries([pcd1, pcd2], window_name='Open3D Origin', width=1920, height=1080, left=50,
                                          top=50,
                                          point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

    def visual_pcl1(self, pcl1, T1=np.eye(4)):

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcl1)
        pcd1 = pcd1.transform(T1)
        o3d.visualization.draw_geometries([pcd1], window_name='Open3D Origin', width=1920, height=1080, left=50,
                                          top=50,
                                          point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

    def mkdir_safe(self, path):
        os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='./kitti/odometry', type=str)
    parser.add_argument('--target', default='../benchmark_datasets/kitti', type=str)
    parser.add_argument('--build_map', default=False, type=bool)
    parser.add_argument('--base_map_x', default=1000., type=float)
    parser.add_argument('--base_map_y', default=1000., type=float)
    parser.add_argument('--visual',default=False,type=bool)
    parser.add_argument('--show_map',default=False,type=bool)
    args = parser.parse_args()
    
    for k in vars(args).keys():
        print(k,':',vars(args)[k])
        
    print('\n')
    for seq_source_dir in glob(join(args.source, "[0-9][0-9]")):
        print(seq_source_dir)
        seq_target_dir = seq_source_dir.replace(args.source, args.target)
        preprocess = Prepross(seq_source_dir, seq_target_dir)
        
