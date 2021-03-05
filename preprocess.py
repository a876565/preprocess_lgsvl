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


class Prepross:
    def __init__(self, source_dir, target_dir):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.mkdir_dirs()
        self.save_trans()
        self.make_gt()
        print('[OK] make_gt')
        self.generate_map()
        print('[OK] generate_map')
        self.generate_samples()
        print('[OK] generate_samples')
        self.generate_pcls(
            self.tar_test_dir, self.timestamp_list_test, self.records_test[:,0:3], self.records_test[:,3:6], [60, 60, 30])
        self.generate_pcls(
            self.tar_train_dir, self.timestamp_list_train, self.records_train[:,0:3], self.records_train[:,3:6], [60, 60, 30])
        print('[OK] generate_pcls')

        self.save_samples_csv()
        print('[OK] save_samples_csv')
        # self.move_pcl(),
        # self.normalize_pcl()
        # self.check()

    def mkdir_dirs(self):

        self.source_odometry_lidar = join(self.source_dir, "odometry_lidar")
        self.mkdir_safe(self.source_odometry_lidar)
        self.source_calib = join(self.source_dir, "calib")
        self.source_velodyne = join(self.source_dir, "velodyne")
        self.source_gps = join(self.source_dir, "gps")

        self.mkdir_safe(self.target_dir)
        self.tar_train_loc = join(
            self.target_dir, "./pointcloud_locations_20m_10overlap.csv")
        self.tar_test_loc = join(
            self.target_dir, "./pointcloud_locations_20m.csv")
        self.tar_trans_loc = join(
            self.target_dir, "./pointcloud_transforms.csv")
        self.tar_train_dir = os.path.join(
            self.target_dir, 'pointcloud_20m_10overlap')
        self.mkdir_safe(self.tar_train_dir)
        self.tar_test_dir = os.path.join(self.target_dir, 'pointcloud_20m')
        self.mkdir_safe(self.tar_test_dir)

    def make_gt(self):
        self.Tr_imu_to_velo, self.Tr_velo_to_imu = self.get_vel2cam()
        gps_files = sorted(glob(join(self.source_gps, '*')))
        T_init = np.eye(4)
        init = True
        self.odometry = []
        for gps_file in gps_files:
            T = self.get_pose(gps_file)
            if init:
                T_init = T
                init = False
            #T = np.linalg.inv(T_init)@T
            #T =  T
            T = self.Tr_velo_to_imu @ T #@ self.Tr_imu_to_velo
            self.odometry.append(T)
        self.write_odo_file(self.odometry)

    def generate_map(self):
        # for pcl_file in sorted(glob(os.path.join(self.source_velodyne, '*.bin'))):

        positions = [[0, 0, 0]]
        timestamp_list = []
        poses = []
        gps_files = sorted(glob(os.path.join(self.source_gps, '*')))
        for gps_file, pose in zip(gps_files, self.odometry):
            timestamp = os.path.splitext(gps_file)[0].split('/')[-1]
            northing, easting, altitude, orientation = self.get_WGS_84(
                gps_file)
            position = [northing, easting, altitude]
            if self.findNearest(position, positions, 50):
                positions.append(position)
                timestamp_list.append(timestamp)
                poses.append(pose)
        map = []
        for timestamp, pose in zip(timestamp_list, poses):
            pcl_file = os.path.join(self.source_velodyne, timestamp + '.bin')
            pcl = np.fromfile(pcl_file, dtype=np.float32).reshape(-1, 4)[:, :3]
            pcl = pcl @ pose[:3, :3].T + pose[:3, 3].reshape(1, 3)  #
            map.append(pcl)

        map = np.concatenate(map, axis=0)
        self.map = map
        map_pcd = o3d.geometry.PointCloud()
        map_pcd.points = o3d.utility.Vector3dVector(map)
        map_pcd = map_pcd.voxel_down_sample(voxel_size=0.2)
        #o3d.visualization.draw_geometries(
        #    [map_pcd], window_name='Open3D Origin')

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
        self.records_train = []
        self.records_test = []

        self.timestamp_list_train = []
        self.timestamp_list_test = []
        gps_files = sorted(glob(os.path.join(self.source_gps, '*')))
        base_map_shift=np.array([6,6,0])

        if args.rand_map_shift:
            rand_map_shift=2*(np.random.rand(3)-0.5)*base_map_shift
        else:
            rand_map_shift=np.zeros(3)

        for gps_file in gps_files:
            timestamp = str(int(os.path.splitext(gps_file)[0].split('/')[-1])+100000)
            #northing, easting, altitude, orientation = self.get_WGS_84(gps_file)
            #position = [northing, easting, altitude]
            gps = self.get_transform(gps_file)
            position = [p1+p2 for p1,p2 in zip(gps[:3],rand_map_shift)]
            rotvec = [0, 0, gps[5]*Degree2Rad]
            scale = [1/60, 1/60, 1/30]
            record = position+rotvec+scale
            if self.findNearest(position, self.positions_train, 10):
                self.positions_train.append(position)
                self.records_train.append(record)
                self.timestamp_list_train.append(timestamp)
            elif self.findNearest(position, self.positions_test, 40):
                self.positions_test.append(position)
                self.records_test.append(record)
                self.timestamp_list_test.append(timestamp)

        # 去除第一个[0,0,0]
        self.positions_train = np.asarray(self.positions_train[1:]).reshape(-1, 3)
        self.positions_test = np.asarray(self.positions_test[1:]).reshape(-1, 3)

        self.records_train = np.asarray(self.records_train).reshape(-1, 9)
        self.records_test = np.asarray(self.records_test).reshape(-1, 9)

    def save_samples_csv(self):
        # 字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'timestamp': self.timestamp_list_train,
                                  'northing': self.records_train[:, 0].reshape(-1),
                                  'easting': self.records_train[:, 1].reshape(-1),
                                  'altitude': self.records_train[:, 2].reshape(-1),
                                  'r1': self.records_train[:, 3].reshape(-1),
                                  'r2': self.records_train[:, 4].reshape(-1),
                                  'r3': self.records_train[:, 5].reshape(-1),
                                  's1': self.records_train[:, 6].reshape(-1),
                                  's2': self.records_train[:, 7].reshape(-1),
                                  's3': self.records_train[:, 8].reshape(-1),
                                  })
                                  
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv(self.tar_train_loc, index=False, sep=',')

        dataframe = pd.DataFrame({'timestamp': self.timestamp_list_test,
                                  'northing': self.records_test[:, 0].reshape(-1),
                                  'easting': self.records_test[:, 1].reshape(-1),
                                  'altitude': self.records_test[:, 2].reshape(-1),
                                  'r1': self.records_test[:, 3].reshape(-1),
                                  'r2': self.records_test[:, 4].reshape(-1),
                                  'r3': self.records_test[:, 5].reshape(-1),
                                  's1': self.records_test[:, 6].reshape(-1),
                                  's2': self.records_test[:, 7].reshape(-1),
                                  's3': self.records_test[:, 8].reshape(-1), })
        dataframe.to_csv(self.tar_test_loc, index=False, sep=',')

    def move_pcl(self):
        for timestamp in self.timestamp_list_train:
            pcl_file = os.path.join(self.source_velodyne, timestamp + '.bin')
            pcl_file_new = os.path.join(self.tar_train_dir, timestamp + '.bin')

            shutil.copy(pcl_file, pcl_file_new)
        for timestamp in self.timestamp_list_test:
            pcl_file = os.path.join(self.source_velodyne, timestamp + '.bin')
            pcl_file_new = os.path.join(self.tar_test_dir, timestamp + '.bin')
            shutil.copy(pcl_file, pcl_file_new)

    def select_points_in(self, center, box, keepground=True):
        cloud = []
        ground = []

        mapped=self.map-center
        sel1=mapped[:,0]<box[0]
        sel2=mapped[:,0]>-box[0]
        sel3=mapped[:,1]<box[1]
        sel4=mapped[:,1]>-box[1]
        sel5=mapped[:,2]<box[2]
        sel6=mapped[:,2]>self.imu_pos[2]+0.1

        sel=np.logical_and(sel1,sel2)
        sel=np.logical_and(sel,sel3)
        sel=np.logical_and(sel,sel4)
        ground=mapped[sel]

        sel=np.logical_and(sel,sel5)
        sel=np.logical_and(sel,sel6)
        cloud=mapped[sel]
        
        return cloud, ground

    def generate_pcls(self, dirpath, timestamps, positions,rotations, box, loginfo=False):
        if loginfo:
            print(' map info:')
            self.print_pcl_info(self.map)
        for (ts, pos,rot) in tqdm(zip(timestamps, positions,rotations), total=len(timestamps),desc=dirpath):
            if loginfo:
                print(" positon=", pos)
            pcl, pcl_g = self.select_points_in(pos, box)

            if pcl.any():
                pcl_norm,newpos = normalize(pcl,args.visual)
                pcl_file_new = os.path.join(dirpath, ts + '.bin')

                pcl_norm.astype(np.float64).tofile(pcl_file_new)
                pos+=newpos

                if loginfo:
                    self.print_pcl_info(pcl_norm)
                    self.visual_pcl1(pcl_norm)
        print('\n')

    def normalize_pcl(self):
        '''
        ranges = []
        for pcl_file in glob(os.path.join(self.tar_train_dir, '*.bin')):
            pcl = np.fromfile(pcl_file, dtype=np.float32).reshape(-1, 4)
            ranges.append(np.max(np.abs(pcl), axis=0).reshape(1,-1))
            # pcl = normalize(pcl)
            #pcl.tofile(pcl_file)
        ranges = np.concatenate(ranges, axis=0)
        ranges = np.mean(ranges, axis=0)
        '''
        # print("ranges x: {}, y: {}, z: {}".format(ranges[0], ranges[1], ranges[2]))

        for pcl_file in glob(os.path.join(self.tar_test_dir, '*.bin')):
            pcl = np.fromfile(pcl_file, dtype=np.float32).reshape(-1, 3)
            pdb.set_trace()
            self.visual_pcl1(pcl)
            #pcl = normalize(pcl)
            self.visual_pcl1(pcl)

            # pcl.tofile(pcl_file)

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

    def print_pcl_info(self, pcl):
        print('-----------------------------------------------------')
        print('normalized:\tx\t\ty\t\tz\n',
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

    def draw_traj(self, gts):
        traj = []
        for gt in gts:
            traj.append(gt[:3, 3].reshape(1, 3))
        traj = np.concatenate(traj, axis=0)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(traj)
        o3d.visualization.draw_geometries([pcd1], window_name='Open3D Origin')

    def get_WGS_84(self, file):
        f = open(file)
        line = f.readlines()[0]
        line = line.replace('GpsData(latitude=', '')
        line = line.replace(', longitude=', ' ')
        line = line.replace(', northing=', ' ')
        line = line.replace(', easting=', ' ')
        line = line.replace(', altitude=', ' ')
        line = line.replace(', orientation=', ' ')
        line = line.replace(')', '')
        gps = [float(e) for e in line.split(' ')]
        northing = gps[2]
        easting = gps[3]
        altitude = gps[-2]
        orientation = gps[-1]
        return northing, easting, altitude, orientation

    def get_transform(self, file):
        f = open(file)
        line = f.readlines()[1]
        line = line.replace('Transform(position=Vector(', '')
        line = line.replace('), rotation=Vector(', ' ')
        line = line.replace('))\n', '')
        line = line.replace(',', ' ')
        gps = [float(e) for e in line.split(' ') if e != '']

        return [gps[2], -gps[0], gps[1], gps[5], gps[3], -gps[4]]

    def get_pose(self, file):
        f = open(file)
        line = f.readlines()[1]
        line = line.replace('Transform(position=Vector(', '')
        line = line.replace('), rotation=Vector(', ' ')
        line = line.replace('))\n', '')
        line = line.replace(',', ' ')
        gps = [float(e) for e in line.split(' ') if e != '']
        T = np.eye(4)

        # changed
        y = gps[4] * Degree2Rad
        rotvec = np.asarray([gps[5] * Degree2Rad, gps[3]* Degree2Rad, -gps[4] * Degree2Rad])
        T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()

        T[0, 3] = gps[2]
        T[1, 3] = -gps[0]
        T[2, 3] = gps[1]
        return T

    def write_odo_file(self, odometry):
        file = open(join(self.source_odometry_lidar,
                         'odometry_lidar.txt'), 'w')
        for pose in odometry:
            for i in range(3):
                for j in range(4):
                    file.write(("%6e" % pose[i, j]))
                    if i != 2 or j != 3:
                        file.write(" ")
            file.write("\n")

    def get_vel2cam(self):
        calib_txt = join(self.source_dir, "calib", "000001.txt")
        file = open(calib_txt)
        lines = file.readlines()
        # changed
        line = lines[6]
        line = line.replace('Tr_imu_to_velo: ', '')
        eles = [float(e) for e in line.split(' ') if e != '']
        Tr_imu_to_velo = np.eye(4)
        for i in range(3):
            for j in range(4):
                Tr_imu_to_velo[i, j] = eles[i * 4 + j]
        self.imu_pos = [eles[3], eles[7], eles[11]]
        return np.linalg.inv(Tr_imu_to_velo), Tr_imu_to_velo

    def mkdir_safe(self, path):
        os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/media/mustar/DATA4/LGSVL/rawdata', type=str)
    parser.add_argument('--target', default='newdata6', type=str)
    parser.add_argument('--rand_map_shift', default=True, type=bool)
    parser.add_argument('--visual',default=True,type=bool)
    args = parser.parse_args()
    for k in list(vars(args).keys()):
        print(k,':',vars(args)[k])
    print('\n')
    for seq_source_dir in tqdm(glob(join(args.source, "[0-9][0-9]"))):
        print(seq_source_dir)
        seq_target_dir = seq_source_dir.replace(args.source, args.target)
        preprocess = Prepross(seq_source_dir, seq_target_dir)
