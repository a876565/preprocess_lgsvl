import numpy as np
import os
from tqdm import tqdm
from glob import glob
import pdb
def check_pcl_dir(dname):
    for fname in os.listdir(dname):
        src_name=os.path.join(dname,fname)
        pcl=np.fromfile(src_name,dtype=np.float32).reshape(-1,3)
        
        impossible_points=pcl[pcl[:,2]<-2.5/30]
        if impossible_points.any():
            print(src_name)

source_dirs='/media/mustar/DATA4/LGSVL/main/newdata'

for seq_source_dir in tqdm(glob(os.path.join(source_dirs, "[0-9][0-9]"))):
    print(seq_source_dir)
    for p in os.listdir(seq_source_dir):
        fname=os.path.join(seq_source_dir,p)
        if os.path.isdir(fname):
            check_pcl_dir(fname)