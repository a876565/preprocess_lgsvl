import os
import pandas as pd
from tqdm import tqdm
from glob import glob
import pdb
import numpy as np
def rewrite_csv(fname):
    df=pd.read_csv(fname,sep=',')
    tss=[ts+100000 if ts< 100000 else ts for ts in df['timestamp']]
    df['timestamp']=tss
    df.to_csv(fname,index=False, sep=',')

def rename_bins(dname):
    for fname in os.listdir(dname):
        p=fname.rfind('.')
        fid=fname[:p]
        fext=fname[p:]
        try:
            iid=int(fid)
            fid=str(iid+100000 if iid<100000 else iid)
        except ValueError as e:
            continue
        src_name=os.path.join(dname,fname)
        dst_name=os.path.join(dname,fid+fext)
        os.rename(src_name,dst_name)

def rewrite_bins(dname):
    for fname in os.listdir(dname):
        src_name=os.path.join(dname,fname)
        np.fromfile(src_name,dtype=np.float32).astype(np.float64).tofile(src_name)

def rewrite_bins2(dname):
    for fname in os.listdir(dname):
        src_name=os.path.join(dname,fname)
        pcl=np.fromfile(src_name,dtype=np.float64)
        for i in range(3):
            pcl[:,i]-=np.mean(pcl[:,i])
        pcl.tofile(src_name)

source_dirs='/media/mustar/DATA4/LGSVL/benchmark_datasets/lgsvl'

for seq_source_dir in tqdm(glob(os.path.join(source_dirs, "[0-9][0-9]"))):
    print(seq_source_dir)
    for p in os.listdir(seq_source_dir):
        fname=os.path.join(seq_source_dir,p)
        if os.path.isdir(fname):
            rename_bins(fname)
            rewrite_bins2(fname)
        elif fname.endswith('.csv'):
            rewrite_csv(fname)