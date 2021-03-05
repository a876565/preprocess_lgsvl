import open3d as o3d
from normalize_pcl import normalize


def mvpcls(source_dir,target_dir):
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/media/mustar/DATA4/LGSVL/benchmark_datasets/kitti/odometry', type=str)
    parser.add_argument('--target', default='kitti_new', type=str)
    parser.add_argument('--rand_map_shift', default=True, type=bool)
    parser.add_argument('--visual',default=False,type=bool)
    args = parser.parse_args()
    for k in list(vars(args).keys()):
        print(k,':',vars(args)[k])
    print('\n')
    for seq_source_dir in tqdm(glob(join(args.source, "[0-9][0-9]"))):
        tqdm.write(seq_source_dir)
        seq_target_dir = seq_source_dir.replace(args.source, args.target)
        mvpcls(seq_source_dir, seq_target_dir)