import os
import sys
import tqdm

import numpy as np

if __name__ == '__main__':
    root_dir = sys.argv[1]
    xyzs = []
    for pose_file in tqdm.tqdm(os.listdir(os.path.join(root_dir, 'pose'))):
        pose = np.loadtxt(os.path.join(root_dir, f'pose/{pose_file}'))
        xyz = pose[:3, -1]
        xyzs.append(xyz)

    xyzs = np.array(xyzs)
    xyz_min = xyzs.min(axis=0)
    xyz_max = xyzs.max(axis=0)

    output = np.array([xyz_min, xyz_max])
    np.savetxt(os.path.join(root_dir, 'cam_bbox.txt'), output)

