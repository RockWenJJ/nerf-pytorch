import os
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

path = "/home/wenjj/Documents/01_Projects/nerf-pytorch/data/nerf_llff_data/apple_banana/sparse"


images = os.path.join(path, 'images.txt')
cameras = os.path.join(path, 'cameras.txt')
# selected_index = list(range(1180, 1235, 2))
selected_index = list(range(1, 18))

cameras_result = {}
with open(cameras, 'r') as f:
    lines = f.readlines()
    for i in range(3, len(lines)):
        parts = lines[i].split(' ')
        width, height, focal = float(parts[2]), float(parts[3]), float(parts[4])
        index = int(parts[0])
        cameras_result[index] = np.array([width, height, focal])

images_result = {}
with open(images, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.endswith('.png\n') or line.endswith('.jpg\n'):
            parts = line.split(' ')
            index = int(parts[0])
            key = int(parts[-1].split('.')[0])
            # if key not in images_result.keys():
            #     images_result[key] = {}
            T = np.zeros((3, 5), dtype=np.float32)
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            r = R.from_quat([qx, qy, qz, qw])
            Rm = r.as_matrix()
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            T[:3, :3] = Rm
            T[:3, 3] = np.array([tx, ty, tz])
            T[:3, 4] = cameras_result[index]
            images_result[key] = T
            
            # print("test")

sorted(images_result)
keys = list(images_result.keys())
keys.sort()

num_img = len(selected_index)
poses_bounds = np.zeros((num_img, 17))
i = 0
for key in keys:
    if key == selected_index[i]:
        poses_bounds[i, :-2] = images_result[key].flatten()
        poses_bounds[i, -2:] = np.array([0.5, 10.])
        i += 1
np.save("poses_bounds.npy", poses_bounds)
# print("test")

# print("result")