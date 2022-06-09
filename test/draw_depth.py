import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

data_dir = "/home/wenjj/Documents/01_Projects/nerf-pytorch/logs/apple_banana_220606/testset_150000"
depth_files = glob(os.path.join(data_dir, "depth_*.npy"))

for depth_file in tqdm(depth_files):
    basename = os.path.basename(depth_file)
    basename = basename.split('.')[0]
    depth = np.load(depth_file)
    # set the colormap range
    norm1 = mcolors.Normalize(vmin=0, vmax=1)
    im = plt.imshow(depth, norm=norm1)
    plt.colorbar(im, ticks=np.linspace(0, 1, 5))
    # plt.imshow(depth)
    plt.savefig(os.path.join(data_dir, "%s.png"%basename))
    plt.close()
    # print(os.path.join(data_dir, "%s.png"%basename))