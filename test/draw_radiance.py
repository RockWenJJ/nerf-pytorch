import os
import cv2
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

input_dir = "./logs/fern_220525/testset_200000"
raws = glob(os.path.join(input_dir, "raw_*.npy"))
zvals = glob(os.path.join(input_dir, "zval_*.npy"))

raws.sort()
zvals.sort()

cols = [260, 400]
rows = [280, 300]

def relu(m):
    m[m<0] = 0
    return m
raw2alpha = lambda raw, dists : 1. - np.exp(-relu(raw)*dists)
# raw2alpha = lambda raw, dists : np.exp(-relu(raw)*dists)

for i, (raw, zval) in tqdm(enumerate(zip(raws, zvals))):
    raw_data = np.load(raw)
    zval_data = np.load(zval)

    if i == 0:
        img = cv2.imread(os.path.join(input_dir, "%03d.png" % i))
        for col, row in zip(cols, rows):
            cv2.circle(img, (col, row), 3, (0, 0, 0), -1)
            dists = zval_data[..., 1:] - zval_data[..., :-1]
            dists = np.concatenate([dists, np.ones((list(dists.shape[:2]) + [1])) * 1e10], -1)
            rgb = 1 / (1 + np.exp(-raw_data[..., :3]))
            alpha = raw2alpha(raw_data[..., 3], dists)
            cum = np.cumprod(np.concatenate([np.ones((list(alpha.shape[:2]) + [1])), 1. - alpha + 1e-10], -1),
                                         -1)[..., :-1]
            weights = alpha * cum
            all_weights = np.sum(weights, -1)
            # print(all_weights)
            rgb_weight = weights[..., None] * rgb
    
            raw_r = rgb_weight[row, col, :, 0]
            raw_g = rgb_weight[row, col, :, 1]
            raw_b = rgb_weight[row, col, :, 2]
            # raw_alpha = rgb_weight[199, 199, :, 3]
            z_val = zval_data[row, col, :]
            plt.plot(z_val, alpha[row, col, :], 'k')
            plt.title("alpha")
            plt.show()
            plt.plot(z_val, cum[row, col, :], 'k')
            plt.title("cum")
            plt.show()
            plt.plot(z_val, weights[row, col, :], 'k--')
            plt.title("weights")
            plt.show()
            plt.plot(z_val, raw_r, 'r')
            plt.plot(z_val, raw_g, 'g')
            plt.plot(z_val, raw_b, 'b')
            plt.title("rgb")
            plt.show()
        cv2.imwrite(os.path.join(input_dir, "%03d_0.png" % i), img)
        break
    # print("test")