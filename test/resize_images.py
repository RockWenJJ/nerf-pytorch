import numpy as np
import cv2
import os


def resize_images(images_path, images_f, factor):
    outpath = images_path+"_%d"%factor
    os.makedirs(outpath, exist_ok=True)
    for image_f in images_f:
        img = cv2.imread(os.path.join(images_path, image_f))
        height, width = img.shape[:2]
        img_r = cv2.resize(img, (int(width/factor), int(height/factor)))
        cv2.imwrite(os.path.join(outpath, image_f), img_r)
        
factor = 8
images_path = "./data/nerf_llff_data/apple_banana/images"
images_f = os.listdir(images_path)

resize_images(images_path, images_f, factor)