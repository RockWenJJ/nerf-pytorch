import numpy as np
import cv2
import os
import shutil

def resize_images(images_path, images_f, factor):
    outpath = images_path+"_%d"%factor
    os.makedirs(outpath, exist_ok=True)
    for image_f in images_f:
        img = cv2.imread(os.path.join(images_path, image_f))
        height, width = img.shape[:2]
        img_r = cv2.resize(img, (int(width/factor), int(height/factor)))
        if not image_f.endswith('.jpg'):
            origin_f = image_f
            image_f += '.jpg'
            shutil.move(os.path.join(images_path, origin_f), os.path.join(images_path, image_f))
        cv2.imwrite(os.path.join(outpath, image_f), img_r)
        
factor = 8
images_path = "./data/nerf_llff_data/pengcheng/images"
images_f = os.listdir(images_path)

resize_images(images_path, images_f, factor)