#!/usr/bin/env python

import cv2
import numpy as np
from glob import glob
import os.path as osp
from os import makedirs 

def histeq(img):
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    ycbcr[:,:,0] = cv2.equalizeHist(ycbcr[:,:,0])
    return cv2.cvtColor(ycbcr, cv2.COLOR_YCR_CB2BGR)

def crop(img, x1, y1, x2, y2, expand = 0):
    dx, dy = x2 - x1, y2 - y1
    dmax = max(dx, dy)
    scale = 48.0 / dmax
    interpolation = cv2.INTER_LANCZOS4
    if scale < 0.5:
        interpolation = cv2.INTER_AREA
    rescaled = cv2.resize(img, (0, 0), fx = scale, fy = scale, interpolation = interpolation)

    origin = scale * (x1 + x2) / 2.0, scale * (y1 + y2) / 2.0
    radius = 24 + expand

    pad = 0
    if origin[0] - radius < 0:
        pad = radius - origin[0]
    if origin[1] - radius < 0:
        pad = max(pad, radius - origin[1])
    if int(origin[0] + radius) >= rescaled.shape[1]:
        pad = max(pad, origin[0] + radius - rescaled.shape[1])
    if int(origin[1] + radius) >= rescaled.shape[0]:
        pad = max(pad, origin[1] + radius - rescaled.shape[0])
    pad = int(pad) + 3

    origin = map(int, origin)
    replicate = cv2.copyMakeBorder(rescaled, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return replicate[pad + origin[1] - radius : pad + origin[1] + radius, pad + origin[0] - radius : pad + origin[0] + radius]

def safe_mkdir(directory_name):
    if not osp.exists(directory_name):
        makedirs(directory_name)

def process_train():
    rootpath = "GTSRB/Final_Training/Images"
    outpath = "gtsrb_processed"
    safe_mkdir(outpath)
    labels = open("{}/gt_train.txt".format(outpath), 'w')
    mean = np.zeros((3, 54, 54), dtype=np.float32)
    total_images = 0
    for dir in glob('{}/*'.format(rootpath)):
        class_id = osp.basename(dir)
        markup = open('{}/{}/GT-{}.csv'.format(rootpath, class_id, class_id), 'r').readlines()[1:]
        image_id = 0
        for image_name,w,h,x1,y1,x2,y2,clid in [x.replace('\r\n', '').split(';') for x in markup]:
            w, h, x1, y1, x2, y2, clid = map(int, [w, h, x1, y1, x2, y2, clid])
            img = cv2.imread("{}/{}/{}".format(rootpath, class_id, image_name))
            if (h, w) != img.shape[:2]:
                print("Incorrent markup file, image size mismatch")
            patch = histeq(crop(img, x1, y1, x2, y2, expand = 3))
            # image mean
            mean[0] += patch[:, :, 0]
            mean[1] += patch[:, :, 1]
            mean[2] += patch[:, :, 2]
            #
            labels.write("{}/{}.png {}\n".format(clid, image_id, clid))
            safe_mkdir("{}/train/{}".format(outpath, clid))
            cv2.imwrite("{}/train/{}/{}.png".format(outpath, clid, image_id), patch)
            image_id = image_id + 1
        total_images += image_id

    mean[0], mean[1], mean[2] = mean[0] / float(total_images), mean[1] / float(total_images), mean[2] / float(total_images)
    b, g, r = np.mean(mean[0]), np.mean(mean[1]), np.mean(mean[2])
    open("{}/mean.txt".format(outpath), "w").write("{} {} {}".format(b, g, r))

def process_test():
    rootpath = "GTSRB/Final_Test/Images"
    outpath = "gtsrb_processed"
    safe_mkdir(outpath)
    labels = open("{}/gt_test.txt".format(outpath), 'w')
    markup = open('{}/GT-final_test.csv'.format(rootpath), 'r').readlines()[1:]
    image_id = 0
    for image_name,w,h,x1,y1,x2,y2,clid in [x.replace('\r\n', '').split(';') for x in markup]:
        w, h, x1, y1, x2, y2, clid = map(int, [w, h, x1, y1, x2, y2, clid])
        img = cv2.imread("{}/{}".format(rootpath, image_name))
        if (h, w) != img.shape[:2]:
            print("Incorrent markup file, image size mismatch")
        qq = crop(img, x1, y1, x2, y2, expand = 0)
        patch = histeq(crop(img, x1, y1, x2, y2, expand = 0))
        labels.write("{}/{}.png {}\n".format(clid, image_id, clid))
        safe_mkdir("{}/test/{}".format(outpath, clid))
        cv2.imwrite("{}/test/{}/{}.png".format(outpath, clid, image_id), patch)
        image_id = image_id + 1

process_train()
process_test()
