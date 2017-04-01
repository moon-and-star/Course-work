#!/usr/bin/env python

import cv2
import PIL.ImageOps
import numpy as np
from glob import glob
import os.path as osp
from os import makedirs 
import copy
import random

from skimage.io import imread, imsave
from skimage.transform import rotate, rescale
from skimage.color import rgb2lab, lab2rgb 
from skimage.exposure import equalize_adapthist, equalize_hist


def histeq(img):
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    ycbcr[:,:,0] = cv2.equalizeHist(ycbcr[:,:,0])
    return cv2.cvtColor(ycbcr, cv2.COLOR_YCR_CB2BGR)



def autoContrast(img, frac=0.1):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Y = lab[:,:, 0].astype(np.float32)
    Ysorted = np.sort(Y.reshape(-1))
    m = int(len(Ysorted) * frac)
    Ysorted = Ysorted[m:-m]

    ymin, ymax = Ysorted[0], Ysorted[-1]
    alpha = 255. / (ymax-ymin)
    # print((Y - ymin) * alpha)
    # print ((ymax - ymin) * alpha )

    Y =  ((Y - ymin) * alpha)
    Y[Y < 0] = 0
    Y[Y > 255] = 255
    lab[:,: ,0] = Y.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)



def adaHE(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2,2))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)



def contrastNorm(img):
    size = 5
    sX = size//6   
    blur = cv2.GaussianBlur(img,(size,size),sigmaX=sX)
    V = img - blur.astype(float)
    
    sigma = (cv2.GaussianBlur((V ** 2), (size, size),sigmaX=sX)) ** 0.5
    mean = np.mean(sigma)
    Y = (V // np.fmax(sigma, mean)) 
    Y -= np.min(Y)
    img =  Y * 255. / np.max(Y)
    return img.astype(np.uint8)



def crop(img, x1, y1, x2, y2, expand = 0):
    dx, dy = x2 - x1, y2 - y1
    x_scale, y_scale = 48.0 / dx, 48.0 / dy
    interpolation = cv2.INTER_LANCZOS4
    if x_scale < 0.5 or y_scale < 0.5:
        interpolation = cv2.INTER_AREA
    rescaled = cv2.resize(img, (0, 0), fx = x_scale, fy = y_scale, interpolation = interpolation)

    origin = x_scale * (x1 + x2) / 2.0, y_scale * (y1 + y2) / 2.0
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
    origin = list(map(int, origin))
    replicate = cv2.copyMakeBorder(rescaled, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return replicate[pad + origin[1] - radius : pad + origin[1] + radius, pad + origin[0] - radius : pad + origin[0] + radius]


def safe_mkdir(directory_name):
    if not osp.exists(directory_name):
        makedirs(directory_name)



def identity(a):
    return a


random.seed()



# add or delete borders of SQUARE image
def to54shape(img):
    m, n, k = img. shape
    tmp = copy.copy(img)
    # tmp = Image.fromarray(np.uint8(img*255))
    if m < 54:
        diff = (54 - m)
        bord_1 = int(diff / 2)
        bord_2 = diff - bord_1
        rows_1, rows_2 = np.zeros((bord_1, n, k)), np.zeros((bord_2, n, k))
        tmp = np.vstack([rows_1,tmp, rows_2])

        m, n, k = tmp.shape
        cols_1, cols_2 = np.zeros((m, bord_1, k)), np.zeros((m, bord_2, k))
        tmp = np.hstack([cols_1, tmp, cols_2])

    elif m > 54:
        diff = (m - 54)
        bord_1 = int(diff / 2)
        bord_2 = diff - bord_1
        tmp = img[bord_1: m - bord_2, bord_1 : n - bord_2]

    return tmp


#returns [0..255] image
def randTrans(img, scaled=True, rotated=True):
    # generates random number in (0.9, 1.1)
    res = img
    # m, n, _ = 
    if scaled:
        scale = random.random() / 5 + 0.9
        res = rescale(res, scale)

    if rotated:
        angle = (random.random() - 0.5) * 10
        res = rotate(res, angle)

    res = to54shape(res)

    #converting to uints
    if np.amax(res) <= 1:
        if np.amax(res) == 0:
            print("WARNING: zero max in image")
        res *= 255

    res = res.astype(np.uint8)
    return res


operations = {
    "orig" : identity, 
    "histeq" : histeq, 
    "AHE" : adaHE, 
    "imajust" : autoContrast, 
    "CoNorm" : contrastNorm
}




rate = 100

def process(rootpath, outpath, phase, mode):
    safe_mkdir(outpath) #create outpath
    labels = open("{}/gt_{}.txt".format(outpath, phase), 'w') #create file to write labels in
    markup = open('{}/gt_{}.csv'.format(rootpath, phase), 'r').readlines() # open file to read labels (and, may be coords)

    mean = np.zeros((3, 54, 54), dtype=np.float32)
    image_id = 0
    total_images = 0
    for image_name,clid in [x.replace('\r\n', '').split(',') for x in markup[1:]]:
        # print (image_id)
        # if image_id > 100:
        #     break
        if image_id % rate == 0:
            print(image_name)

        clid = int(clid)
        img = cv2.imread("{}/{}/{}".format(rootpath,phase,image_name))

        m, n, _ = img.shape
        patch = operations[mode](crop(img, x1=0, y1=0, y2 = m - 1, x2=n - 1, expand = 3))
        for i in range(1):
            transformed = randTrans(patch, scaled=False, rotated=False)
            mean[0] += transformed[:, :, 0]
            mean[1] += transformed[:, :, 1]
            mean[2] += transformed[:, :, 2]

           
            labels.write("{}/{}.png {}\n".format(clid, image_id, clid))
            safe_mkdir("{}/{}/{}".format(outpath, phase, clid))
            cv2.imwrite("{}/{}/{}/{}.png".format(outpath, phase, clid, image_id), transformed)
            image_id = image_id + 1

    total_images = image_id
    with open("{}/{}_size.txt".format(outpath, phase), 'w') as f:
        f.write(str(total_images))

    mean[0] = mean[0] / float(total_images)
    mean[1] = mean[1] / float(total_images)
    mean[2] = mean[2] / float(total_images)

    #caffe works with openCV, so the order of channels is BGR
    b, g, r = np.mean(mean[0]), np.mean(mean[1]), np.mean(mean[2]) 
    open("{}/{}/mean.txt".format(outpath, phase), "w").write("{} {} {}".format(b, g, r))

    if max(b, g, r) < 50:
        print('WARNING: low mean values\nb={}, g={}, r={}'.format(b,g,r))
        print('rootpath={}\n outpath={}\n mode={}\n phase={}'.format(rootpath,outpath, mode, phase))






def launch():
    # modes = ['orig']
    # for dataset in ["rtsd-r1"]:
        # for phase in ["test"]:
    modes = ["orig", "histeq", "AHE", "imajust", "CoNorm" ]
    for dataset in ["rtsd-r1","rtsd-r3"]:
        for phase in ["train", "test"]:
            rootpath = "../global_data/Traffic_signs/RTSD/classification/" + dataset

            for mode in modes:
                print("\nCurrent_path = {}\n".format(rootpath))
                print("Mode = {}\n".format(mode))

                outpath = "../local_data/"+ dataset + "/" + mode
                if phase == "train":
                    process(rootpath, outpath, phase, mode)
                elif phase == "test":
                    process(rootpath, outpath, phase, mode)


launch()

