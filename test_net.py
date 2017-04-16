#!/usr/bin/env python

#!/usr/bin/env python

import numpy as np
from gen_solver import get_dataset_size
from net_generator_exp_num import NoLMDB_Net
from util import ParseParams,  load_image_mean, safe_mkdir
import math
import fileinput
import sys
sys.path.append('/opt/caffe/python/')


from skimage.io import imread, imsave

import caffe

caffe.set_mode_gpu()
caffe.set_device(3)

# def test():
#   size = get_dataset_size(dataset="rtsd-r1", phase="test", mode="orig")
#   net = caffe.Net('./Prototxt/experiment_10/rtsd-r1/orig/trial_1/test.prototxt',1,
#                   weights='./snapshots/experiment_10/rtsd-r1/orig/trial_1/snap_iter_2500.caffemodel')
#   #net = caffe.Net('./Prototxt/experiment_10/rtsd-r1/AHE/trial_1/test.prototxt',
#    #               './snapshots/experiment_10/rtsd-r1/AHE/trial_1/snap_iter_2500.caffemodel', caffe.TEST)



def set_batch_size(n, model):
    tmp = ""
    with  open(model, "r") as file:
        for line in file:
            if "batch_size" in line:
                s = line.split(":")
                tmp += line.replace("batch_size: {}".format(int(s[1])), "batch_size: {}".format(n)) 
            else:
                tmp += line
    
    with  open(model, "w") as file:
        file.write(tmp)
    print("batch size has been set to {}".format(n))


def load_net(exp_num, dataset, mode, trial, phase, batch_size=1):
    model = './Prototxt/experiment_{}/{}/{}/trial_{}/{}.prototxt'.format(exp_num, dataset, mode, trial, phase)
    set_batch_size(batch_size, model)

    weights = './snapshots/experiment_{}/{}/{}/trial_{}/snap_iter_2500.caffemodel'.format(exp_num, dataset, mode, trial)
    net = caffe.Net(model,1, weights=weights)

    return net





def test():
    exp_num = 10
    dataset = "rtsd-r1"
    phase = "test"
    mode = "orig"
    trial = 1
    size = get_dataset_size(dataset=dataset, phase=phase, mode=mode)

    net = load_net(exp_num, dataset, mode, trial, phase)
 

    sum = 0
    for i in range (size):
        if i % 100 == 0:
            print("image in proccess: {}".format(i))
        out = net.forward()
        acc =net.blobs["accuracy_1"].data
        sum += acc
    print("average = {}".format(sum / size))




def LoadWithoutLMDB(exp_num, dataset, mode, trial, phase, batch_size=1):
    model = CreateNoLMDB(exp_num, dataset, mode, phase)
    # set_batch_size(batch_size, model)

    weights = './snapshots/experiment_{}/{}/{}/trial_{}/snap_iter_2500.caffemodel'.format(exp_num, dataset, mode, trial)
    net = caffe.Net(model,1, weights=weights)

    return net




def CreateNoLMDB(exp_num, dataset, mode, phase):
    #reading experiment parameters
    param_path = "./logs/experiment_{}/{}/{}/params.txt".format(exp_num, dataset, mode)
    args = ParseParams(param_path)

    #creating achitecture without specified data path and labels
    content = str(NoLMDB_Net(args, dataset, mode, phase))
    path = './Prototxt/experiment_{}/{}/{}/{}_no-lmdb.prototxt'.format(exp_num, dataset, mode, phase)
    with open(path, "w") as out:
        out.write(content)

    return path



def prepare(net, rootpath, phase, image_name):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    mean_path = '{}/{}/mean.txt'.format(rootpath, phase)
    mean = load_image_mean(mean_path)
    # b = map(int, mean)[0]
    # g = map(int, mean)[1]
    # r = map(int, mean)[2]
    # mean_value = np.array(  [r,g,b])
    mean_value = np.array(map(int, mean))

    transformer.set_mean('data', mean_value)

    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255)

    img_path = "{}/{}/{}".format(rootpath, phase, image_name)
    img = caffe.io.load_image(img_path)[3:-3, 3:-3, :]
    net.blobs['data'].data[...] = transformer.preprocess('data', img)

    



def TestCommitee(exp_num, dataset):
    phase = "test"
    mode = "orig"
    trial = 1
    size = get_dataset_size(dataset=dataset, phase=phase, mode=mode)
    net = LoadWithoutLMDB(exp_num, dataset, mode, trial, phase)

    
    rootpath = "../local_data/{}/{}".format(dataset, mode)
    total = 0
    correct = 0
    with open('{}/gt_{}.txt'.format(rootpath, phase), 'r') as f:
        for image_name,clid in [x.replace('\n', '').split(' ') for x in f]:
            clid = int(clid)
            total +=1
            prepare(net, rootpath, phase, image_name)
            out = net.forward()
            # print(net.blobs["softmax"].data)
            prediction = np.argmax(net.blobs["softmax"].data)
            print(prediction, "   ", clid)
            if prediction == clid:
                print("correct")
                correct +=1
            else:
                print(image_name)

        print("Accuracy:  ", float(correct)/total)
            
            # exit()

    

def test2(exp_num, dataset):
    phase = "test"
    mode = "orig"
    trial = 1
    size = get_dataset_size(dataset=dataset, phase=phase, mode=mode)
    net = LoadWithoutLMDB(exp_num, dataset, mode, trial, phase)

    rootpath = "../local_data/{}/{}".format(dataset, mode)
    sum = 0.0
    total = 0
    correct = 0
    src = "../local_data/rtsd-r1/orig/test.txt"
    with open(src) as f:
    # for i in range (size):
        for line in f:
            
            # if i % 100 == 0:
                # print("image in proccess: {}".format(i))
            out = net.forward()
            
            prediction = np.argmax(net.blobs["softmax"].data)
            label = int(net.blobs["label"].data[0])
            # print(prediction, "  ", label)
            if prediction == label:
                sum += 1.0
                print("correct")
                correct +=1
            else:
                print line


        print("Accuracy:  ", float(correct)/total)
    print("average = {}".format(sum / size))



test2(10, "rtsd-r1")
    # sum = 0
    # for i in range (size):
    #     if i % 100 == 0:
    #         print("image in proccess: {}".format(i))
    #     out = net.forward()
    #     acc =net.blobs["accuracy_1"].data
    #     sum += acc
    # print("average = {}".format(sum / size))

  
# TestCommitee(10, "rtsd-r1")                   
