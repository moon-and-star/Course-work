#!/usr/bin/env python

#!/usr/bin/env python

from gen_solver import get_dataset_size
import math
import fileinput
import sys
sys.path.append('/opt/caffe/python/')


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


def DelLMDB(path):
    out = path.replace(".prototxt", "_no-LMDB.prototxt")
    with  open(path, "r") as fin, open(out, "w") as fout:
        for line in fin:
            if "source:" in line or "backend:" in line:
                pass
            else:
                fout.write(line)
    return out



def LoadWithoutLMDB(exp_num, dataset, mode, trial, phase, batch_size=1):
    path = './Prototxt/experiment_{}/{}/{}/trial_{}/{}.prototxt'.format(exp_num, dataset, mode, trial, phase)
    model = DelLMDB(path)
    set_batch_size(batch_size, model)

    weights = './snapshots/experiment_{}/{}/{}/trial_{}/snap_iter_2500.caffemodel'.format(exp_num, dataset, mode, trial)
    net = caffe.Net(model,1, weights=weights)

    return net

def TestCommitee(exp_num, dataset):
    phase = "test"
    mode = "orig"
    trial = 1
    size = get_dataset_size(dataset=dataset, phase=phase, mode=mode)

    net = LoadWithoutLMDB(exp_num, dataset, mode, trial, phase)

    exit()
    markup = open('{}/gt_{}.csv'.format(rootpath, phase), 'r').readlines()
 

    sum = 0
    for i in range (size):
        if i % 100 == 0:
            print("image in proccess: {}".format(i))
        out = net.forward()
        acc =net.blobs["accuracy_1"].data
        sum += acc
    print("average = {}".format(sum / size))

  
TestCommitee(10, "rtsd-r1")                   
# test()
path = './Prototxt/experiment_///trial_/.prototxt'
print(path)
out = path.replace(".prototxt", "_no-LMDB.prototxt")
print (out)