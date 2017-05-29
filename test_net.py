#!/usr/bin/env python



import numpy as np
from gen_solver import get_dataset_size
from net_generator_exp_num import NoLMDB_Net, NumOfClasses
from util import ParseParams,  load_image_mean, safe_mkdir
import math
import fileinput
import sys
import argparse
sys.path.append('/opt/caffe/python/')


from skimage.io import imread, imsave
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

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
    model, args = CreateNoLMDB(exp_num, dataset, mode, phase)
    size = get_dataset_size(dataset=dataset, phase="train", mode=mode)
    iter_num = int(args.epoch * math.ceil( float(size) / args.batch_size))


    weights = './snapshots/experiment_{}/{}/{}/trial_{}/snap_iter_{}.caffemodel'.format(exp_num, dataset, mode, trial, iter_num)
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

    return path, args



def prepare(net, rootpath, phase, image_name):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    mean_path = '{}/{}/mean.txt'.format(rootpath, phase)
    mean = load_image_mean(mean_path)
    mean_value = np.array(map(int, mean))

    transformer.set_mean('data', mean_value)

    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255)

    img_path = "{}/{}/{}".format(rootpath, phase, image_name)
    img = caffe.io.load_image(img_path)[3:-3, 3:-3, :]
    net.blobs['data'].data[...] = transformer.preprocess('data', img)

    



def Test2(exp_num, dataset):
    phase = "test"
    mode = "orig"
    trial = 1
    size = get_dataset_size(dataset=dataset, phase=phase, mode=mode)
    net = LoadWithoutLMDB(exp_num, dataset, mode, trial, phase)

    rootpath = "../local_data/{}/{}".format(dataset, mode)
    src = "{}/test.txt".format(rootpath)

    sum = 0.0
    with open(src) as f:
        for line in f:
            out = net.forward()
            
            prediction = np.argmax(net.blobs["softmax"].data)
            label = int(net.blobs["label"].data[0])
            # print(prediction, "  ", label)
            if prediction == label:
                sum += 1.0
            else:
                print("{}   prediction = {}". format(line, prediction))

    print("average = {}".format(sum / size))



def CommiteeOutput(exp_num, dataset, phase="test"):
    modes = ["orig", "histeq", "AHE", "imajust", "CoNorm" ]
    print(phase)
    # modes = ['orig']
    for mode in modes:
        num_of_nets = 5.0 * len(modes)
        size = get_dataset_size(dataset=dataset, phase=phase, mode=mode)
        classes = NumOfClasses(dataset)
        softmax = np.zeros((5, size, classes))


        for trial in range(5):
            net = LoadWithoutLMDB(exp_num, dataset, mode, trial + 1, phase)
            for i in range(size):
                if i % 100 == 0:
                    print(i)
                out = net.forward()
                softmax[trial][i] = net.blobs["softmax"].data
                # return softmax.sum(axis=0) / num_of_nets

        softmax = softmax.sum(axis=0) / num_of_nets
        print(softmax.shape)
    
    return softmax



def InitAnswers(length):
    res = []
    for i in range(length):
        tmp = {}
        tmp["correct"] = 0
        tmp['total'] = 0

        res += [tmp]  
    return res  


def ClassAccuracies(answers):
    res = {}
    for i in range(len(answers)):
        if answers[i]['total'] > 0:
            acc = float(answers[i]['correct']) / answers[i]['total']
            res[str(i)] = acc
    return res


def AccuracyAndErrors(exp_num, dataset, phase, softmax, verbose=True):
    mode = "orig"
    rootpath = "../local_data/{}/{}".format(dataset, mode)
    src = "{}/{}.txt".format(rootpath, phase)  #gt file path      
    with open(src) as f:
        lines = f.readlines()


    sum = 0.0
    size = get_dataset_size(dataset=dataset, phase=phase, mode=mode)
    print(size)
    path = "./logs/experiment_{}/{}/misclassified_{}.txt".format(exp_num, dataset, phase)
    class_answers = InitAnswers(NumOfClasses(dataset))
  
    with open(path, 'w') as out:
        for i in range(size):
            label = int(lines[i].replace("\n", "").split(" ")[1])
            prediction = np.argmax(softmax[i])
            class_answers[label]['total'] += 1

            if label == prediction:
                class_answers[label]['correct'] += 1
                sum += 1.0
            else:
                line = lines[i].replace("\n", "")
                content = "name = {}   prediction = {}".format(line, prediction)
                out.write(content + '\n')
                if verbose == True:
                    print(content)


    path = "./logs/experiment_{}/{}/predictions_{}.txt".format(exp_num, dataset, phase)
    with open(path, 'w') as out:
        for i in range(size):
            label = int(lines[i].replace("\n", "").split(" ")[1])
            prediction = np.argmax(softmax[i])
            line = lines[i].replace("\n", "")
            content = "{}   prediction = {}".format(line, prediction)
            out.write(content + '\n')
            

    accuracies = ClassAccuracies(class_answers)
    return sum / size, accuracies
                



def TestCommitee(exp_num, dataset):
    phases = ["test", "train"]
    for phase in phases:
        softmax = CommiteeOutput(exp_num, dataset, phase)
        acc, cl_acc = AccuracyAndErrors(exp_num, dataset, phase, softmax,verbose=False)
        
        path = "./logs/experiment_{}/{}/test_on_{}_results.txt".format(exp_num, dataset, phase)
        with open(path, 'w') as out:
            print("Accuracy: ", acc)
            out.write(str(acc)+ '\n')
            for key in sorted(cl_acc):
                content = "class_{}  acc = {}\n".format(key, cl_acc[key])  
                out.write(content)
                print(content)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("EXPERIMENT_NUMBER",type=int, 
                        help='the number of current experiment with nets ')
    args = parser.parse_args()
    TestCommitee(args.EXPERIMENT_NUMBER, "RTSD")                   
