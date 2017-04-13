#!/usr/bin/env python

import sys
sys.path.append('/opt/caffe/python/')

from util import safe_mkdir, gen_parser, load_image_mean
from gen_solver import GenSingleNetSolver

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def conv_relu(n, name, bottom, kernel_size, num_output, stride=1, pad=1, group=1):
    conv = L.Convolution(
        bottom,
        kernel_size=kernel_size,
        stride=stride,
        num_output=num_output,
        pad=pad,
        group=group,
        weight_filler = dict(type = 'xavier'),
        name = name,
    )

    setattr(n, name, conv)

    return conv, L.ReLU(conv, in_place = True, name = "{}_relu".format(name))


def conv_stanh(n, name, bottom, kernel_size, num_output, stride=1, pad=1, group=1):
    conv = L.Convolution(
        bottom,
        kernel_size=kernel_size,
        stride=stride,
        num_output=num_output,
        pad=pad,
        group=group,
        weight_filler = dict(type = 'xavier'),
        name = name,
    )

    setattr(n, name, conv)

    
    scale1 = L.Scale(conv, in_place = True, name = "{}_prescale".format(name),
                     param=dict(lr_mult=0, decay_mult=0),
                     scale_param=dict(filler=dict(value=0.6666), bias_term=False)
                     )
    tanh =  L.TanH(scale1, in_place = True, name = "{}_sTanH".format(name))
    scale2 = L.Scale(tanh, in_place = True, name = "{}_postscale".format(name),
                     param=dict(lr_mult=0, decay_mult=0),
                     scale_param=dict(filler=dict(value=1.7159), bias_term=False)
                     )
    # scale2 =  L.TanH(conv, in_place = True, name = "{}_sTanH".format(name))
    return conv, scale2


def conv1(n, name, bottom, num_output, kernel_size = 3, pad = None, activ = "relu"):
    if pad is None: pad = kernel_size / 2
    if activ == "relu":
        conv1, relu1 = conv_relu(n, "{}".format(name), bottom, kernel_size, num_output, pad = pad)
        return relu1
    elif activ == "scaled_tanh":
        conv1, scale2 = conv_stanh(n, "{}".format(name), bottom, kernel_size, num_output, pad = pad)
        return scale2
    




def maxpool(name, bottom, kernel_size = 2, stride = 2):
    return L.Pooling(bottom, kernel_size = kernel_size, stride = stride, pool = P.Pooling.MAX, name = name)

def avepool(name, bottom, kernel_size = 2, stride = 2):
    return L.Pooling(bottom, kernel_size = kernel_size, stride = stride, pool = P.Pooling.AVE, name = name)

def fc(name, bottom, num_output, activ="relu"):
    fc = L.InnerProduct(
        bottom,
        num_output = num_output,
        weight_filler = dict(type = 'xavier'),
        name = "{}_{}".format(name, num_output)
    )

    if activ=="relu":
        return fc, L.ReLU(fc, in_place = True, name = "{}_relu".format(name))

    elif activ=="scaled_tanh":
        scale1 = L.Scale(fc, in_place = True, name = "{}_prescale".format(name),
                         param=dict(lr_mult=0, decay_mult=0),
                         scale_param=dict(filler=dict(value=0.6666), bias_term=False))
        tanh =  L.TanH(scale1, in_place = True, name = "{}_sTanH".format(name))
        scale2 = L.Scale(tanh, in_place = True, name = "{}_postscale".format(name),
                         param=dict(lr_mult=0, decay_mult=0),
                         scale_param=dict(filler=dict(value=1.7159), bias_term=False))
        # scale2 =  L.TanH(fc, in_place = True, name = "{}_sTanH".format(name))
        return fc, scale2

    elif activ=="softmax":
        # return fc, L.Softmax(fc, in_place=True)
        return fc, L.Softmax(fc, in_place=False)
    else:
        return fc



def dropout(name, bottom, dropout_ratio):
    return L.Dropout(
        bottom,
        in_place = True,
        dropout_param = dict(dropout_ratio = dropout_ratio),
        name = name
    )

def accuracy(name, bottom, labels, top_k):
    return L.Accuracy(
        bottom,
        labels,
        accuracy_param = dict(top_k = top_k),
        # include = dict(phase = caffe_pb2.Phase.Value("TEST")),
        name = name
    )


def initWithData(lmdb, phase, batch_size, mean_path):
    n = caffe.NetSpec()
    mean = load_image_mean(mean_path)

    if mean is not None:
        transform_param = dict(mirror=False, crop_size = 48, mean_value = map(int, mean), scale=1.0/255)
    else:
        transform_param = dict(mirror=False, crop_size = 48, scale=1.0/255)

    if phase == "train":
        PHASE = "TRAIN"
    elif phase == "test":
        PHASE = "TEST"

    n.data, n.label = L.Data(
        batch_size = batch_size,
        backend = P.Data.LMDB,
        source = lmdb,
        transform_param=transform_param,
        ntop = 2,
        include = dict(phase = caffe_pb2.Phase.Value(PHASE)),
        name = "data"
    )

    return n



def Data(n, lmdb, phase, batch_size, mean_path):
    mean = load_image_mean(mean_path)
    if mean is not None:
        transform_param = dict(mirror=False, crop_size = 48, mean_value = map(int, mean))
    else:
        transform_param = dict(mirror=False, crop_size = 48)


    if phase == "train":
        PHASE = "TRAIN"
    elif phase == "test":
        PHASE = "TEST"

    d_name = "data_{}".format(net_num)
    l_name = "label_{}".format(net_num)
    n[d_name], n[l_name] = L.Data(
        batch_size = batch_size,
        backend = P.Data.LMDB,
        source = lmdb,
        transform_param=transform_param,
        ntop = 2,
        include = dict(phase = caffe_pb2.Phase.Value(PHASE)),
        name = d_name)

    global silence
    if net_num > 0:
        silence += [n[l_name]]
        # n["silence"+ str(net_num%5)] = L.Silence(n[l_name])

 


def make_net(n, args, num_of_classes = 43):
    activ=args.activation
    n.pool1 = maxpool("pool1", conv1(n, "conv1", n.data, 100, kernel_size = 7, pad = 0, activ=activ))
    n.pool2 = maxpool("pool2", conv1(n, "conv2", n.pool1, 150, kernel_size = 4, pad = 0, activ=activ))
    n.pool3 = maxpool("pool3", conv1(n, "conv3", n.pool2, 250, kernel_size = 4, pad = 0, activ=activ))

    n.fc4_300, n.relu4 = fc("fc4", n.pool3, num_output = 300, activ=activ)
    if args.dropout == True:
        n.drop4 = dropout("drop4", n.relu4, dropout_ratio = args.drop_ratio)

    n.fc5_classes, n.softmax = fc("fc5", n.relu4, num_output = num_of_classes, activ="softmax")


    # n.loss = L.SoftmaxWithLoss(n.fc5_classes, n.label)
    n.loss = L.MultinomialLogisticLoss(n.softmax, n.label)
    n.accuracy_1 = accuracy("accuracy_1", n.fc5_classes, n.label, 1)
    n.accuracy_5 = accuracy("accuracy_5", n.fc5_classes, n.label, 5)



    return n.to_proto()



        


def launch():
    parser = gen_parser()
    args = parser.parse_args()
    exp_num = args.EXPERIMENT_NUMBER
    batch_size = args.batch_size
    proto_pref = args.proto_pref
    snap_pref = args.snap_pref

    data_prefix = "../local_data"

    modes = ["orig", "histeq", "AHE", "imajust", "CoNorm" ]
    # modes=["orig"]
    for dataset in ["rtsd-r1","rtsd-r3"]:
        if dataset == "rtsd-r1":
            num_of_classes = 67
        elif dataset == "rtsd-r3":
            num_of_classes = 106


        for mode in modes:
            directory = '{}/experiment_{}/{}/{}/trial_{}/'.format(proto_pref,exp_num, dataset,mode, args.trial_number)
            safe_mkdir(directory)
            for phase in ['train', 'test']:
                print("Generating architectures")
                print("{}  {}".format(directory, phase))

                mean_path = '{}/lmdb/{}/{}/{}/mean.txt'.format(data_prefix,dataset, mode, phase)
                # mean_path = '{}/{}/{}/{}/mean.txt'.format(data_prefix,dataset, mode, phase)
                with open('{}/{}.prototxt'.format(directory, phase), 'w') as f:
                    f.write(str(make_net(initWithData(
                                            '{}/lmdb/{}/{}/{}/lmdb'.format(data_prefix, 
                                                                    dataset, mode, phase), 
                                            batch_size=batch_size,
                                            phase=phase,
                                            mean_path=mean_path
                                            ),
                                        args=args,
                                        num_of_classes=num_of_classes
                    )))

                print("")
            GenSingleNetSolver(dataset, mode, args)
              


launch()

# with open('Prototxt/{}/{}/test.prototxt'.format(dataset,mode), 'w') as f:
                #     f.write(str(make_net(initWithData('{}/lmdb/{}/{}/test_lmdb'.format(data_prefix,dataset, mode), 128))))
