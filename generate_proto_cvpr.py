#!/usr/bin/env python

import sys
sys.path.append('/opt/caffe/python/')

import os.path as osp

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

def conv1(n, name, bottom, num_output, kernel_size = 3, pad = None):
    if pad is None: pad = kernel_size / 2
    conv1, relu1 = conv_relu(n, "{}".format(name), bottom, kernel_size, num_output, pad = pad)
    return conv1, relu1

def conv2(n, name, bottom, num_output, kernel_size = 3, pad = None):
    if pad is None: pad = kernel_size / 2
    conv1, relu1 = conv_relu(n, "{}1".format(name), bottom, kernel_size, num_output, pad = pad)
    conv2, relu2 = conv_relu(n, "{}2".format(name), relu1, kernel_size, num_output, pad = pad)
    return conv2, relu2

def conv3(n, name, bottom, num_output, kernel_size = 3, pad = None):
    if pad is None: pad = kernel_size / 2
    conv1, relu1 = conv_relu(n, "{}1".format(name), bottom, kernel_size, num_output, pad = pad)
    conv2, relu2 = conv_relu(n, "{}2".format(name), relu1, kernel_size, num_output, pad = pad)
    conv3, relu3 = conv_relu(n, "{}3".format(name), relu2, kernel_size, num_output, pad = pad)
    return conv3, relu3

def maxpool(name, bottom, kernel_size = 2, stride = 2):
    return L.Pooling(bottom, kernel_size = kernel_size, stride = stride, pool = P.Pooling.MAX, name = name)

def avepool(name, bottom, kernel_size = 2, stride = 2):
    return L.Pooling(bottom, kernel_size = kernel_size, stride = stride, pool = P.Pooling.AVE, name = name)

def fc_relu(name, bottom, num_output):
    fc = L.InnerProduct(
        bottom,
        num_output = num_output,
        weight_filler = dict(type = 'xavier'),
        name = "{}_{}".format(name, num_output)
    )
    return fc, L.ReLU(fc, in_place = True, name = "{}_relu".format(name))

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
        include = dict(phase = caffe_pb2.Phase.Value("TEST")),
        name = name
    )

def load_image_mean():
    if osp.exists('gtsrb_processed/mean.txt'):
        return map(float, open('gtsrb_processed/mean.txt', 'r').read().split())
    return None

def train_data(lmdb, batch_size):
    n = caffe.NetSpec()

    mean = load_image_mean()

    if mean is not None:
        transform_param = dict(mirror=False, crop_size = 48, mean_value = map(int, mean))
    else:
        transform_param = dict(mirror=False, crop_size = 48)

    n.data, n.label = L.Data(
        batch_size = batch_size,
        backend = P.Data.LMDB,
        source = lmdb,
        transform_param=transform_param,
        ntop = 2,
        include = dict(phase = caffe_pb2.Phase.Value("TRAIN")),
        name = "data"
    )

    return n

def test_data(lmdb, batch_size):
    n = caffe.NetSpec()

    mean = load_image_mean()

    if mean is not None:
        transform_param = dict(mirror=False, crop_size = 48, mean_value = map(int, mean))
    else:
        transform_param = dict(mirror=False, crop_size = 48)

    n.data, n.label = L.Data(
        batch_size = batch_size,
        backend = P.Data.LMDB,
        source = lmdb,
        transform_param=transform_param,
        ntop = 2,
        include = dict(phase = caffe_pb2.Phase.Value("TEST")),
        name = "data"
    )
    return n

def make_net(n):
    n.pool1 = maxpool("pool1", conv1(n, "conv1", n.data, 100, kernel_size = 7, pad = 0)[1])
    n.pool2 = maxpool("pool2", conv1(n, "conv2", n.pool1, 150, kernel_size = 4, pad = 0)[1])
    n.pool3 = maxpool("pool3", conv1(n, "conv3", n.pool2, 250, kernel_size = 4, pad = 0)[1])

    n.fc4_300, n.relu4 = fc_relu("fc4", n.pool3, num_output = 300)
    #n.drop4 = dropout("drop4", n.relu4, dropout_ratio = 0.4)
    n.fc5_43, relu5 = fc_relu("fc5", n.relu4, num_output = 43)

    n.loss = L.SoftmaxWithLoss(n.fc5_43, n.label)
    n.accuracy_1 = accuracy("accuracy_1", n.fc5_43, n.label, 1)
    n.accuracy_5 = accuracy("accuracy_5", n.fc5_43, n.label, 5)

    return n.to_proto()

with open('layers/train.prototxt', 'w') as f:
    f.write(str(make_net(train_data('lmdb/train_lmdb', 128))))

with open('layers/test.prototxt', 'w') as f:
    f.write(str(make_net(test_data('lmdb/test_lmdb', 128))))
