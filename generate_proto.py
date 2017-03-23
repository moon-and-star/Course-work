#!/usr/bin/env python

import sys
sys.path.append('/opt/caffe/python/')

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

def conv1(n, name, bottom, num_output, kernel_size = 3):
    conv1, relu1 = conv_relu(n, "{}1".format(name), bottom, kernel_size, num_output, pad = kernel_size / 2)
    return conv1, relu1

def conv2(n, name, bottom, num_output, kernel_size = 3):
    conv1, relu1 = conv_relu(n, "{}1".format(name), bottom, kernel_size, num_output, pad = kernel_size / 2)
    conv2, relu2 = conv_relu(n, "{}2".format(name), relu1, kernel_size, num_output, pad = kernel_size / 2)
    return conv2, relu2

def conv3(n, name, bottom, num_output, kernel_size = 3):
    conv1, relu1 = conv_relu(n, "{}1".format(name), bottom, kernel_size, num_output, pad = kernel_size / 2)
    conv2, relu2 = conv_relu(n, "{}2".format(name), relu1, kernel_size, num_output, pad = kernel_size / 2)
    conv3, relu3 = conv_relu(n, "{}3".format(name), relu2, kernel_size, num_output, pad = kernel_size / 2)
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

def train_data(lmdb, batch_size):
    n = caffe.NetSpec()

    n.data, n.label = L.Data(
        batch_size = batch_size,
        backend = P.Data.LMDB,
        source = lmdb,
        transform_param=dict(mirror=False, crop_size = 48),
        ntop = 2,
        include = dict(phase = caffe_pb2.Phase.Value("TRAIN")),
        name = "data"
    )

    return n

def test_data(lmdb, batch_size):
    n = caffe.NetSpec()

    n.data, n.label = L.Data(
        batch_size = batch_size,
        backend = P.Data.LMDB,
        source = lmdb,
        ntop = 2,
        include = dict(phase = caffe_pb2.Phase.Value("TEST")),
        name = "data"
    )
    return n

def make_net(n):
    n.pool1 = maxpool("pool1", conv2(n, "conv1", n.data, 32)[1])
    n.pool2 = maxpool("pool2", conv3(n, "conv2", n.pool1, 64)[1])
    n.pool3 = maxpool("pool3", conv3(n, "conv3", n.pool2, 128)[1])
    n.pool4 = maxpool("pool4", conv3(n, "conv4", n.pool3, 256)[1])

    n.fc5_128, n.relu5 = fc_relu("fc5", n.pool4, num_output = 128)
    n.drop5 = dropout("drop5", n.relu5, dropout_ratio = 0.4)
    n.fc6_43, relu6 = fc_relu("fc6", n.drop5, num_output = 43)

    n.loss = L.SoftmaxWithLoss(n.fc6_43, n.label)
    n.accuracy_1 = accuracy("accuracy_1", n.fc6_43, n.label, 1)
    n.accuracy_5 = accuracy("accuracy_5", n.fc6_43, n.label, 5)

    return n.to_proto()

with open('layers/train.prototxt', 'w') as f:
    f.write(str(make_net(train_data('lmdb/train_lmdb', 1024))))

with open('layers/test.prototxt', 'w') as f:
    f.write(str(make_net(test_data('lmdb/test_lmdb', 128))))
