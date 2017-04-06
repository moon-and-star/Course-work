#!/usr/bin/env python

import sys
sys.path.append('/opt/caffe/python/')

from util import safe_mkdir, gen_parser, load_image_mean
from solver_params import gen_solver

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
                     scale_param=dict(filler=dict(value=0.6666))
                     )
    tanh =  L.TanH(scale1, in_place = True, name = "{}_sTanH".format(name))
    scale2 = L.Scale(tanh, in_place = True, name = "{}_postscale".format(name),
                     param=dict(lr_mult=0, decay_mult=0),
                     scale_param=dict(filler=dict(value=1.7159))
                     )
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
    print(name)
    return L.Pooling(bottom, kernel_size = kernel_size, stride = stride, pool = P.Pooling.MAX, name = name)

def avepool(name, bottom, kernel_size = 2, stride = 2):
    return L.Pooling(bottom, kernel_size = kernel_size, stride = stride, pool = P.Pooling.AVE, name = name)

# def fc_relu(name, bottom, num_output):
#     fc = L.InnerProduct(
#         bottom,
#         num_output = num_output,
#         weight_filler = dict(type = 'xavier'),
#         name = "{}_{}".format(name, num_output)
#     )
#     return fc, L.ReLU(fc, in_place = True, name = "{}_relu".format(name))


# def fc_stanh(name, bottom, num_output):
#     # fc = L.OLOLO(
#     fc = L.InnerProduct(
#         bottom,
#         num_output = num_output,
#         weight_filler = dict(type = 'xavier'),
#         name = "{}_{}".format(name, num_output)
#     )

#     scale1 = L.Scale(fc, in_place = True, name = "{}_prescale".format(name),
#                      param=dict(lr_mult=0, decay_mult=0),
#                      scale_param=dict(filler=dict(value=0.6666))
#                      )
#     tanh =  L.TanH(scale1, in_place = True, name = "{}_sTanH".format(name))
#     scale2 = L.Scale(tanh, in_place = True, name = "{}_postscale".format(name),
#                      param=dict(lr_mult=0, decay_mult=0),
#                      scale_param=dict(filler=dict(value=1.7159))
#                      )

#     return fc, scale2
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
                         scale_param=dict(filler=dict(value=0.6666))
                         )
        tanh =  L.TanH(scale1, in_place = True, name = "{}_sTanH".format(name))
        scale2 = L.Scale(tanh, in_place = True, name = "{}_postscale".format(name),
                         param=dict(lr_mult=0, decay_mult=0),
                         scale_param=dict(filler=dict(value=1.7159))
                         )

        return fc, scale2


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


def initWithData(lmdb, phase, batch_size, mean_path):
    n = caffe.NetSpec()
    mean = load_image_mean(mean_path)

    if mean is not None:
        transform_param = dict(mirror=False, crop_size = 48, mean_value = map(int, mean))
    else:
        transform_param = dict(mirror=False, crop_size = 48)

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

def ConvPoolAct(n, net_num, activ):
    cbott = [n.data]
    out_num = [100, 150, 250]
    ker_sz = [7, 4, 4]

    for j in range(3):
        p_name = "pool{}_{}".format(net_num, j)
        c_name="conv{}_{}".format(net_num, j)
        conv = conv1(n=n, name=c_name, 
                     bottom=cbott[j], num_output=out_num[j], 
                     kernel_size = ker_sz[j], 
                     pad = 0, activ=activ)
        pool = maxpool(name=p_name, bottom=conv)
        n[p_name]=pool
        cbott += [pool]

def FcDropAct(n, net_num, classes, activ):
    act_name = "{}__{}".format(activ, net_num)

    fc_name = "fc{}_4".format(net_num)
    bott_name = "pool{}_2".format(net_num)
    n[fc_name], n[act_name] = fc(name=fc_name, bottom=n[bott_name], num_output=300, activ=activ)
    bott_name = fc_name

    d_name = "drop{}_4".format(net_num)
    n[d_name] = dropout(d_name, n[bott_name], dropout_ratio=0.4)
    bott_name = d_name

    fc_name = "fc{}_5".format(net_num)
    n[fc_name], n[act_name] = fc(name=fc_name, bottom=n[bott_name], num_output=classes, activ=activ)

    # n.fc4_300, n.relu4 = fc_relu("fc4_{}".format(i), n.pool3, num_output = 300)
    # n.drop4 = dropout("drop4_{}".format(i), n.relu4, dropout_ratio=0.4)
    # n.fc5_classes, relu5 = fc_relu("fc5_{}".format(i), n.relu4, num_output = num_of_classes)
 


def make_net(n, num_of_classes = 43, activ="relu"):
    for i in range(1):
        ConvPoolAct(n=n, net_num=i , activ=activ)
        FcDropAct(n=n, net_num=i, classes=num_of_classes, activ=activ)
       
        # if activ=="relu":
        # #     n.fc4_300, n.relu4 = fc_relu("fc4_{}".format(i), n.pool3, num_output = 300)
        # #     n.drop4 = dropout("drop4_{}".format(i), n.relu4, dropout_ratio=0.4)
        # #     n.fc5_classes, relu5 = fc_relu("fc5_{}".format(i), n.relu4, num_output = num_of_classes)
      
        # elif activ == "scaled_tanh":
        #     n.fc4_300, n.stanh4 = fc_stanh("fc4_{}".format(i), d["pool1_{}".format(i)], num_output = 300)
        #     n.drop4 = dropout("drop4_{}".format(i), n.stanh4, dropout_ratio=0.4)
        #     n.fc5_classes, stanh5 = fc_stanh("fc5_{}".format(i), n.stanh4, num_output = num_of_classes)





    #     n.softmax=L.Softmax(n.fc5_classes)
    
    # n.__dict__.update(d)

    # n.loss = L.MultinomialLogisticLoss(n.softmax, n.label)
    # # n.loss = L.SoftmaxWithLoss(n.fc5_classes, n.label)
    # n.accuracy_1 = accuracy("accuracy_1", n.fc5_classes, n.label, 1)
    # n.accuracy_5 = accuracy("accuracy_5", n.fc5_classes, n.label, 5)

    return n.to_proto()



        


def launch():
    parser = gen_parser()
    args = parser.parse_args()
    exp_num = args.EXPERIMENT_NUMBER
    batch_size = args.batch_size
    proto_pref = args.proto_pref
    snap_pref = args.snap_pref

    data_prefix = "../local_data"

    modes = ["orig"]
    for dataset in ["rtsd-r1"]:
    # modes = ["orig", "histeq", "AHE", "imajust", "CoNorm" ]
    # for dataset in ["rtsd-r1","rtsd-r3"]:
        if dataset == "rtsd-r1":
            num_of_classes = 67
        elif dataset == "rtsd-r3":
            num_of_classes = 106


        for mode in modes:
            directory = '{}/experiment_{}/{}/{}/'.format(proto_pref,exp_num, dataset,mode)
            safe_mkdir(directory)
            for phase in ['train', 'test']:
                print("Generating architectures")
                print("{}  {}".format(directory, phase))

                mean_path = '{}/lmdb/{}/{}/{}/mean.txt'.format(data_prefix,dataset, mode, phase)
                # mean_path = '{}/{}/{}/{}/mean.txt'.format(data_prefix,dataset, mode, phase)
                with open('{}/{}.prototxt'.format(directory, phase), 'w') as f:
                    content = str(make_net(initWithData(
                                            '{}/lmdb/{}/{}/{}/lmdb'.format(data_prefix, 
                                                                    dataset, mode, phase), 
                                            batch_size=batch_size,
                                            phase=phase,
                                            mean_path=mean_path
                                            ),
                                        num_of_classes=num_of_classes,
                                        activ=args.activation
                    ))
                    f.write(content)
                    if phase=="train":
                        print(content)

                print("")
            gen_solver(dataset, mode, args)
              


launch()
