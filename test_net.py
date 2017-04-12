#!/usr/bin/env python

#!/usr/bin/env python

import sys
sys.path.append('/opt/caffe/python/')


import caffe


net = caffe.Net('./Prototxt/experiment_9/rtsd-r1/orig/trial_1/test.prototxt',
                './snapshots/experiment_9/rtsd-r1/orig/trial_1/snap_iter_2500.caffemodel', caffe.TEST)
out = net.forward()
print (net.blobs)
#print(net.blobs["softmax"].data)
print(net.blobs["accuracy_1"].data)

#print(net.blobs["fc5_classes_fc5_67_0_split_0"].data)
#print(net.blobs["fc5_classes_fc5_67_0_split_1"].data)
#print(net.blobs["fc5_classes_fc5_67_0_split_2"].data)
#print(net.blobs["loss"].data)
# print(net.predict())
                     
