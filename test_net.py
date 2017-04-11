#!/usr/bin/env python

#!/usr/bin/env python

import sys
sys.path.append('/opt/caffe/python/')


import caffe


net = caffe.Net('./Prototxt/experiment_6/rtsd-r1/imajust/trial_1/test.prototxt',
                './snapshots/experiment_6/rtsd-r1/imajust/trial_1/snap_iter_2250.caffemodel', caffe.TEST)
out = net.forward()
print(net.blobs["softmax"].data)
#print(net.blobs["loss"].data)
# print(net.predict())
                     