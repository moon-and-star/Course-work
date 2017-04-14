#!/usr/bin/env python

#!/usr/bin/env python

from gen_solver import get_dataset_size

import sys
sys.path.append('/opt/caffe/python/')


import caffe


def test():
	size = get_dataset_size(dataset="rtsd-r1", phase="test", mode="orig")
	net = caffe.Net('./Prototxt/experiment_10/rtsd-r1/orig/trial_1/test.prototxt',1,
	                weights='./snapshots/experiment_10/rtsd-r1/orig/trial_1/snap_iter_2500.caffemodel')
	#net = caffe.Net('./Prototxt/experiment_10/rtsd-r1/AHE/trial_1/test.prototxt',
	 #               './snapshots/experiment_10/rtsd-r1/AHE/trial_1/snap_iter_2500.caffemodel', caffe.TEST)

	sum = 0
	for i in range (size):
		out = net.forward()
		acc =net.blobs["accuracy_1"].data
		print(acc)

	print("average = {}".format(sum / size))
	                 
test()