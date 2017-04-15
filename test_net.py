#!/usr/bin/env python

#!/usr/bin/env python

from gen_solver import get_dataset_size

import sys
sys.path.append('/opt/caffe/python/')


import caffe


# def test():
# 	size = get_dataset_size(dataset="rtsd-r1", phase="test", mode="orig")
# 	net = caffe.Net('./Prototxt/experiment_10/rtsd-r1/orig/trial_1/test.prototxt',1,
# 	                weights='./snapshots/experiment_10/rtsd-r1/orig/trial_1/snap_iter_2500.caffemodel')
# 	#net = caffe.Net('./Prototxt/experiment_10/rtsd-r1/AHE/trial_1/test.prototxt',
# 	 #               './snapshots/experiment_10/rtsd-r1/AHE/trial_1/snap_iter_2500.caffemodel', caffe.TEST)

# 	sum = 0
# 	for i in range (size):
# 		if i % 100 == 0:
# 			print("image in proccess: {}".format(i))
# 		out = net.forward()
# 		acc =net.blobs["accuracy_1"].data
# 		# print(acc)
# 		sum += acc

# 	print("average = {}".format(sum / size))


def load_net(exp_num, dataset, mode, trial, phase):
	model = './Prototxt/experiment_{}/{}/{}/trial_{}/{}.prototxt'.format(exp_num, dataset, mode, trial, phase)
	weights = './snapshots/experiment_{}/{}/{}/trial_{}/snap_iter_2500.caffemodel'.format(exp_num, dataset, mode, trial)
	net = caffe.Net(model,1, weights=weights)

	with open(model) as f:
		for line in f:
			print(line)
			if "batch_size" in line:
				s = line.split(":")
				print int(s[1])
				break
	return net


def test():
	exp_num = 10
	dataset = "rtsd-r1"
	phase = "test"
	mode = "orig"
	trial = 1
	size = get_dataset_size(dataset, phase, mode)
	net = load_net(exp_num, dataset, mode, trial, phase)


	pass
	sum = 0
	for i in range (size):
		if i % 100 == 0:
			print("image in proccess: {}".format(i))
		out = net.forward()
		acc =net.blobs["accuracy_1"].data
		# print(acc)
		sum += acc

	print("average = {}".format(sum / size))


	                 
test()