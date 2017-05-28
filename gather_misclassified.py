#!/usr/bin/env python
from util import safe_mkdir
from shutil import copyfile

rate = 10
def gather(inpath, outpath):
	safe_mkdir(outpath)
	with open(inpath) as inp:
		i = 0
		for line in inp.readlines(): 
			if i % rate == 0:
				print(i)

			s = line.split(' ')
			src = s[2]
			real = int(s[3])
			predicted = int(s[8])

			dst = "{}/{}r_{}p_{}name.png".format(outpath, real, predicted, i) 
			copyfile(src, dst)

			i += 1



if __name__ == '__main__':
	exp_num = 21
	inpath = "./logs/experiment_{}/RTSD/misclassified.txt".format(exp_num)
	outpath = "../local_data/RTSD/exp_{}/".format(exp_num)
	gather(inpath, outpath)
