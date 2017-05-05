#!/usr/bin/env python
import json
from pprint import pprint
from preprocess import crop
import cv2
import matplotlib.pyplot as plt
from util import safe_mkdir, load_marking



def crop_sign(image_name, params):
	img = cv2.imread(image_name)
	x1, y1 = params["x"], params["y"]
	x2, y2 = x1 + params["w"], y1 + params["h"]
	print ("Coordinates: ",x1, y1, x2, y2)

	return crop(img, x1, y1, x2, y2, expand=3)



def read_classes(path):
	classes = set()
	with open(path) as f:
		lines = f.readlines()[1:]
		for line in lines:
			s = line.split(",")
			classes.add(s[1].strip())

	return classes



def get_label_list(marking):
	labels = set()
	last = set()
	for name in sorted(marking):
		for sign_entry in marking[name]:
			class_name = sign_entry["sign_class"]
			if "unknown" in class_name:
				last.add(class_name)
			else:
				labels.add(class_name)
	# print(labels)			
	return sorted(labels) + sorted(last)



def get_label_set(marking):
	return set(get_label_list(marking))



def CropSortSave(marking, rootpath, phase):
	for name in sorted(marking):
		for sign_entry in marking[name]:
			path = "{}/imgs/{}".format(rootpath, name)
			cropped = crop_sign(path, sign_entry)
			if "unknown" in sign_entry['sign_class'] :
				save_path = "{}/cropped/{}/unknown/{}/".format(rootpath, phase, sign_entry['sign_class'])
			else:
				save_path = "{}/cropped/{}/{}".format(rootpath, phase, sign_entry['sign_class'])
				
			safe_mkdir(save_path)
			im_path = "{}/{}".format(save_path, name)
			a = cv2.imwrite(im_path, cropped)	



def DictForClasses(classes):
	d = {}
	for name in sorted(classes):
		if not "unknown" in name:
			d[name]  = []
	return d




#swaps train/test for classes in which train < test
def  normalize(markings, classes):
	for label in classes:
		train_size = len(markings["train"][label])
		test_size = len(markings["test"][label])
		# print("{:10} {:5} {:5}".format(label, train_size, test_size))
		if train_size < test_size:
			markings["train"][label], markings["test"][label] = markings["test"][label], markings["train"][label]
	return markings




#here marking is dict of lists of tuples: label: [(filename, entry_for_marking)]
def to_std_marking(marking):
	filenames = set()
	for label in sorted(marking):
		for entry in marking[label]:
			filenames.add(entry[0]) #adding entry for filename


	new = {}
	i = 0
	for name in sorted(filenames):
		new[name] = []
	for label in sorted(marking):
		for entry in marking[label]:
			new[entry[0]] += [entry[1]]
			i += 1


	print("imagees amount in set = ", i)
	return new





def Classificationmarking(marking, classes):
	# new = marking
	new = {}
	for phase in ['train', 'test']:
		cdict = DictForClasses(classes)
		m = marking[phase]
		for name in sorted(m):
			for sign_entry in m[name]:
				label =  sign_entry['sign_class']
				if not ("unknown" in label) and (label in classes):
					cdict[label] += [(name, sign_entry)]
		new[phase] = cdict

	new = normalize(new, classes)
	new['train'] = to_std_marking(new['train'])
	new['test'] = to_std_marking(new['test'])

	return new


		 
def getClassificationLabels(classes):
	common = classes['train'].intersection(classes['test'])
	tmp = set()
	for label in common:
		if not "unknown" in label:
			tmp.add(label)

	return tmp



def save_marking(marking):
	for phase in ["train", "test"]:
		filename = "{}/classification_marking_{}.json".format(rootpath, phase)
		with open(filename, 'w') as f:
			content = json.dumps(marking[phase], indent=2, sort_keys=True)
			f.write(content)



def getClassificationMarking():
	classes = {}
	marking = {}
	for phase in ["train", "test"]:
		filename = "{}/marking_{}.json".format(rootpath, phase)
		m = load_marking(filename)
		marking[phase] = m
		classes[phase] = get_label_set(m)

	classes = getClassificationLabels(classes)
	marking = Classificationmarking(marking, classes)
	return marking
	# for key in classes:
	# 	print(key, classes.index(key))

	


			



if __name__ == '__main__':
	rootpath = '../global_data/Traffic_signs/RTSD'
	marking = getClassificationMarking()
	save_marking(marking)
	# launch()
	