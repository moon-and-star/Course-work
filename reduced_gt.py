#!/usr/bin/env python
import argparse

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("th",type=int, 
                        help='threshold for number of signs in one class')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    threshold = init().th

    rootpath = '../local_data/RTSD/'
    modes =  modes = ["orig", "histeq", "AHE", "imajust", "CoNorm" ]
    for mode in modes:
        class_size = [0] * 116
        for phase in ["train", "test"]:
            with open("{}/{}/gt_{}_full.txt".format(rootpath, mode, phase)) as f:
                for line in f.readlines():
                    s = line.split("/")[0]
                    class_size[int(s)] += 1

        print(class_size)
        for i in range(len(class_size)):
            print(i, class_size[i])

        for phase in ["train", "test"]:
            with open("{}/{}/gt_{}_full.txt".format(rootpath, mode, phase)) as f:
                with open("{}/{}/gt_{}.txt".format(rootpath, mode,  phase), 'w') as out:
                    for line in f.readlines():
                        s = line.split("/")[0]
                        if class_size[int(s)] >= threshold:
                            out.write(line)