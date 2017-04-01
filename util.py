import sys
import argparse
import os.path as osp
from os import makedirs


def load_image_mean(mean_path):
    if osp.exists(mean_path):
        return map(float, open(mean_path, 'r').read().split())
    else:
        print('WARNING: no mean file!')
    return None


def safe_mkdir(directory_name):
    if not osp.exists(directory_name):
        makedirs(directory_name)



def gen_parser():
    description = """
    DESCRIPTION:
    This program generates network architectures and solver
    for particular experiment and stores them into prototxt
    files in special folder (which is specified in current code)
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("EXPERIMENT_NUMBER",type=int, 
                        help='the number of current experiment with nets ')
    parser.add_argument('-b','--batch_size',default=512, type=int, 
                        help='size of batch for training (default=512)')
    parser.add_argument('-e','--epoch',default=100, type=int, 
                        help='number of training epoch (default=100)')

    parser.add_argument('-p','--proto_pref',default="./Prototxt", type=str, 
                        help='Path for saving prototxt files (common prefix for all experiments)')
    parser.add_argument('-s', '--snap_pref',default="./snapshots", type=str, 
                        help='Path for saving snapshot files (common prefix for all experiments)')


    return parser
