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




def ParseParams(param_path):
    args = {}
    with open(param_path) as f:
        for line in f:
            if line != "\n":
                line = line.strip()
                s = line.split("=")
                if s[0] in ['GAMMA','LR', 'drop_ratio'] :
                    args[s[0]] = float(s[1])
                elif s[0] in ['activation', 'drop']:
                    args[s[0]] = s[1]
                else:
                    args[s[0]] = int(s[1])
                    
    return args



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

    parser.add_argument('-lr','--learning_rate',default=1e-3, type=float, 
                        help='size of batch for training (default=1e-3)')

    parser.add_argument('-e','--epoch',default=100, type=int, 
                        help='number of training epoch (default=100)')

    parser.add_argument('-tf','--test_frequency',default=1, type=int, 
                        help='test frequensy in epochs (default=1)')

    parser.add_argument('-sn','--snap_epoch',default=10, type=int, 
                        help='snap frequensy in epochs (default=10)')

    parser.add_argument('-st','--step_epoch',default=10, type=int, 
                        help='learning rate decrease frequensy in epochs (default=10)')

    parser.add_argument('-g','--gamma',default=0.5, type=float, 
                        help='learning rate decrease factor (default=0.5)')

    parser.add_argument('-a','--activation',default="scaled_tanh", type=str, 
                        help='activation function type (default = \'scaled_tanh\' )')

    parser.add_argument('-cg','--conv_group',default=1, type=int, 
                        help='how many groups of filters within conv layers (default = 1 )')
    parser.add_argument('-tn','--trial_number',default=1, type=int, 
                        help='number of trial for 1 net (default = 5 )')

    parser.add_argument('-dr','--drop_ratio',default=0.5, type=float, 
                        help='set dropout parameter (default = 0.5 )')
    parser.add_argument('-d','--dropout',action="store_true", 
                        help='enable dropout')


    parser.add_argument('-p','--proto_pref',default="./Prototxt", type=str, 
                        help='Path for saving prototxt files (common prefix for all experiments)')
    parser.add_argument('-s', '--snap_pref',default="./snapshots", type=str, 
                        help='Path for saving snapshot files (common prefix for all experiments)')


    return parser

