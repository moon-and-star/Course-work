#!/usr/bin/env bash

source ~/anaconda3/bin/activate
printf "PREPROCESSING DATA\n\n"
./preprocess.py
printf "CREATING LMDB\n\n"
./create_lmdb.sh
source ~/anaconda3/bin/deactivate

#this script imports caffe which doesn't work with python3
