#!/usr/bin/env bash

# source ~/anaconda3/bin/activate
printf "PREPROCESSING DATA\n\n"
./preprocess.py
# source ~/anaconda3/bin/deactivate

printf "CREATING LMDB\n\n"
./create_lmdb.sh


#this script imports caffe which doesn't work with python3