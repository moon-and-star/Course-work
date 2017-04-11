#!/usr/bin/env bash

EXPERIMENT_NUM=6    
GPU_NUM=1   
BATCH_SZ=1024    
EPOCH=100     
TEST_FR=1     
SNAP_FR=10     
STEP_FR=20     
GAMMA=0.5     
LR=1e-3   
activation=relu  
CONV_GROUP=1
TRY_NUM=5



snap_pref="./snapshots"
proto_pref="./Prototxt"
python2 ./net_generator_exp_num.py \
		$EXPERIMENT_NUM -b $BATCH_SZ -e $EPOCH -tf $TEST_FR -sn $SNAP_FR \
		-st $STEP_FR  -lr $LR -g $GAMMA -a $activation \
		-cg $CONV_GROUP -s $snap_pref -p $proto_pref -tn $k