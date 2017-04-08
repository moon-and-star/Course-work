#!/usr/bin/env bash

# git pull


TOOLS=/opt/caffe/.build_release/tools
EXTRA_TOOLS=/opt/caffe/tools/extra
# TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/.build_release/tools
# EXTRA_TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools/extra
echo " tools = ${TOOLS}"


EXPERIMENT_NUM=6
GPU_NUM=0
BATCH_SZ=150
EPOCH=50
TEST_FR=10
SNAP_FR=10
STEP_FR=20
GAMMA=0.5
LR=1e-2
activation="scaled_tanh"
# activation="relu"


printf "\n\n GENERATING ARCHITECTURES\n\n"
python2 ./gen_commitee.py -b $BATCH_SZ -e $EPOCH -tf $TEST_FR -sn $SNAP_FR \
						   -st $STEP_FR $EXPERIMENT_NUM -lr $LR -g $GAMMA -a $activation




datasets=("rtsd-r1")

# datasets=("rtsd-r1" "rtsd-r3")






printf "\n\n\n Creating log and snapshot folders(if necessary)\n"

mkdir -p logs
mkdir -p snapshots
for i in "${datasets[@]}"
do	
	printf "dataset = ${i}\n"
	#safe directory creating
	mkdir -p ./logs/experiment_${EXPERIMENT_NUM}/${i}/commitee
	mkdir -p ./snapshots/experiment_${EXPERIMENT_NUM}/${i}/commitee
done





printf "\n\n\n Training nets \n"

for i in "${datasets[@]}"
do
	printf "\n\n\n  dataset = ${i} \n"
	GLOG_logtostderr=0 $TOOLS/caffe train -gpu ${GPU_NUM}    \
		--solver=./Prototxt/experiment_${EXPERIMENT_NUM}/${i}/commitee/solver.prototxt    \
		2>&1| tee ./logs/experiment_${EXPERIMENT_NUM}/${i}/commitee/training_log.txt
		
	GLOG_logtostderr=0 python2 $EXTRA_TOOLS/parse_log.py  --verbose     \
		./logs/experiment_${EXPERIMENT_NUM}/${i}/commitee/training_log.txt    \
		./logs/experiment_${EXPERIMENT_NUM}/${i}/commitee/

	python2 ./plot_logs.py ./logs/experiment_${EXPERIMENT_NUM}/${i}/commitee     training_log.txt 

	git pull
	git add ./logs/experiment_${EXPERIMENT_NUM}/${i}/commitee
	git add -f ./Prototxt/experiment_${EXPERIMENT_NUM}/${i}/commitee
	git commit -m "training log for ${i}"
	git push
done


# git push



#  #--weights=pretrained.caffemodel
