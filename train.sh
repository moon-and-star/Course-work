#!/usr/bin/env bash

# git pull


TOOLS=/opt/caffe/.build_release/tools
EXTRA_TOOLS=/opt/caffe/tools/extra
# TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/.build_release/tools
# EXTRA_TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools/extra
echo " tools = ${TOOLS}"


EXPERIMENT_NUM=4
GPU_NUM=1
BATCH_SZ=1024
EPOCH=100
TEST_FR=1
SNAP_FR=10
STEP_FR=20
GAMMA=0.5
LR=1e-3

printf "\n\n GENERATING ARCHITECTURES\n\n"
python2 ./net_generator_exp_num.py -b $BATCH_SZ -e $EPOCH -tf $TEST_FR -sn $SNAP_FR \
						   -st $STEP_FR $EXPERIMENT_NUM -lr $LR -g $GAMMA




datasets=("rtsd-r1")
modes=("AHE")


# datasets=("rtsd-r1" "rtsd-r3")
# modes=("CoNorm" "orig" "AHE" "histeq" "imajust")






printf "\n\n\n Creating log and snapshot folders(if necessary)\n"

mkdir -p logs
mkdir -p snapshots
for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		printf "dataset = ${i},  mode = ${j}\n"
		#safe directory creating
		mkdir -p ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/
		mkdir -p ./snapshots/experiment_${EXPERIMENT_NUM}/${i}/${j}/
	done
done





printf "\n\n\n Training nets \n"

for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		printf "\n\n\n  dataset = ${i},  mode = ${j} \n"
		GLOG_logtostderr=0 $TOOLS/caffe train -gpu ${GPU_NUM}    \
			--solver=./Prototxt/experiment_${EXPERIMENT_NUM}/${i}/${j}/solver.prototxt    \
			2>&1| tee ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/training_log.txt
			
		GLOG_logtostderr=0 python2 $EXTRA_TOOLS/parse_log.py  --verbose     \
			./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/training_log.txt    \
			./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/

		python2 ./plot_logs.py ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}     training_log.txt 

		git pull
		git add ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}
		git add ./Prototxt/experiment_${EXPERIMENT_NUM}/${i}/${j}
		git commit -m "training log for ${i} ${j}"
		git push
	done
done


# git push



#  #--weights=pretrained.caffemodel
