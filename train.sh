#!/usr/bin/env bash

git pull


TOOLS=/opt/caffe/.build_release/tools
EXTRA_TOOLS=/opt/caffe/tools/extra
# TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/.build_release/tools
# EXTRA_TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools/extra
echo " tools = ${TOOLS}"


EXPERIMENT_NUM=1
GPU_NUM=0
BATCH_SZ=512
EOPOCH=100
TEST_FR=10
SNAP_FR=10
STEP_FR=10
LR=1e-4

printf "\n\n GENERATING ARCHITECTURES\n\n"
./net_generator_exp_num.py -b $BATCH_SZ -e $EOPOCH -tf $TEST_FR -sn $SNAP_FR \
						   -st $STEP_FR $EXPERIMENT_NUM -lr $LR


datasets=("rtsd-r1")
modes=("orig")


# datasets=("rtsd-r1" "rtsd-r3")
# modes=("CoNorm" "orig" "AHE" "histeq" "imajust")






printf "\n Creating log and snapshot folders(if necessary)\n"

mkdir -p logs
mkdir -p snapshots
for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		printf "dataset = ${i},  mode = ${j}\n"
		#safe directory creating
		mkdir -p ./logs/experinent_${EXPERIMENT_NUM}/${i}/${j}
		mkdir -p ./snapshots/experinent_${EXPERIMENT_NUM}/${i}/${j}
	done
done





printf "\n Copying prototxt files\n"

for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		printf "\ndataset = ${i},  mode = ${j} \n"
		cp -v -u ./Prototxt/${i}/${j}/train.prototxt ./logs/experinent_${EXPERIMENT_NUM}/${i}/${j}/
		cp -v -u ./Prototxt/${i}/${j}/test.prototxt ./logs/experinent_${EXPERIMENT_NUM}/${i}/${j}/
		cp -v -u ./Prototxt/${i}/${j}/solver.prototxt ./logs/experinent_${EXPERIMENT_NUM}/${i}/${j}/
	done
done




printf "\n Training nets \n"

for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		printf "\ndataset = ${i},  mode = ${j} \n"
		GLOG_logtostderr=0 $TOOLS/caffe train -gpu ${GPU_NUM}    \
			--solver=./logs/experinent_${EXPERIMENT_NUM}/${i}/${j}/solver.prototxt    \
			2>&1| tee ./logs/experinent_${EXPERIMENT_NUM}/${i}/${j}/training_log.txt
			
		GLOG_logtostderr=0 $EXTRA_TOOLS/parse_log.py  --verbose     \
			./logs/experinent_${EXPERIMENT_NUM}/${i}/${j}/training_log.txt    \
			./logs/experinent_${EXPERIMENT_NUM}/${i}/${j}/

		./plot_logs.py ./logs/experinent_${EXPERIMENT_NUM}/${i}/${j}     training_log.txt 

		git add ./logs
		git commit -m "training log for ${i} ${j}"
	done
done


git push



#  #--weights=pretrained.caffemodel
