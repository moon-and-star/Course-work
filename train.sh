#!/usr/bin/env bash

# git pull


TOOLS=/opt/caffe/.build_release/tools
EXTRA_TOOLS=/opt/caffe/tools/extra
# TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/.build_release/tools
# EXTRA_TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools/extra
echo " tools = ${TOOLS}"


EXPERIMENT_NUM=3
GPU_NUM=0
BATCH_SZ=512
EOPOCH=100
TEST_FR=10
SNAP_FR=10
STEP_FR=10

printf "\n\n GENERATING ARCHITECTURES\n\n"
./net_generator_exp_num.py -b $BATCH_SZ -e $EOPOCH -tf $TEST_FR -sn $SNAP_FR -st $STEP_FR $EXPERIMENT_NUM


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
		mkdir -p ./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}
		mkdir -p ./snapshots/${i}/${j}/experinent_${EXPERIMENT_NUM}
	done
done





printf "\n Copying prototxt files\n"

for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		printf "\ndataset = ${i},  mode = ${j} \n"
		cp -v -u ./Prototxt/${i}/${j}/train.prototxt ./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/
		cp -v -u ./Prototxt/${i}/${j}/test.prototxt ./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/
		cp -v -u ./Prototxt/${i}/${j}/solver.prototxt ./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/
	done
done




# printf "\n Training nets \n"

# for i in "${datasets[@]}"
# do
# 	for j in "${modes[@]}"
# 	do
# 		printf "\ndataset = ${i},  mode = ${j} \n"
# 		GLOG_logtostderr=0 $TOOLS/caffe train -gpu ${GPU_NUM}    \
# 			--solver=./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/solver.prototxt    \
# 			2>&1| tee ./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/training_log.txt
			
# 		GLOG_logtostderr=0 $EXTRA_TOOLS/parse_log.py  --verbose     \
# 			./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/training_log.txt    \
# 			./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/

# 		./plot_logs.py ./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}     training_log.txt 

# 		git add ./logs
# 		git commit -m "training log for ${i} ${j}"
# 	done
# done


# git push



#  #--weights=pretrained.caffemodel
