#!/usr/bin/env bash


# TOOLS=/opt/caffe/.build_release/tools
# EXTRA_TOOLS=/opt/caffe/tools/extra
TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/.build_release/tools
EXTRA_TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools/extra

# echo $TOOLS
 #--weights=pretrained.caffemodel


EXPERIMENT_NUM=0


datasets=("rtsd-r1" "rtsd-r3")
modes=("CoNorm" "orig" "AHE" "histeq" "imajust")



printf "\n\n\nTraining nets\n\n"

for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		printf "\ndataset = ${i},  mode = ${j} \n"
		
		GLOG_logtostderr=0 $EXTRA_TOOLS/parse_log.py  --verbose     \
			./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/$training_log.txt    \
			./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/

		./plot_logs.py ./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}     training_log.txt
	done
done

