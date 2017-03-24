#!/usr/bin/env sh


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
	done
done

# GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/AHE.txt ./logs
# GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/imajust.txt ./logs
# GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/histeq.txt ./logs
# GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/CoNorm.txt ./logs
# GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/orig.txt ./logs


