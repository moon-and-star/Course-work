#!/usr/bin/env bash
TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/.build_release/tools
EXTRA_TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools/extra
echo " tools = ${TOOLS}"


EXPERIMENT_NUM=4
names=("histeq"  "imajust")
# names=("CoNorm" "orig" "AHE" "histeq" "imajust")




mkdir -p logs
mkdir -p snapshots
for i in "${names[@]}"
do
	printf "\n\n\n\n\n\n           ${i} data is in process   \n\n\n\n\n\n"
   	mkdir -p logs/$i/${i}_${EXPERIMENT_NUM}
   	mkdir -p snapshots/$i

   	cp -v -u ./layers/solver_${i}.prototxt ./logs/$i/${i}_${EXPERIMENT_NUM}/
	cp -v -u ./layers/train_${i}.prototxt ./logs/$i/${i}_${EXPERIMENT_NUM}/
	cp -v -u ./layers/test_${i}.prototxt ./logs/$i/${i}_${EXPERIMENT_NUM}/

	GLOG_logtostderr=0 $TOOLS/caffe train --solver=layers/solver_$i.prototxt  2>&1| tee ./logs/$i/${i}_${EXPERIMENT_NUM}/$i.txt
	GLOG_logtostderr=0 $EXTRA_TOOLS/parse_log.py  --verbose  ./logs/$i/${i}_${EXPERIMENT_NUM}/$i.txt ./logs/$i/${i}_${EXPERIMENT_NUM}/
	./plot_logs.py $EXPERIMENT_NUM $i
done




#  #--weights=pretrained.caffemodel
