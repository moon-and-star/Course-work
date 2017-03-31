#!/usr/bin/env bash

TOOLS=/opt/caffe/.build_release/tools
EXTRA_TOOLS=/opt/caffe/tools/extra
# TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/.build_release/tools
# EXTRA_TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools/extra
echo " tools = ${TOOLS}"


EXPERIMENT_NUM=0
GPU_NUM=1

datasets=("rtsd-r1")
# modes=("orig")


# datasets=("rtsd-r1" "rtsd-r3")
modes=("CoNorm" "orig" "AHE" "histeq" "imajust")


printf "GENERATING ARCHITECTURES\n\n"
./net_generator_exp_num.py


printf "\n\n\ncreating log and snapshot folders(if necessary)\n\n"

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





printf "\n\n\nCopying prototxt files\n\n"

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



printf "\n\n\nTraining nets\n\n"

for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		printf "\ndataset = ${i},  mode = ${j} \n"
		GLOG_logtostderr=0 $TOOLS/caffe train -gpu ${GPU_NUM}    \
			--solver=./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/solver.prototxt    \
			2>&1| tee ./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/training_log.txt
			
		GLOG_logtostderr=0 $EXTRA_TOOLS/parse_log.py  --verbose     \
			./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/training_log.txt    \
			./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}/

		./plot_logs.py ./logs/${i}/${j}/experinent_${EXPERIMENT_NUM}     training_log.txt 
	done
done



# for i in "${modes[@]}"
# do
# 	printf "\n\n\n\n\n\n           ${i} data is in process   \n\n\n\n\n\n"
#    	mkdir -p logs/$i/${i}_${EXPERIMENT_NUM}
#    	mkdir -p snapshots/$i

#    	cp -v -u ./layers/solver_${i}.prototxt ./logs/$i/${i}_${EXPERIMENT_NUM}/
# 	cp -v -u ./layers/train_${i}.prototxt ./logs/$i/${i}_${EXPERIMENT_NUM}/
# 	cp -v -u ./layers/test_${i}.prototxt ./logs/$i/${i}_${EXPERIMENT_NUM}/

# 	GLOG_logtostderr=0 $TOOLS/caffe train --solver=layers/solver_$i.prototxt  2>&1| tee ./logs/$i/${i}_${EXPERIMENT_NUM}/$i.txt
# 	GLOG_logtostderr=0 $EXTRA_TOOLS/parse_log.py  --verbose  ./logs/$i/${i}_${EXPERIMENT_NUM}/$i.txt ./logs/$i/${i}_${EXPERIMENT_NUM}/
# 	./plot_logs.py $EXPERIMENT_NUM $i
# done



git add logs
git commit -m "last full testing results"
git push



#  #--weights=pretrained.caffemodel
