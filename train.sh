
#!/usr/bin/env bash

# git pull


TOOLS=/opt/caffe/.build_release/tools
EXTRA_TOOLS=/opt/caffe/tools/extra
# TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/.build_release/tools
# EXTRA_TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools/extra
echo " tools = ${TOOLS}"


EXPERIMENT_NUM=6
GPU_NUM=1
BATCH_SZ=1024
EPOCH=100
TEST_FR=1
SNAP_FR=10
STEP_FR=20
GAMMA=0.5
LR=1e-3

CONV_GROUP=1
activation="scaled_tanh"
# activation="relu"

TRY_NUM=5


msg="""
	EXPERIMENT_NUM=$EXPERIMENT_NUM    
	GPU_NUM=$GPU_NUM   
	BATCH_SZ=$BATCH_SZ    
	EPOCH=$EPOCH     
	TEST_FR=$TEST_FR     
	SNAP_FR=$SNAP_FR     
	STEP_FR=$STEP_FR     
	GAMMA=$GAMMA     
	LR=$LR   
	activation=$activation  
	CONV_GROUP=$CONV_GROUP
	TRY_NUM=$TRY_NUM"""

printf "$msg\n"




datasets=("rtsd-r1")
modes=("histeq" "imajust")


# datasets=("rtsd-r1" "rtsd-r3")
# modes=("CoNorm" "orig" "AHE" "histeq" "imajust")




printf "\n\n GENERATING ARCHITECTURES\n\n"
for k in $(seq 1 $TRY_NUM); do
	snap_pref="./snapshots"
	proto_pref="./Prototxt"
	python2 ./net_generator_exp_num.py \
			$EXPERIMENT_NUM -b $BATCH_SZ -e $EPOCH -tf $TEST_FR -sn $SNAP_FR \
			-st $STEP_FR  -lr $LR -g $GAMMA -a $activation \
			-cg $CONV_GROUP -s $snap_pref -p $proto_pref -tn $k
done




printf "\n\n\n Creating log and snapshot folders(if necessary)\n"

mkdir -p logs
mkdir -p snapshots
for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		for k in $(seq 1 $TRY_NUM); do
			printf "dataset = ${i},  mode = ${j}  trial = trial_$k\n"
			#safe directory creating
			mkdir -p ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k
			mkdir -p ./snapshots/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k
		    
		done

	done
done






printf "\n\n\n Training nets \n"
for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		echo "$msg" > ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/params.txt
		cp ./net_generator_exp_num.py ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/
		cp ./train.sh ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/
		
		git add ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/
		git commit -m "script for ${i} ${j}"
		git push




		for k in $(seq 1 $TRY_NUM); do
			printf "Training: dataset = ${i},  mode = ${j} trial=trial_$k \n"

			GLOG_logtostderr=0 $TOOLS/caffe train -gpu ${GPU_NUM}    \
				--solver=./Prototxt/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k/solver.prototxt    \
				2>&1| tee ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k/training_log.txt
				
			GLOG_logtostderr=0 python2 $EXTRA_TOOLS/parse_log.py  --verbose     \
				./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k/training_log.txt    \
				./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k

			python2 ./plot_logs.py ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k     training_log.txt 


			git add ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k
			# git add -f ./Prototxt/experiment_${EXPERIMENT_NUM}/${i}/${j}
			git commit -m "training log for ${i} ${j} trial=trial_$k"
			git push


			
		done
	done
done


# git push



#  #--weights=pretrained.caffemodel
