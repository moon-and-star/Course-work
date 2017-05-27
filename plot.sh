
#!/usr/bin/env bash


TOOLS=/opt/caffe/.build_release/tools
EXTRA_TOOLS=/opt/caffe/tools/extra

TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/.build_release/tools
EXTRA_TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools/extra
echo " tools = ${TOOLS}"



EXPERIMENT_NUM=10    
TRY_NUM=5






# datasets=("rtsd-r1")
# modes=("histeq" "imajust")


datasets=("rtsd-r1" "rtsd-r3")
modes=("CoNorm" "orig" "AHE" "histeq" "imajust")




printf "PLOTTING"
for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		for k in $(seq 1 $TRY_NUM); do
			python2 ./plot_logs.py ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k     training_log.txt 

			git add ./logs/experiment_${EXPERIMENT_NUM}/${i}/${j}/trial_$k/*.png
			git commit -m "training log plots for ${i} ${j} trial=trial_$k"
			git push
	
		done
	done
done



