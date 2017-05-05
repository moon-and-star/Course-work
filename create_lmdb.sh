# !/usr/bin/env bash
# TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/build/tools
TOOLS=/opt/caffe/build/tools

prefix="../local_data"
# datasets=("rtsd-r1" "rtsd-r3")
datasets=("RTSD")
phases=("train" "test")
modes=("CoNorm" "orig" "AHE" "histeq" "imajust")
# modes=("imajust")



printf "removing old files (if any)\n"
for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		for k in "${phases[@]}"
		do
			printf "dataset = ${i},  mode = ${j},  phase = ${k} \n"
			rm -rf ${prefix}/lmdb/${i}/${j}/${k}/lmdb
			rm -rf ${prefix}/lmdb/${i}/${j}/${k}/mean.txt
		done
	done
done




printf "\n\n\ncreating lmdb folders(if necessary)\n\n"
mkdir -p ${prefix}/lmdb
for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		for k in "${phases[@]}"
		do
			printf "dataset = ${i},  mode = ${j},  phase = ${k} \n"
			#safe directory creating
			mkdir -p ${prefix}/lmdb/${i}/${j}/${k}
		done
	done
done



printf "\n\n\nCreating lmdb files\n\n"
mkdir -p ${prefix}/lmdb
for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		for k in "${phases[@]}"
		do
			printf "dataset = ${i},  mode = ${j},  phase = ${k} \n"
			#WARNING: do not shuffle for commitee
			#input data folder    input label file      output lmdb file
			# GLOG_logtostderr=1 $TOOLS/convert_imageset -backend lmdb  \
			GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb  \
			     ${prefix}/${i}/${j}/${k}/          \
			     ${prefix}/${i}/${j}/gt_${k}.txt    \
			     ${prefix}/lmdb/${i}/${j}/${k}/lmdb
		done
	done
done



printf "\n\n\nCopying corresponding mean.txt files \n\n"
mkdir -p ${prefix}/lmdb
for i in "${datasets[@]}"
do
	for j in "${modes[@]}"
	do
		for k in "${phases[@]}"
		do
			printf "dataset = ${i},  mode = ${j},  phase = ${k} \n"
			#input mean.txt file     output folder
			cp ${prefix}/${i}/${j}/${k}/mean.txt ${prefix}/lmdb/${i}/${j}/${k} 
		done
	done
done


