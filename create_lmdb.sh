#!/usr/bin/env sh
TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/build/tools
# TOOLS=/opt/caffe/build/tools

prefix="../local_data"
datasets=("rtsd-r1" "rtsd-r3")
# $prefix/
preprocessing=("CoNorm" "orig" "AHE" "histeq" "imajust")

for i in "$datasets[@]"
do
	for j in "${preprocessing[@]}"
	do
		# printf "\n\n\n\n\n\n dataset = ${i}, preprocessing_type = ${j}    \n\n\n\n\n\n"
		rm -rf ../local_data/lmdb/${i}/train_${j}_lmdb
		rm -rf ../local_data/lmdb/${i}/test_${j}_lmdb
	done
done

mkdir -p local_data
mkdir -p local_data/lmdb/


# rm -rf lmdb/train_orig_lmdb
# rm -rf lmdb/train_histeq_lmdb
# rm -rf lmdb/train_AHE_lmdb
# rm -rf lmdb/train_CoNorm_lmdb
# rm -rf lmdb/train_imajust_lmdb

# rm -rf lmdb/test_orig_lmdb
# rm -rf lmdb/test_histeq_lmdb
# rm -rf lmdb/test_AHE_lmdb
# rm -rf lmdb/test_CoNorm_lmdb
# rm -rf lmdb/test_imajust_lmdb
# mkdir -p lmdb/

GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/test/orig/   RTSD_cropped/test/orig/gt_test.txt lmdb/test_orig_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/test/AHE/   RTSD_cropped/test/AHE/gt_test.txt lmdb/test_AHE_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/test/CoNorm/   RTSD_cropped/test/CoNorm/gt_test.txt lmdb/test_CoNorm_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/test/imajust/   RTSD_cropped/test/imajust/gt_test.txt lmdb/test_imajust_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/test/histeq/   RTSD_cropped/test/histeq/gt_test.txt lmdb/test_histeq_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/train/orig/   RTSD_cropped/train/orig/gt_train.txt lmdb/train_orig_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/train/AHE/   RTSD_cropped/train/AHE/gt_train.txt lmdb/train_AHE_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/train/CoNorm/   RTSD_cropped/train/CoNorm/gt_train.txt lmdb/train_CoNorm_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/train/imajust/   RTSD_cropped/train/imajust/gt_train.txt lmdb/train_imajust_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_cropped/train/histeq/   RTSD_cropped/train/histeq/gt_train.txt lmdb/train_histeq_lmdb



#GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb RTSD_processed/test/ RTSD_processed/gt_test.txt lmdb/test_lmdb
