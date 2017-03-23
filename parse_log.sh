#!/usr/bin/env sh


mkdir -p logs

TOOLS=/home/katydagoth/Downloads/caffe3/caffe-master/tools
echo $TOOLS
 #--weights=pretrained.caffemodel


GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/AHE.txt ./logs
GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/imajust.txt ./logs
GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/histeq.txt ./logs
GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/CoNorm.txt ./logs
GLOG_logtostderr=0 $TOOLS/extra/parse_log.py  --verbose  ./logs/orig.txt ./logs


