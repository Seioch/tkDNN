#!/bin/bash
# $1 is path to cfg file
# $2 is path to weights file
# $3 is path to feature_list.names

if test -z "$1"
then
        echo "Usage: bash autoconvert_darknet.sh <path to cfg> <path to weights> <path to feature_list.names>" 
        exit
fi

if test -z "$2"
then
        echo "Usage: bash autoconvert_darknet.sh <path to cfg> <path to weights> <path to feature_list.names>"
        exit
fi

if test -z "$3"
then
        echo "Usage: bash autoconvert_darknet.sh <path to cfg> <path to weights> <path to feature_list.names>"
        exit
fi

CFG=$(realpath $1)
WEIGHTS=$(realpath $2)
NAMES=$(realpath $3)

mkdir /export_darknet/darknet/layers
mkdir /export_darknet/darknet/debug
cd /export_darknet/darknet
./darknet export $CFG $WEIGHTS layers
mkdir /tkdnn/tkDNN_seioch/tkDNN_predict/yolov4
mv /export_darknet/darknet/layers /tkdnn/tkDNN_seioch/tkDNN_predict/yolov4
mv /export_darknet/darknet/debug /tkdnn/tkDNN_seioch/tkDNN_predict/yolov4
cd /tkdnn/tkDNN_seioch/tkDNN_predict/
/tkdnn/tkDNN_seioch/tkDNN_predict/build/darknet_to_tensorrt /tkdnn/tkDNN_seioch/tkDNN_predict/yolov4 $CFG $NAMES