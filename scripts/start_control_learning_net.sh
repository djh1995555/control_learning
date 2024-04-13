#!/bin/bash
pwd=$(pwd -P)
lwd=$(dirname $pwd)

task=${1:-dynamic_model}
mode=${2:-train}
config_name=train_weight_estimation_config.yaml
# data_root_dir=${lwd}/data
data_root_dir=/mnt/intel/jupyterhub/jianhao.dong/data

if [ ${task} == 'weight' ];then
    config_name=train_weight_estimation_config.yaml
elif [ ${task} == 'dynamic_model' ];then
    config_name=train_dynamic_model_config.yaml
elif [ ${task} == 'silver_box' ];then
    config_name=train_silver_box_config.yaml
elif [ ${task} == 'simple_system' ];then
    config_name=train_simple_system_config.yaml
else
	echo "[ERROR]: task must be 'weight', 'dynamic_model' or 'silver_box'"
fi

python ${lwd}/src/time_series_network.py --config-filepath ${config_name} --task ${mode}
 