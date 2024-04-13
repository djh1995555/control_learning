#!/bin/bash
pwd=$(pwd -P)
lwd=$(dirname $pwd)
python ${lwd}/src/data_collector_for_dynamic_model.py
 