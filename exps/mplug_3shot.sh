#!/bin/bash
# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# Run the python script
python run_vllm.py \
--label qwen_3shot \
--language en \
--test_split test \
--test_number 100 \
--shot_number 0 \
--model /cephfs/panwenbo/work/mmcot_assets/models/Qwen-VL \
--device-map cuda:5 \
--prompt_format MCoT-Two \
--seed 3