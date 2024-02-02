#!/bin/bash
# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# Define the array of language codes
languages=("en" "zh-CN" "ar" "bg" "de" "el" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi")

# Loop through the array
for lang in "${languages[@]}"; do
    echo "Processing language: $lang"
    python models/run_vllm.py \
    --label exp \
    --language $lang \
    --data_root data/multi_lingual \
    --output_root results \
    --test_split test \
    --test_number 100 \
    --shot_number 0 \
    --model XComposer2 \
    --device-map cuda:3 \
    --prompt_format MCoT-Two \
    --seed 42
done
