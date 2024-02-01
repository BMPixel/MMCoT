export PYTHONPATH=$PYTHONPATH:$(pwd)/..
python run_vllm.py \
--label test \
--test_split test \
--test_number -1 \
--shot_number 2 \
--prompt_format QCM-ALE \
--seed 3
