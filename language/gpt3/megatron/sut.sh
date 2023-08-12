
mode=$1
model=./model/
dataset=./data/cnn_eval.json

python main.py \
    --scenario=${mode} \
    --dataset-path=${dataset} \
    --max_examples=50 \
    --accuracy
