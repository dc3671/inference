#!/bin/bash
# This example will start serving the 345M model.
#DISTRIBUTED_ARGS="--nproc_per_node 1 \
#                  --nnodes 1 \
#                  --node_rank 0 \
#                  --master_addr localhost \
#                  --master_port 6000"

DISTRIBUTED_ARGS="-np 1 --prepend-rank python -u"
TOKENIZER_MODEL_FILE=./data/c4_en_301_5Mexp2_spm.model

export CUDA_DEVICE_MAX_CONNECTIONS=1

# pip install flask-restful

# mpirun -np 1 python dump_env.py

mpirun $DISTRIBUTED_ARGS text_generation_server.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 24  \
       --hidden-size 1024  \
       --num-attention-heads 16  \
       --max-position-embeddings 4096  \
       --tokenizer-type SentencePieceTokenizer  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --tokenizer-model $TOKENIZER_MODEL_FILE \
       --seed 42 \
       --ds-inference \
       --deepspeed
