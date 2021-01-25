#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=02:00:00
#PJM -L "node=1:noncont"
#PJM -j
#PJM -S

set -ex

. ../../env.src
. $INSTALL_PATH/$VENV_NAME/bin/activate

export OMP_PROC_BIND=false

WORK_PATH=`pwd`
MODEL_PATH=$INSTALL_PATH/models
cd $MODEL_PATH
export PYTHONPATH=`pwd`
cd official/r1/resnet

INTER="-inter=1"
#INTRA="-intra=1"
DATA="--synth"
CHECKPOINTDIR="resnet"
BSIZE="--batch_size=61"

TRAIN_OPT="$DATA --num_gpus=0 --max_train_steps=25 --train_epochs=1 --model_dir=${WORK_PATH}/checkpoint/${CHECKPOINTDIR} --clean $INTER $INTRA $BSIZE"
EVAL_OPT="$DATA --num_gpus=0 --max_train_steps=25 --train_epochs=1 --model_dir=${WORK_PATH}/checkpoint/${CHECKPOINTDIR} --eval_only $INTER $INTRA $BSIZE"

ulimit -s 8192

NUMA_NODE=1
NUMA="numactl --membind=$NUMA_NODE --cpunodebind=$NUMA_NODE"

# train / eval
python3 imagenet_main.py ${TRAIN_OPT}
#$NUMA python3 imagenet_main.py ${TRAIN_OPT}

# only eval
#python3 imagenet_main.py ${EVAL_OPT}
