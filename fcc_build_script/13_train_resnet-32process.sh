#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=02:00:00
#PJM -L "node=8:noncont"
#PJM --mpi "shape=8,proc=32"
#PJM -j
#PJM -S

set -ex

. ./env.src
. $INSTALL_PATH/$VENV_NAME/bin/activate

export HOROVOD_MPI_THREADS_DISABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=11

WORK_PATH=`pwd`
MODEL_PATH=$INSTALL_PATH/models
cd $MODEL_PATH
export PYTHONPATH=`pwd`
cd official/r1/resnet

PROC=32
MPI="mpirun -np $PROC"
INTER="-inter=1"
#INTRA="-intra=1"
HOROVOD="--horovod=True"
DATA="--synth"
CHECKPOINTDIR="resnet"
#BSIZE="--batch_size=61"
BSIZE="--batch_size=48"

TRAIN_OPT="$DATA --num_gpus=0 --max_train_steps=25 --train_epochs=1 --model_dir=${WORK_PATH}/checkpoint/${CHECKPOINTDIR} --clean $INTER $INTRA $HOROVOD $BSIZE"
EVAL_OPT="$DATA --num_gpus=0 --max_train_steps=25 --train_epochs=1 --model_dir=${WORK_PATH}/checkpoint/${CHECKPOINTDIR} --eval_only $INTER $INTRA $HOROVOD $BSIZE"

ulimit -s 8192

# train / eval
$MPI python3 imagenet_main.py ${TRAIN_OPT}

# only eval
#$MPI python3 imagenet_main.py ${EVAL_OPT}
