#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=02:00:00
#PJM -L "node=1:noncont"
#PJM --mpi "shape=1,proc=4"
#PJM -j
#PJM -S

set -ex

. ../../env.src
. $INSTALL_PATH/$VENV_NAME/bin/activate

export HOROVOD_MPI_THREADS_DISABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=11

export KMP_SETTINGS=1
export KMP_BLOCKTIME=1

WORK_PATH=`pwd`
MODEL_PATH=$INSTALL_PATH/models
cd $MODEL_PATH
export PYTHONPATH=`pwd`
cd official/r1/resnet

PROC=4
MPI="mpirun -np $PROC"
#MPI="mpirun --prefix /opt/FJSVstclanga/v1.1.0 --hostfile $WORK_PATH/hostfile -mca pml ob1"
INTER="-inter=1"
#INTRA="-intra=1"
HOROVOD="--horovod=True"
DATA="--synth"
CHECKPOINTDIR="resnet"
BSIZE="--batch_size=60"

TRAIN_OPT="$DATA --num_gpus=0 --max_train_steps=25 --train_epochs=1 --model_dir=${WORK_PATH}/checkpoint/${CHECKPOINTDIR} --clean $INTER $INTRA $HOROVOD $BSIZE"
EVAL_OPT="$DATA --num_gpus=0 --max_train_steps=25 --train_epochs=1 --model_dir=${WORK_PATH}/checkpoint/${CHECKPOINTDIR} --eval_only $INTER $INTRA $HOROVOD $BSIZE"

ulimit -s 8192

# train / eval
$MPI python3 imagenet_main.py ${TRAIN_OPT}

# only eval
#$MPI python3 imagenet_main.py ${EVAL_OPT}
