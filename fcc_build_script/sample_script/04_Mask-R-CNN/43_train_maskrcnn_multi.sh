#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1:noncont"
#PJM --mpi "shape=1,proc=2"
#PJM -j
#PJM -S

set -ex

run_maskrcnn_multi () {
  . ../../env.src
  . $INSTALL_PATH/$VENV_NAME/bin/activate

  WORK_PATH=`pwd`
  MODEL_PATH=$INSTALL_PATH/MaskRCNN
  MODEL_DIR=$WORK_PATH/model_outputs_multi

  if [ -d $MODEL_DIR ];then
    rm -rf $MODEL_DIR
  fi

  cd $MODEL_PATH
  export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/research:$(pwd)/research/slim
  export PIPELINE_CONFIG_PATH=${WORK_PATH}/config/mask_rcnn_resnet50_fpn_coco.config
  cd research/object_detection

  ldd `dirname \`which python3\``/../lib/python3.8/site-packages/tensorflow/libtensorflow_framework.so.2

  NPROC=${NPROC:=2}

  export CORE_PER_PROC=$((48 / ${NPROC}))
  export OMP_NUM_THREADS=$((${CORE_PER_PROC} - 2))
  export TF_NUM_INTEROP_THREADS=1
  export TF_NUM_INTRAOP_THREADS=${OMP_NUM_THREADS}

  export KMP_SETTINGS=1
  export KMP_BLOCKTIME=1
  export KMP_AFFINITY=granularity=fine,compact,1,0

  export OMPI_ALLOW_RUN_AS_ROOT=1
  export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  export MPI_RANK=${OMPI_COMM_WORLD_RANK}
  export MPI_SIZE=${OMPI_COMM_WORLD_SIZE}

  MPIRUN="mpirun --allow-run-as-root --map-by slot:PE=${CORE_PER_PROC}"

  export BATCH_SIZE=${BATCH_SIZE:=4}

  # Used only MultiWorkerMirroredStrategy
  export TF_CONFIG_PATH=${WORK_PATH}/config/tf_config_${NPROC}proc.json

  export USE_HOROVOD="--use_horovod"

  ${MPIRUN} -np $NPROC --display-map \
    python3 model_main_tf2.py \
       ${USE_HOROVOD} \
	     --model_dir=${MODEL_DIR} \
       --num_train_steps=20 \
       --sample_1_of_n_eval_examples=1 \
       --pipeline_config_path=$PIPELINE_CONFIG_PATH \
       --alsologtostderr \
       --num_workers ${NPROC} \
       --batch_size ${BATCH_SIZE}
}


DATE=`date "+%Y%m%d_%H%M%S"`
LOGFILE="./${DATE}_MaskRCNN_multi.log"

run_maskrcnn_multi 2>&1 | tee ${LOGFILE}

