#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

run_maskrcnn () {
  . ../../env.src
  . $INSTALL_PATH/$VENV_NAME/bin/activate

  WORK_PATH=`pwd`
  MODEL_PATH=$INSTALL_PATH/MaskRCNN
  MODEL_DIR=$WORK_PATH/model_outputs

  if [ -d $MODEL_DIR ];then
    rm -rf $MODEL_DIR
  fi

  cd $MODEL_PATH
  export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/research:$(pwd)/research/slim
  export PIPELINE_CONFIG_PATH=${WORK_PATH}/config/mask_rcnn_resnet50_fpn_coco.config
  cd research/object_detection

  ldd `dirname \`which python3\``/../lib/python3.8/site-packages/tensorflow/libtensorflow_framework.so.2

  export OMP_NUM_THREADS=22
  export TF_NUM_INTEROP_THREADS=1
  export TF_NUM_INTRAOP_THREADS=22
  export KMP_SETTINGS=1
  export KMP_BLOCKTIME=1
  export KMP_AFFINITY=granularity=fine,scatter,1,0

  export BATCH_SIZE=${BATCH_SIZE:=2}
  #NUMACTL="numactl -m 4,5 -N 4,5"

  #${NUMACTL} python3 model_main_tf2.py \
  python3 model_main_tf2.py \
	   --model_dir=${MODEL_DIR} \
	   --num_train_steps=20 \
	   --sample_1_of_n_eval_examples=1 \
	   --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	   --alsologtostderr \
	   --batch_size ${BATCH_SIZE}
}


DATE=`date "+%Y%m%d_%H%M%S"`
LOGFILE="./${DATE}_MaskRCNN.log"

run_maskrcnn 2>&1 | tee ${LOGFILE}

