#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ../../env.src
. $INSTALL_PATH/$VENV_NAME/bin/activate

WORK_PATH=`pwd`

DATA_DIR=$WORK_PATH/pretraining_data
MODEL_DIR=$WORK_PATH/pretraining_output
BERT_DIR=$WORK_PATH/cased_L-12_H-768_A-12

pushd $INSTALL_PATH/Bert
export PYTHONPATH=`pwd`
cd official/nlp/bert

export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=24
export OMP_NUM_THREADS=9

if [ -d $MODEL_DIR ];then
  rm -rf $MODEL_DIR
fi

#numactl -m 2,3 -N 2,3 python run_pretraining.py  \
python3 run_pretraining.py                       \
  --input_files=$DATA_DIR/tf_examples.tfrecord   \
  --model_dir=$MODEL_DIR                         \
  --bert_config_file=$BERT_DIR/bert_config.json  \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt    \
  --train_batch_size=32                          \
  --max_seq_length=128                           \
  --max_predictions_per_seq=20                   \
  --num_train_epochs=2                           \
  --warmup_steps=10                              \
  --num_steps_per_epoch=10                       \
  --steps_per_loop=1                             \
  --log_steps=4

popd
