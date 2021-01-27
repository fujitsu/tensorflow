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

BERT_DIR=$WORK_PATH/cased_L-12_H-768_A-12
GLUE_DIR=$WORK_PATH/finetuning_glue_data
MODEL_DIR=$WORK_PATH/finetuning_output
TASK=MRPC

pushd $INSTALL_PATH/Bert
export PYTHONPATH=`pwd`
cd official/nlp/bert

export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=24
export OMP_NUM_THREADS=4

if [ -d $MODEL_DIR ];then
  rm -rf $MODEL_DIR
fi

#numactl -m 2,3 -N 2,3 python3 run_classifier.py                          \
python3 run_classifier.py                                                 \
  --mode='train_and_eval'                                                 \
  --input_meta_data_path=${GLUE_DIR}/glue_data/${TASK}/${TASK}_meta_data  \
  --train_data_path=${GLUE_DIR}/glue_data/${TASK}/${TASK}_train.tf_record \
  --eval_data_path=${GLUE_DIR}/glue_data/${TASK}/${TASK}_eval.tf_record   \
  --bert_config_file=${BERT_DIR}/bert_config.json                         \
  --init_checkpoint=${BERT_DIR}/bert_model.ckpt                           \
  --train_batch_size=4                                                    \
  --eval_batch_size=4                                                     \
  --steps_per_loop=1                                                      \
  --learning_rate=2e-5                                                    \
  --num_train_epochs=1                                                    \
  --model_dir=${MODEL_DIR}                                                \
  --distribution_strategy=mirrored                                        \
  --log_steps=10                                                          \
  --use_keras_compile_fit True

popd
