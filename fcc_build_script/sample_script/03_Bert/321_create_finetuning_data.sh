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
PATCH=$WORK_PATH/download_glue_data.py.patch

mkdir -p $GLUE_DIR

pushd $GLUE_DIR
wget http://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/1502038877f6a88c225a34450793fbc3ea87eaba/download_glue_data.py

patch -u < $PATCH
python3 download_glue_data.py --tasks MRPC 

popd

pushd $INSTALL_PATH/Bert
export PYTHONPATH=`pwd`
cd official/nlp/bert

TASK_NAME=MRPC
OUTPUT_DIR=${GLUE_DIR}/glue_data/${TASK_NAME}

python3 ../data/create_finetuning_data.py                             \
  --input_data_dir=${GLUE_DIR}/glue_data/${TASK_NAME}/                \
  --vocab_file=${BERT_DIR}/vocab.txt                                  \
  --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
  --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record   \
  --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data          \
  --fine_tuning_task_type=classification --max_seq_length=128         \
  --classification_task_name=${TASK_NAME}

popd
