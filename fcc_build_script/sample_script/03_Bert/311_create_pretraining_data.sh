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
BERT_DIR=$WORK_PATH/cased_L-12_H-768_A-12

mkdir -p $DATA_DIR

pushd $DATA_DIR
wget http://raw.githubusercontent.com/google-research/bert/master/sample_text.txt
popd

pushd $INSTALL_PATH/Bert
export PYTHONPATH=`pwd`
cd official/nlp/bert

python3 ../data/create_pretraining_data.py        \
  --input_file=$DATA_DIR/sample_text.txt          \
  --output_file=$DATA_DIR/tf_examples.tfrecord    \
  --vocab_file=$BERT_DIR/vocab.txt                \
  --do_lower_case=False                           \
  --max_seq_length=128                            \
  --max_predictions_per_seq=20                    \
  --masked_lm_prob=0.15                           \
  --random_seed=12345

popd
