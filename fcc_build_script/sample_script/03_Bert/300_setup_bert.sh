#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ../../env.src
. $INSTALL_PATH/$VENV_NAME/bin/activate

PATCH=`pwd`/Bert.patch

# download checkpoint for finetuning
wget https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/cased_L-12_H-768_A-12.tar.gz
tar -xvzf cased_L-12_H-768_A-12.tar.gz

pushd $INSTALL_PATH

# download bert src
git clone http://github.com/tensorflow/models Bert
cd Bert
git checkout v2.2.0
patch -p1 < $PATCH

# install pip pkg
pip3 install --no-cache-dir sentencepiece
pip3 install --no-cache-dir gin-config
pip3 install --no-cache-dir tensorflow-hub

popd
