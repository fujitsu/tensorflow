#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src

pushd $INSTALL_PATH

$INSTALL_PATH/.local/bin/python3 -m venv ./$VENV_NAME

. ./$VENV_NAME/bin/activate

pip3 install cython
pip3 install numpy==1.18.4
CC=gcc pip3 install grpcio==1.29.0

# for scipy install
export LAPACK=$TCSDS_PATH/lib64/libfjlapack.so
export BLAS=${LAPACK}
pip3 install scipy==1.4.1

pip3 install $WHEEL_PATH/h5py-2.10.0-cp38-cp38-linux_aarch64.whl

pip3 install Keras-Applications==1.0.8 --no-deps
pip3 install Keras-Preprocessing==1.1.2 --no-deps
pip3 install wheel

pip3 list 

popd
