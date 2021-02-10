#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src
. $INSTALL_PATH/$VENV_NAME/bin/activate

pushd ../
sed -i "119s/\/INSTALL_PATH/${INSTALL_PATH//\//\\/}/g" WORKSPACE
sed -i "127s/\/INSTALL_PATH/${INSTALL_PATH//\//\\/}/g" WORKSPACE
popd

pushd $INSTALL_PATH
wget https://www.r-ccs.riken.jp/labs/lpnctrt/projects/batchedblas/BatchedBLAS-1.0.tar.gz
tar -zxvf BatchedBLAS-1.0.tar.gz
cd BatchedBLAS-1.0
cp include/* ./
sed -i '325d' old/batched_blas.py
python3 old/batched_blas.py data/batched_blas_data.csv
ln -s $INSTALL_PATH/BatchedBLAS-1.0/batched_blas_src $INSTALL_PATH/batchedblas

cd batched_blas_src

sed -i '16d' batched_blas_common.h
sed -i '1,2d' Makefile
sed -i '1i CC=fcc' Makefile
sed -i '2i CCFLAG=-O3 -Nclang -Kopenmp,fast -Wall -D_CBLAS_ -I./' Makefile
sed -i "3i CCFLAG+=-I${TCSDS_PATH}/include/" Makefile

make -j32

popd
