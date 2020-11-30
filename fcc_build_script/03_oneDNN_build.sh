#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src

pushd $INSTALL_PATH

git clone http://github.com/fujitsu/dnnl_aarch64 oneDNN
cd oneDNN
git checkout v1_0_0_base_0.21
git submodule update --init
sed -i 's/git@github.com:/http:\/\/github.com\//g' third_party/xbyak_translator_aarch64/.gitmodules
git submodule update --init --recursive

DNNL_SRC_DIR=`pwd`

echo "# XED build"
mkdir third_party/build_xed_aarch64
pushd third_party/build_xed_aarch64/
../xbyak_translator_aarch64/translator/third_party/xed/mfile.py --shared examples install
cd kits/
ln -sf xed-install-base-* xed
popd

DNNL_BUILD_DIR=${INSTALL_PATH}/oneDNN-build
mkdir -p ${DNNL_BUILD_DIR}
ln -sf ${DNNL_BUILD_DIR} ./build

export CC="fcc -Nclang -Kfast -Knolargepage -Kopenmp"
export CXX="FCC -Nclang -Kfast -Knolargepage -Kopenmp"

cd ${DNNL_BUILD_DIR}
cmake ${DNNL_SRC_DIR} -DWITH_BLAS=ssl2 2>&1 | tee -a cmake_fcc.sh.log

make -j 30
popd

ldd ${DNNL_BUILD_DIR}/src/libmkldnn.so

echo "#end"

