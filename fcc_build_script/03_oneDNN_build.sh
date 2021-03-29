#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src

pushd $INSTALL_PATH

git clone http://github.com/fujitsu/oneDNN
cd oneDNN
# checkout v2.1.0L01_aarch64
git checkout v2.1.0L01_aarch64
git submodule update --init --recursive

DNNL_BUILD_DIR=${INSTALL_PATH}/oneDNN/build
mkdir -p ${DNNL_BUILD_DIR}

export CC="fcc -Nclang -Kfast -Knolargepage -Kopenmp -Nlibomp"
export CXX="FCC -Nclang -Kfast -Knolargepage -Kopenmp -Nlibomp"

.github/automation/env/xed.sh -n
.github/automation/env/xbyak_translator_aarch64.sh
.github/automation/build.sh --threading omp --mode Release --source-dir $(pwd) --build-dir $(pwd)/build

popd

ldd ${DNNL_BUILD_DIR}/src/libdnnl.so

echo "#end"

