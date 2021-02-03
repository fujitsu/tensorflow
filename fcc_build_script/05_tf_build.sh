#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=05:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src
. $INSTALL_PATH/$VENV_NAME/bin/activate

pushd ../

BAZEL_BIN="$INSTALL_PATH/bazel"

# Environment variable for Tensorflow build.
export PYTHON_BIN_PATH="$INSTALL_PATH/$VENV_NAME/bin/python3"
export PYTHON_LIB_PATH="$INSTALL_PATH/$VENV_NAME/lib/python3.8/site-packages"
export TF_ENABLE_XLA=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0

export CC="fcc"
export CXX="FCC"

CONFIG="--config=noaws --config=nogcp --config=nohdfs --config=nonccl --distinct_host_configuration=false --config=mkl"
CONFIG_CC="--copt=-march=armv8.2-a+sve --copt=-O3"
CONFIG_CPP="--cxxopt=-march=armv8.2-a+sve --cxxopt=-O3 --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0"

#CONFIG_BAZEL="--verbose_failures --local_ram_resources=$((24*1024)) --local_cpu_resources=24 --jobs=24"
CONFIG_BAZEL="--verbose_failures --subcommands=pretty_print"

$BAZEL_BIN build ${CONFIG} ${CONFIG_CC} ${CONFIG_CPP} ${CONFIG_BAZEL}            \
           --action_env=fcc_ENV="-Nclang -Kfast -Knolargepage -Kopenmp -Nlibomp" \
           --action_env=FCC_ENV="-Nclang -Kfast -Knolargepage -Kopenmp -Nlibomp" \
           //tensorflow/tools/pip_package:build_pip_package

popd
