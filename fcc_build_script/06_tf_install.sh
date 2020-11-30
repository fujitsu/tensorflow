#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src
pushd ../
. $INSTALL_PATH/$VENV_NAME/bin/activate

TF_PKG=tf_pkg

bazel-bin/tensorflow/tools/pip_package/build_pip_package $INSTALL_PATH/$TF_PKG

cd $INSTALL_PATH/$TF_PKG

pip3 uninstall tensorflow -y
pip3 --no-cache-dir install tensorflow-2.2.0-cp38-cp38-linux_aarch64.whl

pip3 list
popd
