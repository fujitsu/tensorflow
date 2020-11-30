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

bazel-bin/tensorflow/tools/pip_package/build_pip_package $INSTALL_PATH/tf_pkg

cd $INSTALL_PATH/tf_pkg
pip3 --no-cache-dir install tensorflow-*

pip3 list
popd
