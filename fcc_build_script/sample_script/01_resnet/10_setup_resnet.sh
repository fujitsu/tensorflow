#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ../../env.src

PATCH=`pwd`/resnet.patch

pushd $INSTALL_PATH

git clone http://github.com/tensorflow/models
cd models
git checkout v2.0
patch -p1 < $PATCH

popd
