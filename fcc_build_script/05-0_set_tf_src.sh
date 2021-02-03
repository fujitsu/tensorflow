#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src

pushd ../

sed -i "26s/\/ONEDNN_DIR/${ONEDNN_DIR//\//\\/}/g" third_party/mkl_dnn/mkldnn_v1.BUILD
sed -i "178s/\/INSTALL_PATH/${INSTALL_PATH//\//\\/}/g" tensorflow/workspace.bzl

popd
