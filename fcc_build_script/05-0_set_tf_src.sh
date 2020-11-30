#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src

pushd ../

sed -i "52s/\/TCSDS_PATH/${TCSDS_PATH//\//\\/}/g" third_party/eigen.BUILD
sed -i "151s/\/INSTALL_PATH/${INSTALL_PATH//\//\\/}/g" WORKSPACE
sed -i "159s/\/INSTALL_PATH/${INSTALL_PATH//\//\\/}/g" WORKSPACE

popd
