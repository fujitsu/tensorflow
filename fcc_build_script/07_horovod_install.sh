#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src
. $INSTALL_PATH/$VENV_NAME/bin/activate

export MPI_HOME=$TCSDS_PATH
export HOROVOD_MPICXX_SHOW="${MPI_HOME}/bin/mpiFCC -show"

export CC="fcc -Nclang -Kfast -Knolargepage"
export CXX="FCC -Nclang -Kfast -Knolargepage"
export MAX_JOBS=48

VER="0.19.5"

HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_MPI=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1 pip3 install horovod==$VER --no-cache-dir

pip3 list
