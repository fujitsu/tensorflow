#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=02:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

run_transformer () {
  . ../../env.src
  . $INSTALL_PATH/$VENV_NAME/bin/activate

  export OMP_NUM_THREADS=9
  export OMP_PROC_BIND=FALSE
  export TF_NUM_INTEROP_THREADS=2
  export TF_NUM_INTRAOP_THREADS=24

  export KMP_SETTINGS=1
  export KMP_BLOCKTIME=1

  ldd `dirname \`which python3\``/../lib/python3.8/site-packages/tensorflow/libtensorflow_framework.so.2

  #NUMA_NODE="0,1"
  #NUMA="numactl --membind=$NUMA_NODE --cpunodebind=$NUMA_NODE"

  onmt-main --model_type Transformer --config config.yml --auto_config train
  #$NUMA onmt-main --model_type Transformer --config config.yml --auto_config train
}

DATE=`date "+%Y%m%d_%H%M%S"`
LOGFILE="./${DATE}_OpenNMT.log"
RUNFILE="./${DATE}_run"

run_transformer 2>&1 | tee ${LOGFILE}
mv run/ $RUNFILE/
mv ${LOGFILE} $RUNFILE/
rm $RUNFILE/ckpt* -f
