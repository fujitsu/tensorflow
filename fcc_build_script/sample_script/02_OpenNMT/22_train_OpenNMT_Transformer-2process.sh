#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=02:00:00
#PJM -L "node=1:noncont"
#PJM --mpi "shape=1,proc=2"
#PJM -j
#PJM -S

set -ex

run_transformer () {
  . ../../env.src
  . $INSTALL_PATH/$VENV_NAME/bin/activate

  export OMP_NUM_THREADS=22
  export KMP_AFFINITY=granularity=fine,compact,1,0
  export TF_NUM_INTEROP_THREADS=1
  export TF_NUM_INTRAOP_THREADS=22

  export KMP_SETTINGS=1
  export KMP_BLOCKTIME=1

  ldd `dirname \`which python3\``/../lib/python3.8/site-packages/tensorflow/libtensorflow_framework.so.2

  export HOROVOD_MPI_THREADS_DISABLE=1
  MPI="mpirun -np 2"
  #MPI="mpirun --prefix /opt/FJSVstclanga/v1.1.0 --rankfile rankfile -np 2 -mca pml ob1"

  $MPI onmt-main --model_type Transformer --config config.yml --auto_config train --horovod

}

DATE=`date "+%Y%m%d_%H%M%S"`
LOGFILE="./${DATE}_OpenNMT_multi.log"
RUNFILE="./${DATE}_run_multi"

run_transformer 2>&1 | tee ${LOGFILE}
mv run/ $RUNFILE/
mv ${LOGFILE} $RUNFILE/
rm $RUNFILE/ckpt* -f
