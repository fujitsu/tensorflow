#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

. ./env.src

export CC="fcc -Nclang -Kfast -Knolargepage"
export CXX="FCC -Nclang -Kfast -Knolargepage"
export OPT=-O3
export ac_cv_opt_olimit_ok=no
export ac_cv_olimit_ok=no
export ac_cv_cflags_warn_all=''


pushd $INSTALL_PATH

wget https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tgz
tar -zxvf Python-3.8.2.tgz
cd Python-3.8.2

./configure --enable-shared --disable-ipv6 --target=aarch64 --build=aarch64 --prefix=${INSTALL_PATH}/.local
make -j48
mv python python_org
${CXX} --linkfortran -SSL2 -Kopenmp -Nlibomp -o python Programs/python.o -L. -lpython3.8 -ldl  -lutil   -lm
make install

cd ${INSTALL_PATH}/.local/bin
ln -s python3 python

popd


