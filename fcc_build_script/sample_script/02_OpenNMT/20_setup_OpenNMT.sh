#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=03:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ../../env.src
. $INSTALL_PATH/$VENV_NAME/bin/activate

BAZEL_BIN="$INSTALL_PATH/bazel"
PATCH=`pwd`/OpenNMT.patch

pushd $INSTALL_PATH

# build extension pkg
## tensorflow-addons
git clone http://github.com/tensorflow/addons
cd addons
git checkout v0.10.0
python3 ./configure.py
$BAZEL_BIN build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip3 install --no-cache-dir artifacts/tensorflow_addons-0.10.0-cp38-cp38-linux_aarch64.whl

## pyonmttok
cd $INSTALL_PATH
PYONMTTOK_VERSION="v1.18.3"
git clone https://github.com/OpenNMT/Tokenizer.git -b ${PYONMTTOK_VERSION}
cd Tokenizer
git submodule update --init --recursive
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}/.local
make -j32 && make install

cd ../bindings/python/
CFLAGS="-L${INSTALL_PATH}/.local/lib64 -I${INSTALL_PATH}/.local/include -I${INSTALL_PATH}/Tokenizer/include" python3 setup.py install

# build OpenNMT-tf
cd $INSTALL_PATH
git clone http://github.com/OpenNMT/OpenNMT-tf opennmt-tf
cd opennmt-tf
git checkout v2.11.0
patch -p1 < $PATCH

python3 setup.py sdist bdist_wheel

cd dist
pip3 install --no-cache-dir OpenNMT_tf-2.11.0-py3-none-any.whl
pip3 install --no-cache-dir pybind11
pip3 list

# make train dataset
popd
pushd train_data
bash make_dataset.sh

popd
