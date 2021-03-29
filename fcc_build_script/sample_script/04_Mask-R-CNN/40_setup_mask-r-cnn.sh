#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ../../env.src

SETUP_PATH="${INSTALL_PATH}/.local"
VENV_PATH="${INSTALL_PATH}/${VENV_NAME}"
export CC="fcc -Nclang -Kfast -Knolargepage"
export CXX="FCC -Nclang -Kfast -Knolargepage"

. $VENV_PATH/bin/activate

PATCH=`pwd`/MASK-R-CNN.patch
WORK_PATH=`pwd`

protocl_buffer () {
# --------------
# Protocol Buffer
# --------------
cd ${SETUP_PATH}
wget -O protobuf.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.12.4/protoc-3.12.4-linux-aarch_64.zip
unzip -o protobuf.zip && rm protobuf.zip
}

jpeg () {
# --------------
# Install jpeg-9d
# --------------
cd ${INSTALL_PATH}
curl -O http://www.ijg.org/files/jpegsrc.v9d.tar.gz
tar zxf jpegsrc.v9d.tar.gz
cd jpeg-9d/
./configure --prefix="${SETUP_PATH}" --enable-shared
make clean
make -j32
make install
}

pillow () {
# --------------
# Install Pillow
# --------------
PILLOW_VERSON="7.2.0"
cd ${INSTALL_PATH}
git clone https://github.com/python-pillow/Pillow.git --depth 1 -b ${PILLOW_VERSON}
cd Pillow
MAX_CONCURRENCY=16 CFLAGS="${CFLAGS} -I${SETUP_PATH}/include/" pip3 install .
cd ${INSTALL_PATH}
python3 -c 'import PIL'
}

matplotlib () {
# --------------
# Install Matplotlib
# --------------
pip3 uninstall enum34
pip3 install matplotlib==3.3.2
}

opencv () {
# --------------
# Install OpenCV
# --------------

cd ${INSTALL_PATH}
git clone https://github.com/opencv/opencv.git --depth 1 -b 4.3.0
cd opencv && mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${SETUP_PATH} \
      -DBUILD_opencv_python3=ON \
      -DPYTHON_INCLUDE_DIRS=${SETUP_PATH}/include \
      -DPYTHON_LIBRARY=${SETUP_PATH}/lib/libpython3.so \
      -DPYTHON3_EXECUTABLE=${VENV_PATH}/bin/python3 \
      -DPYTHON3_LIBRARIES=${SETUP_PATH}/lib/libpython3.so \
      -DPYTHON3_PACKAGES_PATH=${VENV_PATH}/lib/python3.8/site-packages \
      -DPYTHON3_NUMPY_INCLUDE_DIRS=${VENV_PATH}/lib/python3.8/site-packages/numpy/core/include ..

make -j32
make install

# Test
python3 -c 'import cv2'
}

others () {
pip3 install pycocotools dataclasses tf_slim lxml contextlib2
pip3 install lvis --no-deps
}

download_models () {
cd ${INSTALL_PATH}
git clone https://github.com/tensorflow/models.git MaskRCNN
cd MaskRCNN
git checkout dc4d11216b738920d
patch -p1 < $PATCH

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/research:$(pwd)/research/slim
cd research
protoc object_detection/protos/*.proto --python_out=.
#echo test
#python3 object_detection/builders/model_builder_tf2_test.py
}

make_config () {
cd ${WORK_PATH}
sed -i "168s/\/INSTALL_PATH/${INSTALL_PATH//\//\\/}/g" config/mask_rcnn_resnet50_fpn_coco.config
sed -i "186s/\/INSTALL_PATH/${INSTALL_PATH//\//\\/}/g" config/mask_rcnn_resnet50_fpn_coco.config
}

protocl_buffer
jpeg
pillow
matplotlib
opencv
others
download_models
make_config

