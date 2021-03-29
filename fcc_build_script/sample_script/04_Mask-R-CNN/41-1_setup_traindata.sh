#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=08:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ../../env.src

VENV_PATH="${INSTALL_PATH}/${VENV_NAME}"
COCO_DATA_DIR=${INSTALL_PATH}/coco

. $VENV_PATH/bin/activate

pushd $COCO_DATA_DIR

unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip annotations_trainval2017.zip
unzip image_info_test2017.zip

cd ${INSTALL_PATH}/MaskRCNN
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/research:$(pwd)/research/slim
cd research/object_detection

python3 dataset_tools/create_coco_tf_record.py --logtostderr \
       --train_image_dir="${COCO_DATA_DIR}/train2017" \
       --val_image_dir="${COCO_DATA_DIR}/val2017" \
       --test_image_dir="${COCO_DATA_DIR}/test2017" \
       --train_annotations_file="${COCO_DATA_DIR}/annotations/instances_train2017.json" \
       --val_annotations_file="${COCO_DATA_DIR}/annotations/instances_val2017.json" \
       --testdev_annotations_file="${COCO_DATA_DIR}/annotations/image_info_test-dev2017.json" \
       --output_dir="${COCO_DATA_DIR}/tf_record" \
       --include_masks

popd
