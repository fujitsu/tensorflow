set -ex

. ../../env.src

COCO_DATA_DIR=${INSTALL_PATH}/coco

mkdir -p $COCO_DATA_DIR
pushd $COCO_DATA_DIR

curl -O http://images.cocodataset.org/zips/train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
curl -O http://images.cocodataset.org/zips/test2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -O http://images.cocodataset.org/annotations/image_info_test2017.zip

popd
