#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

set -ex

. ./env.src

PATCH=$PREFIX/bazel.patch

pushd $INSTALL_PATH

# download dist src
VER="0.29.1"
wget https://github.com/bazelbuild/bazel/releases/download/$VER/bazel-$VER-dist.zip
mkdir bazel-$VER
cd bazel-$VER
unzip ../bazel-$VER-dist.zip

patch -p1 < $PATCH

export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk
EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh

cd ../
ln -s ./bazel-$VER/output/bazel ./bazel

popd

