*****************************************************************************************************************
Documentation :
  TensorFlow for A64FX (2020/12/3)
*****************************************************************************************************************

0. Target node
  - FX1000

1. Requirements
　・FUJITSU Software Compiler Package is already installed.
  ・The login/compute node has access to the external network.

2. Preparation
  2-1. Checkout from Repository.
          # git clone https://github.com/fujitsu/tensorflow.git
          # cd tensorflow
          # git checkout -b fujitsu_v2.1.0_for_a64fx origin/fujitsu_v2.1.0_for_a64fx

  2-2. Environment Setting

          # cd fcc_build_script

      Modify the following environment variables in "env.src".

          ```
          ################################################
          ## Please change the following to suit your environment.
          ## PREFIX       : The directory where this file is located.
          ## TCSDS_PATH   : TCS installation path
          ## INSTALL_PATH : The directory to install TensorFlow environment
          ################################################
          export PREFIX=/home/user/work/tensorflow/fcc_build_script
          export TCSDS_PATH=/opt/FJSVxtclanga/tcsds-1.2.27a
          export INSTALL_PATH=/home/user/work/tf_env
          ```

      Modify each batch files to suit your environment.

          ```
          #!/bin/bash
          #PJM -L "rscunit=rscunit_ft01,rscgrp=ai-default"
          #PJM -L elapse=04:00:00
          #PJM -L "node=1"
          #PJM -j
          #PJM -S
          ```

3. Make TensorFlow Env
  Login Node[Estimated time:4h]

  3-1. Build fcc-Python 
    Run build-script on compute node.
        # pjsub 01_python_build.sh

  3-2. Build Bazel
    Run build-script on compute node.
        # pjsub 02_bazel_build.sh

    * Bazel build needs some requirements, please refer to the URL below. 
      https://docs.bazel.build/versions/0.29.1/install-compile-source.html#bootstrap-bazel

  3-3. Build oneDNN(dnnl_aarch64)
    Run build-script on compute node.
        # pjsub 03_oneDNN_build.sh

  3-4. Make venv
    Run build-script on compute node.
        # pjsub 04_make_venv.sh

  3-5. Build TensorFlow
    1. Run script and modify the tf src code.
        # bash 05-0_set_tf_src.sh

    2. Run build and install script on compute node.
        # pjsub 05_tf_build.sh
        (after build)
        # pjsub 06_tf_install.sh

    3. Run horovod build and install script on compute node.
        # pjsub 07_horovod_install.sh

  3-6. Check the environment
    1. Download and setup sample programs(Resnet50)
        # pjsub 10_setup_resnet.sh

    2. Run the sample programs
        # pjsub 11_train_resnet-single.sh
        # pjsub 12_train_resnet-4process.sh
        # pjsub 13_train_resnet-32process.sh

        Example of output(J11_train_resnet-single.sh.xxx.out)
        ```
        ~
        I1126 15:03:27.441569 281473877850720 basic_session_run_hooks.py:262] loss = 9.019469, step = 0
        INFO:tensorflow:global_step/sec: 0.104333
        I1126 15:03:36.979701 281473877850720 basic_session_run_hooks.py:702] global_step/sec: 0.104333
        INFO:tensorflow:loss = 9.019469, step = 1 (9.539 sec)
        I1126 15:03:36.980389 281473877850720 basic_session_run_hooks.py:260] loss = 9.019469, step = 1 (9.539 sec)
        INFO:tensorflow:global_step/sec: 0.450652
        I1126 15:03:39.198705 281473877850720 basic_session_run_hooks.py:702] global_step/sec: 0.450652
        INFO:tensorflow:loss = 8.997392, step = 2 (2.219 sec)
        I1126 15:03:39.199366 281473877850720 basic_session_run_hooks.py:260] loss = 8.997392, step = 2 (2.219 sec)
        INFO:tensorflow:global_step/sec: 0.450858
        I1126 15:03:41.416687 281473877850720 basic_session_run_hooks.py:702] global_step/sec: 0.450858

        ~
        INFO:tensorflow:Starting to evaluate.
        ~
        INFO:tensorflow:step = 1 time = 2.359 [sec]
        I1126 15:05:07.577002 281473877850720 resnet_run_loop.py:777] step = 1 time = 2.359 [sec]
        INFO:tensorflow:step = 2 time = 0.709 [sec]
        I1126 15:05:08.287052 281473877850720 resnet_run_loop.py:777] step = 2 time = 0.709 [sec]
        INFO:tensorflow:Evaluation [2/25]
        I1126 15:05:08.287598 281473877850720 evaluation.py:167] Evaluation [2/25]
        ~
        ```
