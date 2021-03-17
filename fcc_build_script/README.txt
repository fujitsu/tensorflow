*****************************************************************************************************************
Documentation :
  TensorFlow for A64FX (fujitsu_v2.2.0L02_for_a64fx 2021/2/24)
*****************************************************************************************************************

0. Target node
  - FX1000 / FX700

1. Requirements
　・FUJITSU Software Compiler Package is already installed.
  ・The login/compute node has access to the external network.

2. Preparation
  2-1. Checkout from Repository.
          # git clone https://github.com/fujitsu/tensorflow.git
          # cd tensorflow
          # git checkout -b fujitsu_v2.2.0_for_a64fx origin/fujitsu_v2.2.0_for_a64fx

  2-2. Environment Setting

          # cd fcc_build_script

      Make directory(with any name)
        - to install TensorFlow environment
        - temporary directory

          # mkdir -p /home/user/work/tf_env
          # mkdir -p ~/pytemp

      Modify the following environment variables in "env.src".

          ```
          ################################################
          ## Please change the following to suit your environment.
          ## PREFIX       : The directory where this file is located.
          ## TCSDS_PATH   : TCS installation path
          ## INSTALL_PATH : The directory to install TensorFlow environment
          ## TMPDIR       : The temporary directory
          ################################################
          export PREFIX=/home/user/work/tensorflow/fcc_build_script
          export TCSDS_PATH=/opt/FJSVxtclanga/tcsds-1.2.30a
          export INSTALL_PATH=/home/user/work/tf_env
          export TMPDIR=~/pytemp
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
      https://docs.bazel.build/versions/2.0.0/install-compile-source.html#bootstrap-bazel

  3-3. Build oneDNN
    Run build-script on compute node.
        # pjsub 03_oneDNN_build.sh

  3-4. Make venv
    Run build-script on compute node.
        # pjsub 04_make_venv.sh

  3-5. Build TensorFlow
    1. Run script and modify the tf src code.
        # bash 05-0_set_tf_src.sh

    2. Run script and build batched blas lib.
        # pjsub 05-1_build_batchedblas.sh

    3. Run build and install script on compute node.
        # pjsub 05_tf_build.sh
        (after build)
        # pjsub 06_tf_install.sh

    4. Run horovod build and install script on compute node.
        # pjsub 07_horovod_install.sh

4. Check the environment
  4-1. Resnet50
    1. Download and setup sample programs
        # cd sample_script/01_resnet
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

  4-2. OpenNMT-tf(Transformer)
    1. Download and setup sample programs
        # cd sample_script/02_OpenNMT
        # pjsub 20_setup_OpenNMT.sh

    2. Run the sample programs
        # pjsub 21_train_OpenNMT_Transformer-single.sh
        # pjsub 22_train_OpenNMT_Transformer-2process.sh

        Example of output(J22_train_OpenNMT_Transformer-2process.sh.xxx.out)
        ```
        ~
        INFO:tensorflow:Number of model parameters: 93655532
        INFO:tensorflow:Number of model weights: 260 (trainable = 260, non trainable = 0)
        INFO:tensorflow:Step = 1 ; steps/s = 0.00, source words/s = 0, target words/s = 0 ; Learning rate = 0.000000 ; Loss = 10.506068
        INFO:tensorflow:Saved checkpoint run/ckpt-1
        INFO:tensorflow:Step = 2 ; steps/s = 0.27, source words/s = 8, target words/s = 9 ; Learning rate = 0.000000 ; Loss = 10.490356
        INFO:tensorflow:Step = 3 ; steps/s = 2.79, source words/s = 89, target words/s = 111 ; Learning rate = 0.000000 ; Loss = 10.538855
        INFO:tensorflow:Step = 4 ; steps/s = 2.24, source words/s = 224, target words/s = 295 ; Learning rate = 0.000001 ; Loss = 10.467386
        INFO:tensorflow:Step = 5 ; steps/s = 1.94, source words/s = 520, target words/s = 609 ; Learning rate = 0.000001 ; Loss = 10.489294
        ~
        ```

  4-1. BERT(supported only single process)
    1. Download and setup sample programs
        # cd sample_script/03_Bert
        # pjsub 300_setup_bert.sh
        # pjsub 311_create_pretraining_data.sh
        # pjsub 321_create_finetuning_data.sh

    2. Run the sample programs
        # pjsub 312_run_pretraining.sh
        # pjsub 313_run_pretraining-2process.sh
        # pjsub 322_run_finetuning.sh
        # pjsub 323_run_finetuning-2process.sh

        Example of output(J312_run_pretraining.sh.xxx.out)
        ```
        ~
        29/917 [..............................] - ETA: 14:00 - test_accuracy: 0.7328 - loss: 0.6060I0128 13:08:30.564977 281473617147888 keras_utils.py:119] TimeHistory: 9.62 seconds, 4.16 examples/second between steps 20 and 30
        39/917 [>.............................] - ETA: 13:54 - test_accuracy: 0.7179 - loss: 0.6198I0128 13:08:40.184872 281473617147888 keras_utils.py:119] TimeHistory: 9.61 seconds, 4.16 examples/second between steps 30 and 40
        49/917 [>.............................] - ETA: 13:47 - test_accuracy: 0.7041 - loss: 0.6273I0128 13:08:49.796334 281473617147888 keras_utils.py:119] TimeHistory: 9.61 seconds, 4.16 examples/second between steps 40 and 50
        59/917 [>.............................] - ETA: 13:39 - test_accuracy: 0.7119 - loss: 0.6184I0128 13:08:59.445973 281473617147888 keras_utils.py:119] TimeHistory: 9.64 seconds, 4.15 examples/second between steps 50 and 60
        69/917 [=>............................] - ETA: 13:30 - test_accuracy: 0.7101 - loss: 0.6114I0128 13:09:09.058540 281473617147888 keras_utils.py:119] TimeHistory: 9.61 seconds, 4.16 examples/second between steps 60 and 70
        79/917 [=>............................] - ETA: 13:21 - test_accuracy: 0.7152 - loss: 0.6069I0128 13:09:18.673382 281473617147888 keras_utils.py:119] TimeHistory: 9.61 seconds, 4.16 examples/second between steps 70 and 80
        89/917 [=>............................] - ETA: 13:12 - test_accuracy: 0.7107 - loss: 0.6129I0128 13:09:28.282830 281473617147888 keras_utils.py:119] TimeHistory: 9.60 seconds, 4.17 examples/second between steps 80 and 90
        99/917 [==>...........................] - ETA: 13:03 - test_accuracy: 0.7096 - loss: 0.6098I0128 13:09:37.908488 281473617147888 keras_utils.py:119] TimeHistory: 9.62 seconds, 4.16 examples/second between steps 90 and 100
        ~
       ```
 
# Copyright
  Copyright RIKEN LIMITED 2021
  Copyright FUJITSU LIMITED 2021
