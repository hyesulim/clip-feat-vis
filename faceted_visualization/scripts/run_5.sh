#!/bin/bash

# if [[ "$OSTYPE" == "linux-gnu"* ]]; then
#   RANDOM_SEED=$(date +%s%N)
# elif [[ "$OSTYPE" == "darwin"* ]]; then
#   RANDOM_SEED=$(gdate +%s%N)
# elif [[ "$OSTYPE" == "cygwin" ]]; then
#   RANDOM_SEED=$(date +%s%N)
# elif [[ "$OSTYPE" == "msys" ]]; then
#  RANDOM_SEED=$(date +%s%N)
# elif [[ "$OSTYPE" == "freebsd"* ]]; then
#  RANDOM_SEED=$(date +%s%N)
# else
RANDOM_SEED=1
#fi

for lpd in "celeba" "sun397" "aircraft"
do
LP_DATASET=${lpd}

# Paths to be set by the user
FT_CKPT_PATH='/data1/changdae/11785-f23-prj/RN50x4_FT.pt'
FACETED_VIS_HOME=/data1/changdae/11785-f23-prj/faceted_visualization
CONFIG_FILE_PATH=/data1/changdae/11785-f23-prj/faceted_visualization/visualizer/config/run_configs.json
LINEAR_PROBE_PATH=/data1/changdae/11785-f23-prj/linear_probe/logs/${LP_DATASET}/_RN50x4_FT
OUT_PATH=/data1/changdae/11785-f23-prj/faceted_visualization/runs/${LP_DATASET}/RN50x4_FT

# model settings
MODELS=("RN50x4")
LAYERS=("layer4_5_conv3")
LINEAR_PROBE_LAYERS=("layer1_3_relu" "layer2_5_relu" "layer3_9_relu")
OBJECTIVES=("neuron")

#OPTIMIZERS=("AdamW" "Adam" "SGD")
OPTIMIZERS=("Adam")
ITERATIONS=(512)
LEARNING_RATES=(5e-2)
CHANNELS=(1 100 200 300 400 512 600 700 800 900 1000)
NEURON_XS=(5)
NEURON_YS=(5)
IMAGE_WS=(224)
IMAGE_HS=(224)

# other settings
#WANDB_RUN_NAME="local-testing-with-transforms"
WANDB_PROJECT=idl_fvis_${lpd}
FFT=(1)
DECORRELATE=(1)
#WANDBFLAGS=("--no-wandb") # "--wandb" 
WANDBFLAGS=("--wandb")
TRANSFORMSFLAGS=("--use-transforms") #"--no-use-transforms" 

for model in ${MODELS[@]}; do
  for layer in ${LAYERS[@]}; do
    for lp in ${LINEAR_PROBE_LAYERS[@]}; do
      for obj in ${OBJECTIVES[@]}; do
        for opt in ${OPTIMIZERS[@]}; do
          for it in ${ITERATIONS[@]}; do
            for lr in ${LEARNING_RATES[@]}; do
              for channel in ${CHANNELS[@]}; do
                for neuron_x in ${NEURON_XS[@]}; do
                  for neuron_y in ${NEURON_YS[@]}; do
                    for image_w in ${IMAGE_WS[@]}; do
                      for image_h in ${IMAGE_HS[@]}; do
                        for fft in ${FFT[@]}; do
                          for decorrelate in ${DECORRELATE[@]}; do
                            for wandb in ${WANDBFLAGS[@]}; do
                              for transforms in ${TRANSFORMSFLAGS[@]}; do
                                # you can change string like this: 
                                # "$model"_"$layer"_"$lp"_"$obj"_"$opt"_"$it"_"$lr"_"$channel"
                                WANDB_RUN_NAME="$WANDB_RUN_NAME" 
                                CUDA_VISIBILE_DEVICES=7 python $FACETED_VIS_HOME/visualizer/main.py \
                                  --random-seed $RANDOM_SEED \
                                  --config-file-path $CONFIG_FILE_PATH \
                                  --ckpt-path $FT_CKPT_PATH \
                                  --linear-probe-path $LINEAR_PROBE_PATH \
                                  --output_directory $OUT_PATH \
                                  --model $model \
                                  --layer $layer \
                                  --linear-probe-layer $lp \
                                  --objective $obj \
                                  --opt $opt \
                                  --num-iterations $it \
                                  --learning-rate $lr \
                                  --channel $channel \
                                  --neuron-x $neuron_x \
                                  --neuron-y $neuron_y \
                                  --image-width $image_w \
                                  --image-height $image_h \
                                  --fft $fft \
                                  --decorrelate $decorrelate \
                                  $wandb \
                                  $transforms \
                                  --wandb-run-name tar:${layer}_c${channel}_${lp}_${obj}_nx${neuron_x}_ny${neuron_y} \
                                  --wandb-pj-name $WANDB_PROJECT
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

done