#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  RANDOM_SEED=$(date +%s%N)
elif [[ "$OSTYPE" == "darwin"* ]]; then
  RANDOM_SEED=$(gdate +%s%N)
elif [[ "$OSTYPE" == "cygwin" ]]; then
  RANDOM_SEED=$(date +%s%N)
elif [[ "$OSTYPE" == "msys" ]]; then
 RANDOM_SEED=$(date +%s%N)
elif [[ "$OSTYPE" == "freebsd"* ]]; then
 RANDOM_SEED=$(date +%s%N)
else
 RANDOM_SEED=1
fi

# Paths to be set by the user
FACETED_VIS_HOME=/Users/rohan/repositories/11785-f23-prj/faceted_visualization/
CONFIG_FILE_PATH=/Users/rohan/repositories/11785-f23-prj/faceted_visualization/visualizer/config/run_configs.json
LINEAR_PROBE_PATH=/Users/rohan/repositories/11785-f23-prj/faceted_visualization/linear_probes/model_checkpoint\ \(1\).pth

# model settings
MODELS=("RN50")
LAYERS=("layer4_2_conv3")
LINEAR_PROBE_LAYERS=("layer1_2_relu3")
OBJECTIVES=("channel" "neuron")

OPTIMIZERS=("AdamW" "Adam" "SGD")
ITERATIONS=(512)
LEARNING_RATES=(1e-5)
CHANNELS=(512)
NEURON_XS=(3)
NEURON_YS=(3)
IMAGE_WS=(224)
IMAGE_HS=(224)

# other settings
WANDB_RUN_NAME="local-testing-with-transforms"
FFT=(0 1)
DECORRELATE=(0 1)
WANDBFLAGS=("--no-wandb" "--wandb")
TRANSFORMSFLAGS=("--no-use-transforms" "--use-transforms")

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
                                python $FACETED_VIS_HOME/visualizer/main.py \
                                  --random-seed $RANDOM_SEED \
                                  --config-file-path $CONFIG_FILE_PATH \
                                  --linear-probe-path "$LINEAR_PROBE_PATH" \
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
                                  --wandb-run-name $WANDB_RUN_NAME 
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