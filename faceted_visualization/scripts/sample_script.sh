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

FACETED_VIS_HOME=/home/nas2_userH/hyesulim/Dev/2023/11785-f23-prj/faceted_visualization


CONFIG_FILE_PATH=/home/nas2_userH/hyesulim/Dev/2023/11785-f23-prj/faceted_visualization/visualizer/config/run_configs.json
OPTIMIZER="AdamW"
# MODEL="RN50"
MODEL="RN50x4"
# LAYER="layer4_2_conv3"
LAYER="layer4_5_conv3"
# LINEAR_PROBE_LAYER="layer1_2_relu3"
LINEAR_PROBE_LAYER="layer1_0_conv3"
OBJECTIVE="channel"
LEARNING_RATE=0.05
WANDB_RUN_NAME="local-testing-with-transforms"
CHANNEL=512
# LINEAR_PROBE_PATH=/home/nas2_userH/hyesulim/Dev/2023/11785-f23-prj/linear_probe/logs/celeba/layer1_2_relu3/version_29/model_checkpoint.pth
LINEAR_PROBE_PATH=/home/nas2_userH/hyesulim/Dev/2023/11785-f23-prj/logs/celeba/RN50x4/layer1_0_conv3/version_2/model_checkpoint-last.pth
NEURON_X=3
NEURON_Y=3
IMAGE_W=224
IMAGE_H=224


python $FACETED_VIS_HOME/visualizer/main.py \
--random-seed $RANDOM_SEED \
--no-wandb \
--config-file-path $CONFIG_FILE_PATH \
--model $MODEL \
--layer $LAYER \
--linear-probe-layer $LINEAR_PROBE_LAYER \
--opt $OPTIMIZER \
--objective $OBJECTIVE \
--learning-rate $LEARNING_RATE \
--wandb-run-name $WANDB_RUN_NAME \
--linear-probe-path "$LINEAR_PROBE_PATH" \
--channel $CHANNEL \
--neuron-x $NEURON_X \
--neuron-y $NEURON_Y \
--image-width $IMAGE_W \
--image-height $IMAGE_H \
--use-transforms \
--decorrelate 1 \
--fft 1 \
--num-iterations 512
