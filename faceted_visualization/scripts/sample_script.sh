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

FACETED_VIS_HOME=/Users/rohan/repositories/11785-f23-prj/faceted_visualization/

CONFIG_FILE_PATH=/Users/rohan/repositories/11785-f23-prj/faceted_visualization/visualizer/config/run_configs.json
OPTIMIZER="AdamW"
MODEL="RN50"
LAYER="layer4_2_conv3"
LINEAR_PROBE_LAYER="layer1_2_relu3"
OBJECTIVE="channel"
LEARNING_RATE=0.05
WANDB_RUN_NAME="local-testing-with-transforms"
CHANNEL=512
LINEAR_PROBE_PATH=/Users/rohan/repositories/11785-f23-prj/faceted_visualization/linear_probes/model_checkpoint\ \(1\).pth
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
--decorrelate 0 \
--fft 0 \
--num-iterations 512
