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
# "celeba" "sun397" "air"
for lpd in "air"
do
LP_DATASET=${lpd}

# Paths to be set by the user
for FT_CKPT_PATH in '/data1/changdae/11785-f23-prj/RN50x4_FT.pt' '0'
do
FACETED_VIS_HOME=/data1/changdae/11785-f23-prj/faceted_visualization
CONFIG_FILE_PATH=/data1/changdae/11785-f23-prj/faceted_visualization/visualizer/config/run_configs.json

# model settings
MODELS=("RN50x4")
LAYERS=("layer4_5_relu" "layer4_5_conv3")
LINEAR_PROBE_LAYERS=("layer2_5_conv3" "layer2_5_relu" "layer3_9_conv3" "layer3_9_relu")
OBJECTIVES=("neuron")

#OPTIMIZERS=("AdamW" "Adam" "SGD")
OPTIMIZERS=("Adam")
ITERATIONS=(512)
LEARNING_RATES=(5e-2)
CHANNELS=(100 200 300 400 512 600 700 800 900 1000)
NEURON_XS=(2)
NEURON_YS=(2)
IMAGE_WS=(224)
IMAGE_HS=(224)

# other settings
#WANDB_RUN_NAME="local-testing-with-transforms"
if [[ $FT_CKPT_PATH == '0' ]]; then
  WANDB_PROJECT=idl_fvis_${lpd}_pt
  LINEAR_PROBE_PATH=/data1/changdae/11785-f23-prj/linear_probe/logs/${LP_DATASET}/RN50x4
  OUT_PATH=/data1/changdae/11785-f23-prj/faceted_visualization/runs/${LP_DATASET}/RN50x4_PT
else
  WANDB_PROJECT=idl_fvis_${lpd}_ft
  LINEAR_PROBE_PATH=/data1/changdae/11785-f23-prj/linear_probe/logs/${LP_DATASET}/_RN50x4_FT
  OUT_PATH=/data1/changdae/11785-f23-prj/faceted_visualization/runs/${LP_DATASET}/RN50x4_FT
fi
FFT=(1)
DECORRELATE=(1)
#WANDBFLAGS=("--no-wandb") # "--wandb" 
WANDBFLAGS=("--wandb")
STDTS=(0 1)
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
                                for stdt in ${STDTS[@]}; do
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
                                    --use-std-transforms $stdt \
                                    $wandb \
                                    $transforms \
                                    --wandb-run-name tar:${layer}_c${channel}_lp:${lp}_${obj}_std${stdt} \
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

done
done