#!/bin/bash


wd=0.1
bs=512
method=ft
fp16=1

ts=0.0

for lr in 1e-5
do
CUDA_VISIBLE_DEVICES='6' python3 main.py \
--train-dataset=ImageNet --epochs=1 --lr ${lr} --wd ${wd} --batch-size $bs \
--model=RN50 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
--template=openai_imagenet_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" \
--csv-img-key filepath --csv-caption-key title --exp_name ImageNet/${method} \
--use_fp16 $fp16 --temperature_scale $ts \
--wb_project "IDL" --method $method
done