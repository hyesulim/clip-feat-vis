#!bin/bash

cd ..
cd linear_probe

CUDA_VISIBLE_DEVICES=7 python main.py \
--lp_dataset 'flower' \
--optim 'sgd' \
--lr 1e-3 \
--batch_size 256 \
--obj "layer1_2_relu3"