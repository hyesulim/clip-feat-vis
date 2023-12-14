#!bin/bash

cd ..
cd linear_probe

for obj in 'layer1_0_relu3' 'layer1_0_conv3' 'layer1_3_relu3' 'layer1_3_conv3' 'layer2_5_relu3' 'layer2_5_conv3' 'layer3_9_relu3' 'layer3_9_conv3'
do

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--lp_dataset 'air' \
--optim 'sgd' \
--lr 1e-3 \
--batch_size 256 \
--subset_samples 3000 \
--num_epochs 20 \
--obj $obj

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--lp_dataset 'flower' \
--optim 'sgd' \
--lr 1e-3 \
--batch_size 256 \
--subset_samples 1000 \
--num_epochs 20 \
--obj $obj

done