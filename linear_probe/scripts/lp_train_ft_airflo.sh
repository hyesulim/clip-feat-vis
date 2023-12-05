#!bin/bash

cd ..
#cd linear_probe
model_arch='RN50x4'
ftp="/data1/changdae/11785-f23-prj/RN50x4_FT.pt"

for obj in 'layer1_0_relu' 'layer1_0_conv3' 'layer1_3_relu' 'layer1_3_conv3' 'layer2_5_relu' 'layer2_5_conv3' 'layer3_9_relu' 'layer3_9_conv3'
do
# for obj in 'layer1_2_relu'  'layer2_3_relu' 'layer3_5_relu' 'layer4_2_relu'
# do

CUDA_VISIBLE_DEVICES=2 python3 main.py \
--lp_dataset 'air' \
--optim 'sgd' \
--lr 1e-3 \
--batch_size 256 \
--subset_samples 3000 \
--num_epochs 20 \
--ftckpt_dir $ftp \
--obj $obj \
--model_arch $model_arch

CUDA_VISIBLE_DEVICES=2 python3 main.py \
--lp_dataset 'flower' \
--optim 'sgd' \
--lr 1e-3 \
--batch_size 256 \
--subset_samples 1000 \
--num_epochs 20 \
--ftckpt_dir $ftp \
--obj $obj \
--model_arch $model_arch

done