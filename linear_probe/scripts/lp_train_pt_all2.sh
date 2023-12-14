#!bin/bash

cd ..
#cd linear_probe
model_arch='RN50x4'
ftp="0"

for obj in 'layer1_3_relu' 'layer1_3_conv3' 'layer2_5_relu' 'layer2_5_conv3' 'layer3_9_relu' 'layer3_9_conv3'
do
# for obj in 'layer1_3_relu'  'layer2_3_relu' 'layer3_5_relu' 'layer4_2_relu'
# do

# CUDA_VISIBLE_DEVICES=3 python3 main.py \
# --lp_dataset 'celeba' \
# --optim 'sgd' \
# --lr 1e-3 \
# --batch_size 256 \
# --subset_samples 10000 \
# --num_epochs 20 \
# --ftckpt_dir $ftp \
# --obj $obj \
# --model_arch $model_arch

# CUDA_VISIBLE_DEVICES=3 python3 main.py \
# --lp_dataset 'air' \
# --optim 'sgd' \
# --lr 1e-3 \
# --batch_size 256 \
# --subset_samples 3000 \
# --num_epochs 20 \
# --ftckpt_dir $ftp \
# --obj $obj \
# --model_arch $model_arch

CUDA_VISIBLE_DEVICES=2 python3 main.py \
--lp_dataset 'sun397' \
--optim 'sgd' \
--lr 1e-3 \
--batch_size 256 \
--subset_samples 10000 \
--num_epochs 10 \
--ftckpt_dir $ftp \
--obj $obj \
--model_arch $model_arch

done