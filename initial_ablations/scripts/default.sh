#!bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--optim 'adam' \
--lr 5e-2 \
--iters 512 \
--c_decorr \
--fourier_basis \
--obj "layer4_2_relu3:486" \
--tfm "pad;jitter;rscale;rotate" \
--save "./results" \
--filename default_cfg