#!bin/bash

cd ..

CUDA_VISIBLE_DEVICES=1 python3 main.py \
--optim 'adam' \
--lr 5e-2 \
--iters 512 \
--c_decorr \
--fourier_basis \
--obj "layer4_2_relu:486" \
--tfm "pad;jitter;rscale;rotate" \
--save "./results" \
--ckpt_path "/data1/changdae/11785-f23-prj/RN50_FT.pt" \
--filename default_cfg