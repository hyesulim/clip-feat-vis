#!bin/bash

cd ..

for lr in 10.0 1.0 0.1 0.05 0.01
do
for opt in 'adam' 'adamw' 'sgd'
do
for channel in 1 486 512 1000
do
for its in 512
do

CUDA_VISIBLE_DEVICES=6 python main.py \
--optim $opt \
--lr $lr \
--iters $its \
--c_decorr \
--fourier_basis \
--obj layer4_2_relu3:${channel} \
--tfm "pad;jitter;rscale;rotate" \
--save "./results" \
--filename c${channel}_${opt}_lr${lr}_it${its}

done
done
done
done