#!bin/bash

cd ..

for random in 1 2 3 4
do
for lr in 0.05
do
for opt in 'adam'
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
--filename run${random}_c${channel}_${opt}_lr${lr}_it${its}

done
done
done
done
done