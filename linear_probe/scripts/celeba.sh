python script.py --root_data '/new/data/path' --lr 0.005 --num_epochs 20

CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 256

CUDA_VISIBLE_DEVICES=1 python main.py --batch_size 256 --optim sgd --obj layer4_2_relu2 --num_epochs 3

CUDA_VISIBLE_DEVICES=3 python main.py --batch_size 256 --optim sgd --obj layer2_3_relu3

CUDA_VISIBLE_DEVICES=1 python main.py --batch_size 256 --optim sgd --obj layer3_5_relu3 