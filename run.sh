CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Submodular --model ResNet18 --lr 0.1 -sp ./result/10 --batch 128

CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.05 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Submodular --model ResNet18 --lr 0.1 -sp ./result/05 --batch 128

CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.01 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Submodular --model ResNet18 --lr 0.1 -sp ./result/01 --batch 128