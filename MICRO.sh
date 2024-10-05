#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256  --mode train --dataset MICRO  --data_path MICRO   --input_c 15 --output_c 15
echo "==============================="
echo "Testing CVE: CVE-2016-5195"
echo "==============================="
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset MICRO --data_path ../CVE-2016-5195 --input_c 15 --output_c 15 --pretrained_model 20

echo "==============================="
echo "Testing CVE: CVE-2016-9793"
echo "==============================="
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset MICRO --data_path ../CVE-2016-9793 --input_c 15 --output_c 15 --pretrained_model 20

echo "==============================="
echo "Testing CVE: CVE-2017-1000112"
echo "==============================="
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset MICRO --data_path ../CVE-2017-1000112 --input_c 15 --output_c 15 --pretrained_model 20

echo "==============================="
echo "Testing CVE: CVE-2017-16939"
echo "==============================="
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset MICRO --data_path ../CVE-2017-16939 --input_c 15 --output_c 15 --pretrained_model 20

echo "==============================="
echo "Testing CVE: CVE-2017-16995"
echo "==============================="
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset MICRO --data_path ../CVE-2017-16995 --input_c 15 --output_c 15 --pretrained_model 20

echo "==============================="
echo "Testing CVE: CVE-2017-7308"
echo "==============================="
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset MICRO --data_path ../CVE-2017-7308 --input_c 15 --output_c 15 --pretrained_model 20

echo "==============================="
echo "Testing CVE: CVE-2019-5736"
echo "==============================="
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset MICRO --data_path ../CVE-2019-5736 --input_c 15 --output_c 15 --pretrained_model 20

echo "==============================="
echo "Testing CVE: CVE-2022-0492"
echo "==============================="
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset MICRO --data_path ../CVE-2022-0492 --input_c 15 --output_c 15 --pretrained_model 20