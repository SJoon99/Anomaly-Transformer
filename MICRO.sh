export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256  --mode train --dataset MICRO  --data_path MICRO   --input_c 15 --output_c 15