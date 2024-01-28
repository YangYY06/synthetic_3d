CUDA_VISIBLE_DEVICES=0 python prepare_training_data.py \
     --dataset_name car \
     --categories car \
     --dataset_path ../data/car_raw \
     --splits train \
     --num_images 7000 \
     --part_mask False \
     --filename_prefix P3D-Diffusion \
>car.txt 2>&1 &