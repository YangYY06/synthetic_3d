CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_name P3D-Diffusion_car \
    --category car \
    --dataset_path ../data/car \
    --bg_path ../data/bg \
    --crop_object True \
    --rotate 5 \
    --batch_size 24 \
    --iterations 2000 \
    --save_itr 2000 \
    --log_itr 1 \
    --lr 0.00002 \
    --distance 6 \
    > car_train.txt 2>&1 &