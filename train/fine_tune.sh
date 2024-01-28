# Adjust the number of images used for fine-tuning by --num
# By default, --num 92 means using 10 percent of the training data

CUDA_VISIBLE_DEVICES=0 python pseudo.py \
    --exp_type fine_tune \
    --num 92 \
    > car_fine_tune.txt 2>&1 &