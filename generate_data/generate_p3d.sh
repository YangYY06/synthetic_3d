# Please update the path to Blender with your own path
# Modify elevation for different categories as per the paper's supplementary material
# Modify distance parameters as follows:
# if category in ['bus', 'car', 'motorbike', 'aeroplane', 'bicycle', 'boat', 'diningtable', 'train']:
#    distance = 6
# if category in ['chair', 'bottle']:
#    distance = 10
# if category in ['sofa', 'tvmonitor']:
      distance = 8
# Do not generate more than 500 images (num_images) for each thread.

CUDA_VISIBLE_DEVICES=0 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 0 --num_images 500 \
>car_0.txt  2>&1 &

CUDA_VISIBLE_DEVICES=1 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 500 --num_images 500 \
>car_1.txt  2>&1 &

CUDA_VISIBLE_DEVICES=2 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 1000 --num_images 500 \
>car_2.txt  2>&1 &

CUDA_VISIBLE_DEVICES=3 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 1500 --num_images 500 \
>car_3.txt  2>&1 &

CUDA_VISIBLE_DEVICES=0 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 2000 --num_images 500 \
>car_4.txt  2>&1 &

CUDA_VISIBLE_DEVICES=1 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 2500 --num_images 500 \
>car_5.txt  2>&1 &

CUDA_VISIBLE_DEVICES=2 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 3000 --num_images 500 \
>car_6.txt  2>&1 &

CUDA_VISIBLE_DEVICES=3 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 3500 --num_images 500 \
>car_7.txt  2>&1 &

CUDA_VISIBLE_DEVICES=0 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 4000 --num_images 500 \
>car_8.txt  2>&1 &

CUDA_VISIBLE_DEVICES=1 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 4500 --num_images 500 \
>car_9.txt  2>&1 &

CUDA_VISIBLE_DEVICES=2 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 5000 --num_images 500 \
>car_10.txt  2>&1 &

CUDA_VISIBLE_DEVICES=3 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 5500 --num_images 500 \
>car_11.txt  2>&1 &

CUDA_VISIBLE_DEVICES=0 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 6000 --num_images 500 \
>car_12.txt  2>&1 &

CUDA_VISIBLE_DEVICES=1 /home/jiahao/blender-2.79-linux-glibc219-x86_64/blender --background --python render_p3d.py -- \
    --model_dir data/pascal3dp_smartuv \
    --split train \
    --categories car_01 car_02 car_03 car_04 car_05 car_06 car_07 car_08 car_09 car_10 \
    --width 520 --height 352 \
    --properties_json data/properties_p3d.json \
    --min_objects 1 --max_objects 1 \
    --output_dir ../data/car_raw \
    --enable_dtd True \
    --save_part_mask True \
    --distance 6 \
    --elevation_mean 5 \
    --elevation_variance 8 \
    --elevation_max 50 \
    --elevation_min -15 \
    --stretch_x 0.05 \
    --stretch_y 0.05 \
    --stretch_z 0.05 \
    --start_idx 6500 --num_images 500 \
>car_13.txt  2>&1 &


