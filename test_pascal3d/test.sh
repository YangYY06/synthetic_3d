CUDA_VISIBLE_DEVICES=0 python pred_imagenet.py \
--ckpt ../experiments/P3D-Diffusion_car/ckpts/saved_model_2000.pth \
--crop_object True \
--category car \
--generate_pseudo False \
--save_results car_test.npy \
--px_sample 1 \
--py_sample 1 \
--azimuth_sample 25 \
--elevation_sample 10 \
--theta_sample 10 \
--evaluate_PCK False \
--distance 6 \
--split val \
> car_test.txt 2>&1 &


