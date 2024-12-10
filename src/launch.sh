#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision "fp16" /home/chaichuk/Team73-Annual-Project/src/train_ip_adapter.py \
  --pretrained_model_name_or_path='dreamlike-art/dreamlike-anime-1.0' \
  --image_encoder_path="/home/chaichuk/IP-Adapter/sdxl_models/image_encoder" \
  --data_csv_file="/home/chaichuk/CelebAMask-HQ/new_celeba_captions.csv" \
  --data_root_path="/home/chaichuk/CelebAMask-HQ/CelebA-HQ-img" \
  --mixed_precision="fp16" \
  --wandb_run_name="run_512_sdxl_image_encoder_anime" \
  --num_train_epochs=100 \
  --resolution=512 \
  --train_batch_size=64 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/home/chaichuk/Team73-Annual-Project/weights/512_res_sdxl_image_encoder_anime_model" \
  --save_steps=-1 \
  --save_epochs=10