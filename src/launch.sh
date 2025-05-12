#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision "fp16" /home/chaichuk/Team73-Annual-Project/src/train_ip_adapter.py \
  --pretrained_model_name_or_path='stable-diffusion-v1-5/stable-diffusion-v1-5' \
  --image_encoder_path="/home/chaichuk/IP-Adapter/models/image_encoder" \
  --data_csv_file="/home/chaichuk/datasets/CelebAMask-HQ/captions.csv" \
  --data_root_path="/home/chaichuk/datasets/CelebAMask-HQ/CelebA-HQ-img" \
  --mixed_precision="fp16" \
  --wandb_run_name="IP-adapter-Plus" \
  --num_train_epochs=100 \
  --resolution=512 \
  --train_batch_size=64 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/home/chaichuk/Team73-Annual-Project/checkpoints/ip-adapter-plus" \
  --save_steps=-1 \
  --save_epochs=10 \
  --adapter_type='plus'