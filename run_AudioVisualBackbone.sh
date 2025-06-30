#!/bin/bash

cd audio_visual_backbobe



python audio_visual_only_script.py


python audio_visual_only_script.py \
    --epochs 100 \
    --train_batch_size 64 \
    --lr 0.001 \
    --sim_loss_weight 0.1 \
    --lr_step_size 30