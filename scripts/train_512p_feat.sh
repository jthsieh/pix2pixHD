### Adding instances and encoded features
python train.py --name trained_512p_feat \
    --continue_train \
    --instance_feat  \
    --gpu_ids 3 \
    --checkpoints_dir ckpt/
