################################ Testing ################################
# first precompute and cluster all features
#python encode_features.py --name trained_512p_feat \
#    --phase train  \
#    --gpu_ids 1 \
#    --load_pretrain ckpt/trained_512p_feat

# use instance-wise features
python test.py --name trained_512p_feat --instance_feat \
    --gpu_ids 3 \
    --phase train \
    --cluster_path features_clustered_010.npy \
    --how_many 500 \
    --start_index 2500
