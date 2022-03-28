import os

#coarse-net

loss_weight=' '
loss_weight+=' --weight_cd 1.0'
loss_weight+=' --weight_uniform -10000000'
loss_weight+=' --weight_reg -0.1'
loss_weight+=' --weight_arap 0.03'
loss_weight+=' --weight_overlap 0.3'
loss_weight+=' --weight_proj -1'
loss_weight+=' --weight_normal -1'
loss_weight+=' --weight_cycle -1'
loss_weight+=' --weight_ndirection 0.0001'


for control_i in range(0,1):
    os.system('CUDA_VISIBLE_DEVICES=1 python train_view_toy.py \
        --training_up_ratio 16 \
        --testing_up_ratio 16 \
        --over_sampling_scale 4 \
        --visualization_while_testing 1 \
        --last_sample_id '+str(control_i*10000)+' \
        --test_blank 10000 \
        --train_max_samples '+str((control_i+1)*10000)+' \
        --learning_rate '+str(0.001* 0.9**control_i)+'  \
        --batchsize 8  \
        --out_baseline \'out_baseline_101_test\' \
        --num_point 256 \
        --gt_num_point 4096 \
        --pack_path \'../../data/Sketchfab2/packed_data/version_2\'  \
        --over_fitting_id 0 \
        --if_over_fitting_this_time 0 \
        --if_only_test 1 \
        --if_only_test_max_num 14 \
        --network_name \'Net_conpu_v7\'  \
        --emb_dims 512 \
        --neighbor_k 10 \
        --mlp_fitting_str \'256 128 64\' \
        --pretrained \'./pre_trained/v3.pt\' \
        --if_fix_sample 0 \
        --if_use_siren 0 \
        --feature_unfolding_nei_num 4 \
        '+loss_weight)
    
    
    
