import argparse
import os
from configparser import SafeConfigParser

def parse_args():
    # argparse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--wq_test', type=int, default=0,help='if test by wq method')
    
    parser.add_argument('--device_id',help='Specify the index of the cuda device, e.g. 0, 1 ,2',default=0, type=int)
    parser.add_argument('--num_point', type=int, default=256,help='Point Number')
    parser.add_argument('--gt_num_point', type=int, default=4096,help='Point Number of GT points')
    parser.add_argument('--training_up_ratio', type=int, default=4,help='The Upsampling Ratio during training') 
    parser.add_argument('--testing_up_ratio', type=int, default=4, help='The Upsampling Ratio during testing')  
    parser.add_argument('--over_sampling_scale', type=float, default=1.5, help='The scale for over-sampling')
    parser.add_argument('--limited_testing_model_num', type=int, default=-1, help='The max allowed num of tested model')
    parser.add_argument('--emb_dims', type=int, default=8192, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--testing_anchor_num', type=int, default=114, metavar='N',help='The number of patches on the testing models')
    parser.add_argument('--pe_out_L', type=int, default=5, metavar='N',help='The parameter L in the position code')
    parser.add_argument('--feature_unfolding_nei_num', type=int, default=4, metavar='N',help='The number of neighbour points used while feature unfolding')
    parser.add_argument('--repulsion_nei_num', type=int, default=5, metavar='N',help='The number of neighbour points used in repulsion loss')

    # for phase train
    parser.add_argument('--batchsize', type=int, default=8, help='Batch Size during training')
    parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run')
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--reg_normal1', type=float, default=0.1)
    parser.add_argument('--reg_normal2', type=float, default=0.1)
    parser.add_argument('--jitter_sigma', type=float, default=0.01)
    parser.add_argument('--jitter_max', type=float, default=0.03)
    parser.add_argument('--if_bn', type=int, default=0, help='If using batch normalization')
    parser.add_argument('--neighbor_k', type=int, default=5, help='The number of neighbour points used in DGCNN')
    # parser.add_argument('--mlpchanels_uv_encoder_str', type=str, default='None', metavar='None',help='mlp layers of the uv position encoding (default: None)')
    parser.add_argument('--mlp_fitting_str', type=str, default='None', metavar='None',help='mlp layers of the part surface fitting (default: None)')
    parser.add_argument('--mlp_projecting_str', type=str, default='None', metavar='None',help='mlp layers of the part surface projecting (default: None)')
    # parser.add_argument('--mlp_refining_str', type=str, default='None', metavar='None',help='mlp layers of the point-wise refining (default: None)')
    # parser.add_argument('--if_refine_by_net', type=int, default=0, help='if to use the refining module in the network')
    parser.add_argument('--glue_neighbor', type=int, default=4, help='The number of neighbour points used in glue process')
    parser.add_argument('--proj_neighbor', type=int, default=4, help='The number of neighbour points used in projection process')

    # control the training
    parser.add_argument('--last_sample_id',help='the id in the last saved trained model',default=0, type=int)    
    parser.add_argument('--train_max_samples',help='the max number of samples used in the training',default=500000, type=int)
    parser.add_argument('--test_blank',help='how often the testing process is performed',default=100, type=int)
    parser.add_argument('--visualization_while_testing', default=1, type=int, metavar='visual', help='1 if visualize; 0 if not')

    # the trained results
    parser.add_argument('--pack_path', type=str, default='None', metavar='None',help='the path of packed_data (default: None)')
    parser.add_argument('--out_baseline',help='the file of the baseline training results',default='./output_baseline', type=str)  

    #for phase test
    parser.add_argument('--pretrained', default='', help='Model stored')
    parser.add_argument('--eval_xyz', default='test_5000', help='Folder to evaluate')
    parser.add_argument('--num_shape_point', type=int, default=5000,help='Point Number per shape')
    parser.add_argument('--patch_num_ratio', type=int, default=3,help='Number of points covered by patch')

    #loss terms weights
    parser.add_argument('--weight_cd', type=float, default=-1)
    parser.add_argument('--weight_refined_cd', type=float, default=-1)
    parser.add_argument('--weight_repulsion', type=float, default=-1)
    parser.add_argument('--weight_pre', type=float, default=-1)
    parser.add_argument('--weight_center', type=float, default=-1)
    parser.add_argument('--weight_exclude', type=float, default=-1)
    parser.add_argument('--weight_uniform', type=float, default=-1)
    parser.add_argument('--weight_reg', type=float, default=-1)
    parser.add_argument('--weight_arap', type=float, default=-1)
    parser.add_argument('--weight_overlap', type=float, default=-1)
    parser.add_argument('--weight_proj', type=float, default=-1)
    parser.add_argument('--weight_normal', type=float, default=-1)
    parser.add_argument('--weight_cycle', type=float, default=-1)
    parser.add_argument('--weight_ndirection', type=float, default=-1)


    parser.add_argument('--weight_decay',default=0.00005, type=float)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--clip', type=float, default=1.0)

    # control the using mode
    parser.add_argument('--over_fitting_id', type=int, default=0, help='The id that you want to overfit')
    parser.add_argument('--if_over_fitting_this_time', type=int, default=0, help='whether you want to overfit, default is False')
    parser.add_argument('--if_only_test', type=int, default=0, help='whether you only want to test, default is False')
    parser.add_argument('--if_only_test_max_num', type=int, default=3, help='the max number of models that you want to test on')
    parser.add_argument('--network_name', type=str, default='Net_conpu_v1', help='the name of the network that you would like to use')
    parser.add_argument('--if_fix_sample', type=int, default=0, help='whether to use fix sampling')
    parser.add_argument('--if_use_siren', type=int, default=0, help='whether to use siren activation function')


    
    
    '''
    #basic settings
    
                                     
    # arguments for training process
    

    parser.add_argument('--patch_num', default=10, type=int,
                        metavar='pn', help='number of patches')
    parser.add_argument('--point_num', default=8192, type=int,
                        metavar='pn', help='number of patches')
    parser.add_argument('--dim_k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')
    parser.add_argument('--learn_delta', dest='learn_delta', action='store_true',
                        help='flag for training step size delta')
    parser.add_argument('--neighbour_num', default=4, type=int,
                        metavar='nn', help='neighbour_num of weight smoothing term')
    
    
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=1, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
                        
    # PointNet settings
    parser.add_argument('--radius', type=float, default=0.3, help='Neighborhood radius for computing pointnet features')
    parser.add_argument('--num_neighbors', type=int, default=64, metavar='N', help='Max num of neighbors to use')
    # RPMNet settings
    parser.add_argument('--features', type=str, choices=['ppf', 'dxyz', 'xyz'], default=['ppf', 'dxyz', 'xyz'],
                        nargs='+', help='Which features to use. Default: all')
    parser.add_argument('--feat_dim', type=int, default=96,
                        help='Feature dimension (to compute distances on). Other numbers will be scaled accordingly')
    parser.add_argument('--no_slack', action='store_true', help='If set, will not have a slack column.')
    parser.add_argument('--num_sk_iter', type=int, default=5,
                        help='Number of inner iterations used in sinkhorn normalization')
    parser.add_argument('--num_reg_iter', type=int, default=5,
                        help='Number of outer iterations used for registration (only during inference)')
    parser.add_argument('--loss_type', type=str, choices=['mse', 'mae'], default='mae',
                        help='Loss to be optimized')
    parser.add_argument('--wt_inliers', type=float, default=1e-2, help='Weight to encourage inliers')
                        
    parser.add_argument('--lambda_data', type=float, default=1.0, help='weight of depth loss')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='weight of regularization loss')
        
    parser.add_argument('--num_adja', type=int, default=8, help='number of nodes who affect a point')
    parser.add_argument('--max_num_edges', type=int, default=3000, help='number of edges')
    parser.add_argument('--max_num_nodes', type=int, default=400, help='number of nodes')
    parser.add_argument('--max_num_points', type=int, default=4096, help='number of points')
    '''             
    args = parser.parse_args()

    

    return args
