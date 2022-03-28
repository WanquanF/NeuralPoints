import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import grad
import math
import numpy as np
import torch.nn.init as init
import struct
import os
import sys
import glob
import h5py
import copy
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../code')
import igl
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch
import torch_tensor_functions
import mesh_operations
from pointnet2 import pointnet2_utils as pn2_utils
#from chamfer_distance import ChamferDistance
#chamfer_dist = ChamferDistance()


######## TODO: START PART: FUNCTIONS ABOUT DGCNN. IT IS USED AS THE FEATURE EXTRACTOR IN OUR FRAMEWORK. ########
#### The DGCNN network ####
class DGCNN_multi_knn_c5(nn.Module):
    def __init__(self, emb_dims=512, args=None):
        super(DGCNN_multi_knn_c5, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv1.weight, gain=1.0)
        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv2.weight, gain=1.0)
        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv3.weight, gain=1.0)
        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv4.weight, gain=1.0)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv5.weight, gain=1.0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)
    def forward(self, x, if_relu_atlast = False):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x) # This sub model get the graph-based features for the following 2D convs
        # The x is similar with 2D image
        if self.args.if_bn == True: x = F.relu(self.bn1(self.conv1(x)))
        else: x = F.relu(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1)
        if self.args.if_bn == True: x = F.relu(self.bn2(self.conv2(x))) 
        else: x = F.relu(self.conv2(x))
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2)
        if self.args.if_bn == True: x = F.relu(self.bn3(self.conv3(x))) 
        else: x = F.relu(self.conv3(x))
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3)
        if self.args.if_bn == True: x = F.relu(self.bn4(self.conv4(x))) 
        else: x = F.relu(self.conv4(x))
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1).unsqueeze(3)
        if if_relu_atlast == False:
            return torch.tanh(self.conv5(x)).view(batch_size, -1, num_points)
        x = F.relu(self.conv5(x)).view(batch_size, -1, num_points)
        return x
#### The knn function used in graph_feature ####
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
#### The edge_feature used in DGCNN ####
def get_graph_feature(x, k=4):
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature
######## TODO: END PART: FUNCTIONS ABOUT DGCNN. IT IS USED AS THE FEATURE EXTRACTOR IN OUR FRAMEWORK. ########

######## TODO: START PART: NEURAL IMPLICIT FUNCTION, MLP with ReLU. ########
#### Construct the neural implicit function. ####
class MLPNet_relu(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn = True):
        super().__init__()
        list_layers = mlp_layers_relu(nch_input, nch_layers, b_shared, bn_momentum, dropout, if_bn)
        self.layers = torch.nn.Sequential(*list_layers)
    def forward(self, inp):
        out = self.layers(inp)
        return out
#### Construct the mlp_layers of the neural implicit function. ####
def mlp_layers_relu(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn=True):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
            init.xavier_normal_(weights.weight, gain=1.0)
            # if i==0: init.uniform_(weights.weight, a=-(6/last)**0.5*30, b=(6/last)**0.5*30)
            # else: init.uniform_(weights.weight, a=-(6/last)**0.5, b=(6/last)**0.5)
        else:
            weights = torch.nn.Linear(last, outp)
            init.xavier_normal_(weights.weight, gain=1.0)
        layers.append(weights)
        if if_bn==True:
            layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        # layers.append(Sine())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers
######## TODO: END PART: NEURAL IMPLICIT FUNCTION, MLP with ReLU. ########


######## TODO: START PART: NEURAL IMPLICIT FUNCTION, MLP with SIREN. ########
#### Construct the neural implicit function. ####
class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn = True):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout, if_bn)
        self.layers = torch.nn.Sequential(*list_layers)
    def forward(self, inp):
        out = self.layers(inp)
        return out
#### Construct the mlp_layers of the neural implicit function. ####
def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn=True):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
            #init.xavier_normal_(weights.weight, gain=1.0)
            if i==0: init.uniform_(weights.weight, a=-(6/last)**0.5*30, b=(6/last)**0.5*30)
            else: init.uniform_(weights.weight, a=-(6/last)**0.5, b=(6/last)**0.5)
        else:
            weights = torch.nn.Linear(last, outp)
            init.xavier_normal_(weights.weight, gain=1.0)
        layers.append(weights)
        if if_bn==True:
            layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        #layers.append(torch.nn.ReLU())
        layers.append(Sine())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers
#### The nn.Moudle Sine, as the activation function, used in the nearal implicit function. ####
class Sine(nn.Module):
    def __init(self):
        super().__init__()
    def forward(self, input):
        return torch.sin(input)
######## TODO: END PART: NEURAL IMPLICIT FUNCTION, MLP with SIREN. ########


######## TODO: START PART: OUR OWN NETWORK ########
#### The main network ####
class Net_conpu_v7(nn.Module):
    def __init__(self, args):
        super(Net_conpu_v7, self).__init__()
        # basic settings
        self.args = args # the args
        self.emb_dims = args.emb_dims # the dim of the embedded feture
        self.up_ratio = -1 # the upsampling factor
        self.over_sampling_up_ratio = -1 # the scale of over-sampling
        self.mlp_fitting_str = self.args.mlp_fitting_str
        self.mlp_fitting = convert_str_2_list(self.mlp_fitting_str) # the channels of the layers in the MLP
        ######################## START PART : LAYERS #########################
        ## 1. The point-wise feature extraction, DGCNN.
        self.emb_nn_sparse = DGCNN_multi_knn_c5(emb_dims=self.emb_dims, args=self.args) # the DGCNN backbone, which is shared by all the local parts
        ## 2. The Neural Field, MLP.
        if self.args.if_use_siren==True: self.fitting_mlp = MLPNet(2*self.emb_dims+(self.args.pe_out_L*4+2), self.mlp_fitting, b_shared=True, if_bn =False).layers
        else: self.fitting_mlp = MLPNet_relu(2*self.emb_dims+(self.args.pe_out_L*4+2), self.mlp_fitting, b_shared=True, if_bn =False).layers   
        self.reconstruct_out_p = torch.nn.Conv1d(self.mlp_fitting[-1], 3, 1)
        init.xavier_normal_(self.reconstruct_out_p.weight, gain=1.0)
        self.convert_feature_to_point_2to3 = torch.nn.Sequential(self.fitting_mlp, self.reconstruct_out_p)   # the Neural Field Fuction (MLP) 
        ######################## END PART : LAYERS #########################
    
    def forward(self, points_sparse):
        # The input [points_sparse] should be in shape (thisbatchsize, self.args.num_point, 3)
        thisbatchsize = points_sparse.size()[0]
        neighbour_indexes_ = torch_tensor_functions.get_neighbor_index(points_sparse, self.args.feature_unfolding_nei_num)   # thisbatchsize, self.args.num_point, neighbor_num
        ######### How to set the uv_sampling_coors ?
        #### We DON'T NEED TO give the network the uv_sampling_coors, it would be computed automatically. And the up_ratio should be training_up_ratio/testing_up_ratio, depending on self.training.
        uv_sampling_coors=torch.ones([1]).float().cuda()
        if self.training == True : self.up_ratio = self.args.training_up_ratio
        else : self.up_ratio = self.args.testing_up_ratio
        self.over_sampling_up_ratio = int(self.up_ratio * self.args.over_sampling_scale)
        if self.args.if_fix_sample == True: uv_sampling_coors = fix_sample(thisbatchsize, self.args.num_point, self.over_sampling_up_ratio)
        else: 
            uv_sampling_coors_1 = uniform_random_sample(thisbatchsize, self.args.num_point, self.over_sampling_up_ratio-4)
            uv_sampling_coors_2 = fix_sample(thisbatchsize, self.args.num_point, 4)
            uv_sampling_coors_ = torch.cat((uv_sampling_coors_1, uv_sampling_coors_2), dim=2) 
            uv_sampling_coors = copy.deepcopy(uv_sampling_coors_.detach())
        uv_sampling_coors = uv_sampling_coors.detach().contiguous()   # thisbatchsize, self.args.num_point, self.over_sampling_up_ratio, 2
        uv_sampling_coors.requires_grad=True
        ######### Set the uv_sampling_coors, Done.

        # compute the point-wise feature, updated with local pooling
        neighbour_indexes_feature_extract = torch_tensor_functions.get_neighbor_index(points_sparse, self.args.neighbor_k)   # bs, vertice_num, neighbor_num
        points_in_local_patch_form = torch_tensor_functions.indexing_by_id(points_sparse,neighbour_indexes_feature_extract)
        points_in_local_patch_form = points_in_local_patch_form - points_sparse.view(thisbatchsize,self.args.num_point,1,3)
        points_in_local_patch_form = points_in_local_patch_form.view(thisbatchsize*self.args.num_point, self.args.neighbor_k, 3)
        sparse_embedding = self.emb_nn_sparse(points_in_local_patch_form.transpose(1,2))  # B*num_point, self.emb_dims, self.neighbor_k
        sparse_embedding = torch.max(sparse_embedding,dim=-1,keepdim=False)[0].view(thisbatchsize,self.args.num_point,-1).permute(0,2,1)
        local_features_pooling = torch_tensor_functions.indexing_neighbor(sparse_embedding.transpose(1,2), neighbour_indexes_).permute(0,3,2,1)
        local_features_pooling = torch.max(local_features_pooling, dim=2, keepdim=False)[0]
        sparse_embedding = torch.cat((sparse_embedding,local_features_pooling),dim=1)
        sparse_embedding = sparse_embedding.permute(0,2,1)  # thisbatchsize, self.args.num_point, self.emb_dims*2
        

        # get the uv_sampling_coors_id_in_sparse
        uv_sampling_coors_id_in_sparse = torch.arange(self.args.num_point).view(1,-1,1).long()
        uv_sampling_coors_id_in_sparse = uv_sampling_coors_id_in_sparse.expand(thisbatchsize,-1,self.over_sampling_up_ratio).reshape(thisbatchsize,-1,1)
        upsampled_p, upsampled_np = self.convert_uv_to_xyzn(uv_sampling_coors.reshape(thisbatchsize,-1,2), uv_sampling_coors_id_in_sparse, sparse_embedding, points_sparse) # thisbatchsize, self.args.num_point*self.over_sampling_up_ratio, 3
        

        upsampled_p_fps_id = pn2_utils.furthest_point_sample(upsampled_p.contiguous(), self.up_ratio*self.args.num_point)
        querying_points_3d = pn2_utils.gather_operation(upsampled_p.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
        querying_points_n_3d = pn2_utils.gather_operation(upsampled_np.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
        querying_points_3d = querying_points_3d.permute(0,2,1).contiguous()
        querying_points_n_3d = querying_points_n_3d.permute(0,2,1).contiguous()

        # Get the final upsampled points from the 3D querying points
        glued_points, glued_normals = self.project_3d_query_point_to_patches(querying_points_3d, querying_points_n_3d, points_sparse, upsampled_p, upsampled_np)

        

        

        # Notice that the returned uv_sampling_coors is not differentiable, just used to compute the loss.
        return upsampled_p, upsampled_np, uv_sampling_coors, querying_points_3d, querying_points_n_3d, glued_points, glued_normals

    def project_3d_query_point_to_patches(self, querying_points_3d, querying_points_n_3d, points_sparse, upsampled_p, upsampled_np):
        # All3dQueryPointNum = self.args.num_point * self.up_ratio
        # All2dQueryPointNum = self.args.num_point * self.over_sampling_up_ratio
        # querying_points_3d     | should be in size : thisbatchsize, All3dQueryPointNum, 3
        # querying_points_n_3d   | should be in size : thisbatchsize, All3dQueryPointNum, 3
        # points_sparse          | should be in size : thisbatchsize, self.args.num_point, 3
        # upsampled_p            | should be in size : thisbatchsize, All2dQueryPointNum, 3
        # upsampled_np           | should be in size : thisbatchsize, All2dQueryPointNum, 3
        
        thisbatchsize = querying_points_3d.size()[0]
        All3dQueryPointNum = querying_points_3d.size()[1]
        All2dQueryPointNum = upsampled_p.size()[1]
        #### Distribute the 3d querying points to the center points. ####
        # 1. compute the distance map bwtween 3d querying points and center points. 
        querying_points_3d__center_p__dismap = torch_tensor_functions.compute_sqrdis_map(querying_points_3d, points_sparse)    # thisbatchsize, All3dQueryPointNum, self.args.num_point
        # 2. find the neighbour ID from the 3d querying points to the center points.
        querying_points_3d_distribute_to_centers_nei_id = torch.topk(querying_points_3d__center_p__dismap, k=self.args.glue_neighbor, dim=2, largest=False)[1] # thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor
        # 3. find the nearest distance from the 3d querying points to the center points.
        querying_points_3d_distribute_to_centers_nei_dis = torch.topk(querying_points_3d__center_p__dismap, k=self.args.glue_neighbor, dim=2, largest=False)[0].detach() # thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor 
        # 4. find the nearest points coordinates from the 3d querying points to the center points.
        querying_points_3d_distribute_to_centers_nei_coor = torch_tensor_functions.indexing_by_id(points_sparse, querying_points_3d_distribute_to_centers_nei_id) # thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor, 3 
        # 5. compute the weight of the 3d querying points distributed to their neighbour center points. 
        Alpha_glue = 1.0/torch.mean(querying_points_3d_distribute_to_centers_nei_dis) 
        querying_points_3d_distribute_to_centers_nei_weight = torch.exp( -1 * Alpha_glue * querying_points_3d_distribute_to_centers_nei_dis )
        querying_points_3d_distribute_to_centers_nei_weight = querying_points_3d_distribute_to_centers_nei_weight / (torch.sum(querying_points_3d_distribute_to_centers_nei_weight,dim=-1,keepdim=True)+0.0000001)  # thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor. The last dim should sum up to 1.

        #### Project the 3d querying points to their neighbour patches. ####
        #### In this part, we can get a (thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor, 3)-shaped tensor, which should be multiplied with the weight above.
        # For each 3d querying point's each neighbour patch , find the projection points in the patch.  
        querying_points_3d_ = querying_points_3d.view(thisbatchsize, All3dQueryPointNum, 1, 3)
        querying_points_3d_ = querying_points_3d_.expand(-1, -1, self.args.glue_neighbor, -1)
        querying_points_3d_ = querying_points_3d_.reshape(thisbatchsize, All3dQueryPointNum*self.args.glue_neighbor, 1, 3)  

        upsampled_p_ = upsampled_p.view(thisbatchsize, self.args.num_point, -1, 3)
        upsampled_np_ = upsampled_np.view(thisbatchsize, self.args.num_point, -1, 3)
        up_ratio_here = upsampled_p_.size()[2]
        upsampled_p_ = upsampled_p_.reshape(thisbatchsize, self.args.num_point, -1)
        upsampled_np_ = upsampled_np_.reshape(thisbatchsize, self.args.num_point, -1)
        all_queried_patches = torch_tensor_functions.indexing_by_id(upsampled_p_, querying_points_3d_distribute_to_centers_nei_id)
        all_queried_patchesn = torch_tensor_functions.indexing_by_id(upsampled_np_, querying_points_3d_distribute_to_centers_nei_id)
        all_queried_patches = all_queried_patches.view(thisbatchsize, All3dQueryPointNum*self.args.glue_neighbor, up_ratio_here, 3)
        all_queried_patchesn = all_queried_patchesn.view(thisbatchsize, All3dQueryPointNum*self.args.glue_neighbor, up_ratio_here, 3)
        

        dis_from_3d_querying_points_to_its_corresponidng_patch = querying_points_3d_ - all_queried_patches
        dis_from_3d_querying_points_to_its_corresponidng_patch = torch.sum( torch.mul(dis_from_3d_querying_points_to_its_corresponidng_patch, dis_from_3d_querying_points_to_its_corresponidng_patch) , dim = -1, keepdim = False)
        nei_id_from_3d_querying_points_to_its_corresponidng_patch = torch.topk(dis_from_3d_querying_points_to_its_corresponidng_patch, dim =-1, k=self.args.proj_neighbor,largest=False)[1].reshape(thisbatchsize*All3dQueryPointNum*self.args.glue_neighbor, 1, self.args.proj_neighbor)
        nei_dis_from_3d_querying_points_to_its_corresponidng_patch = torch.topk(dis_from_3d_querying_points_to_its_corresponidng_patch, dim =-1, k=self.args.proj_neighbor,largest=False)[0].reshape(thisbatchsize*All3dQueryPointNum*self.args.glue_neighbor, 1, self.args.proj_neighbor)
        all_queried_patches_ = all_queried_patches.view(thisbatchsize*All3dQueryPointNum*self.args.glue_neighbor, up_ratio_here, 3)
        all_queried_patchesn_ = all_queried_patchesn.view(thisbatchsize*All3dQueryPointNum*self.args.glue_neighbor, up_ratio_here, 3)
        nei_coor_from_3d_querying_points_to_its_corresponidng_patch = torch_tensor_functions.indexing_by_id(all_queried_patches_, nei_id_from_3d_querying_points_to_its_corresponidng_patch)
        nei_ncoor_from_3d_querying_points_to_its_corresponidng_patch = torch_tensor_functions.indexing_by_id(all_queried_patchesn_, nei_id_from_3d_querying_points_to_its_corresponidng_patch)
        nei_weight_from_3d_querying_points_to_its_corresponidng_patch = torch.exp( -1000 * nei_dis_from_3d_querying_points_to_its_corresponidng_patch)
        nei_weight_from_3d_querying_points_to_its_corresponidng_patch = nei_weight_from_3d_querying_points_to_its_corresponidng_patch / (torch.sum(nei_weight_from_3d_querying_points_to_its_corresponidng_patch, dim=-1, keepdim=True) +0.0000001 )
        nei_weight_from_3d_querying_points_to_its_corresponidng_patch = nei_weight_from_3d_querying_points_to_its_corresponidng_patch.view(thisbatchsize*All3dQueryPointNum*self.args.glue_neighbor, 1, self.args.proj_neighbor,1)
        projected_points = torch.sum(nei_weight_from_3d_querying_points_to_its_corresponidng_patch * nei_coor_from_3d_querying_points_to_its_corresponidng_patch, dim =2, keepdim=False )
        projected_pointsn = torch.sum(nei_weight_from_3d_querying_points_to_its_corresponidng_patch * nei_ncoor_from_3d_querying_points_to_its_corresponidng_patch, dim =2, keepdim=False )
        projected_points = projected_points.view(thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor, 3)  # thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor, 3
        projected_pointsn = projected_pointsn.view(thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor, 3)  # thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor, 3
        
        projected_pointsn_sign = projected_pointsn.detach()
        projected_pointsn_sign_ref = projected_pointsn_sign[:,:,0:1,:].expand(-1,-1,self.args.glue_neighbor,-1)
        projected_pointsn_sign = torch.sum(torch.mul(projected_pointsn_sign, projected_pointsn_sign_ref) ,dim=-1, keepdim=True ).expand(-1,-1,-1,3)
        projected_pointsn_sign = torch.sign(projected_pointsn_sign+0.1)
        
        # correct the direction of the normals.
        projected_pointsn = torch.mul(projected_pointsn, projected_pointsn_sign)
        #### Glue the 3d upsampled points. ####
        glued_points = torch.sum( projected_points * querying_points_3d_distribute_to_centers_nei_weight.view(thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor, 1), dim = 2 , keepdim=False)
        glued_normals = torch.sum( projected_pointsn * querying_points_3d_distribute_to_centers_nei_weight.view(thisbatchsize, All3dQueryPointNum, self.args.glue_neighbor, 1), dim = 2 , keepdim=False)
        return glued_points, glued_normals
    
    
    def convert_uv_to_xyzn(self, uv_coor, uv_coor_idx_in_sparse, sparse_embedding, points_sparse):
        # uv_coor                | should be in size : thisbatchsize, All2dQueryPointNum, 2
        # uv_coor_idx_in_sparse  | should be in size : thisbatchsize, All2dQueryPointNum, 1
        # sparse_embedding       | should be in size : thisbatchsize, sparse_point_num, embedding_dim
        # points_sparse          | should be in size : thisbatchsize, sparse_point_num, 3
        thisbatchsize = uv_coor.size()[0]
        All2dQueryPointNum = uv_coor.size()[1]
        converted2to3_p = self.convert_uv_to_xyz(uv_coor, uv_coor_idx_in_sparse, sparse_embedding, points_sparse)
        
        converted2to3_p_x = converted2to3_p[:,:,0:1].reshape(thisbatchsize*All2dQueryPointNum,1)
        grad_x_uv = cal_grad(uv_coor, converted2to3_p_x).reshape(thisbatchsize*All2dQueryPointNum,2,1)
        converted2to3_p_y = converted2to3_p[:,:,1:2].reshape(thisbatchsize*All2dQueryPointNum,1)
        grad_y_uv = cal_grad(uv_coor, converted2to3_p_y).reshape(thisbatchsize*All2dQueryPointNum,2,1)
        converted2to3_p_z = converted2to3_p[:,:,2:3].reshape(thisbatchsize*All2dQueryPointNum,1)
        grad_z_uv = cal_grad(uv_coor, converted2to3_p_z).reshape(thisbatchsize*All2dQueryPointNum,2,1)

        grad_uv = torch.cat((grad_x_uv, grad_y_uv, grad_z_uv), dim=-1)
        grad_u = grad_uv[:,0:1,:].view(-1,3)
        grad_v = grad_uv[:,1:2,:].view(-1,3)

        converted2to3_np = torch.cross(grad_u.reshape(-1,3), grad_v.reshape(-1,3))
        converted2to3_np_norm = torch.norm(converted2to3_np, dim=1).view(-1,1) +0.000001
        converted2to3_np = converted2to3_np/converted2to3_np_norm
        converted2to3_np = converted2to3_np.view(thisbatchsize,-1,3)

        return converted2to3_p, converted2to3_np


    def convert_uv_to_xyz(self, uv_coor, uv_coor_idx_in_sparse, sparse_embedding, points_sparse):
        # uv_coor                | should be in size : thisbatchsize, All2dQueryPointNum, 2
        # uv_coor_idx_in_sparse  | should be in size : thisbatchsize, All2dQueryPointNum, 1
        # sparse_embedding       | should be in size : thisbatchsize, sparse_point_num, embedding_dim
        # points_sparse          | should be in size : thisbatchsize, sparse_point_num, 3
        thisbatchsize = uv_coor.size()[0]
        All2dQueryPointNum = uv_coor.size()[1]
        coding_dim = 4*self.args.pe_out_L + 2
        uv_encoded = position_encoding(uv_coor.reshape(-1,2).contiguous(), self.args.pe_out_L).view(thisbatchsize, All2dQueryPointNum, coding_dim).permute(0,2,1) # bs, coding_dim, All2dQueryPointNum
        indexed_sparse_feature = torch_tensor_functions.indexing_by_id(sparse_embedding, uv_coor_idx_in_sparse)  # bs, All2dQueryPointNum, 1, embedding_num 
        indexed_sparse_feature = indexed_sparse_feature.view(thisbatchsize, All2dQueryPointNum, -1).transpose(2,1)  # bs, embedding_num, All2dQueryPointNum
        coding_with_feature = torch.cat((indexed_sparse_feature, uv_encoded), dim=1)
        out_p = self.convert_feature_to_point_2to3(coding_with_feature).view(thisbatchsize, -1, All2dQueryPointNum).permute(0,2,1)
        indexed_center_points = torch_tensor_functions.indexing_by_id(points_sparse, uv_coor_idx_in_sparse).view(thisbatchsize, All2dQueryPointNum, 3)
        out_p = out_p + indexed_center_points
        return out_p
    
    def convert_xyz_to_uv(self, xyz_coor, xyz_coor_idx_in_sparse, sparse_embedding, points_sparse):
        # xyz_coor               | should be in size : thisbatchsize, All2dQueryPointNum, 3
        # uv_coor_idx_in_sparse  | should be in size : thisbatchsize, All2dQueryPointNum, 1
        # sparse_embedding       | should be in size : thisbatchsize, sparse_point_num, embedding_dim
        # points_sparse          | should be in size : thisbatchsize, sparse_point_num, 3
        # return : out_uv        | should be in size : thisbatchsize, All2dQueryPointNum, 2
        thisbatchsize = xyz_coor.size()[0]
        All2dQueryPointNum = xyz_coor.size()[1]
        coding_dim = 6*self.args.pe_out_L + 3
        indexed_center_points = torch_tensor_functions.indexing_by_id(points_sparse, xyz_coor_idx_in_sparse).view(thisbatchsize, All2dQueryPointNum, 3)
        xyz_coor_remove_center = xyz_coor - indexed_center_points
        xyz_encoded = position_encoding(xyz_coor.reshape(-1,3), self.args.pe_out_L).view(thisbatchsize, All2dQueryPointNum, coding_dim).permute(0,2,1) # bs, coding_dim, All2dQueryPointNum
        indexed_sparse_feature = torch_tensor_functions.indexing_by_id(sparse_embedding, xyz_coor_idx_in_sparse)  # bs, All2dQueryPointNum, 1, embedding_num 
        indexed_sparse_feature = indexed_sparse_feature.view(thisbatchsize, All2dQueryPointNum, -1).transpose(2,1)  # bs, embedding_num, All2dQueryPointNum
        coding_with_feature = torch.cat((xyz_encoded, indexed_sparse_feature), dim = 1)
        out_uv = self.convert_feature_to_point_3to2(coding_with_feature).view(thisbatchsize, -1, All2dQueryPointNum).permute(0,2,1)
        return out_uv
        

#### Convert a string to num_list ####      
def convert_str_2_list(str_):
    words = str_.split(' ')
    trt = [int(x) for x in words]
    return trt
#### Compute the position code for uv or xyz. ####
def position_encoding(input_uv, pe_out_L):
    ## The input_uv should be with shape (-1, X)
    ## The returned tensor should be with shape (-1, X+2*X*L)
    ## X = 2/3 if the input is uv/xyz.
    trt = input_uv
    for i in range(pe_out_L):
        trt = torch.cat((trt, torch.sin(input_uv*(2**i)*(3.14159265))) , dim=-1 )
        trt = torch.cat((trt, torch.cos(input_uv*(2**i)*(3.14159265))) , dim=-1 )
    return trt
#### Sample uv by a fixed manner. #### 
def fix_sample(thisbatchsize, num_point, up_ratio, if_random=False):
    if if_random==True: 
        print('Random sampling mode is not supported right now.')
        exit()
    if up_ratio == 4:
        one_point_fixed = [ [ [0,0] for i in range(2)] for j in range(2) ]
        for i in range(2):
            for j in range(2):
                one_point_fixed[i][j][0] = (i/1) *2 -1
                one_point_fixed[i][j][1] = (j/1) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(4,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    if up_ratio == 9:
        one_point_fixed = [ [ [0,0] for i in range(3)] for j in range(3) ]
        for i in range(3):
            for j in range(3):
                one_point_fixed[i][j][0] = (i/2) *2 -1
                one_point_fixed[i][j][1] = (j/2) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(9,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    if up_ratio == 16:
        one_point_fixed = [ [ [0,0] for i in range(4)] for j in range(4) ]
        for i in range(4):
            for j in range(4):
                one_point_fixed[i][j][0] = (i/3) *2 -1
                one_point_fixed[i][j][1] = (j/3) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(16,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    if up_ratio == 64:
        one_point_fixed = [ [ [0,0] for i in range(8)] for j in range(8) ]
        for i in range(8):
            for j in range(8):
                one_point_fixed[i][j][0] = (i/7) *2 -1
                one_point_fixed[i][j][1] = (j/7) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(64,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    if up_ratio == 1024:
        one_point_fixed = [ [ [0,0] for i in range(32)] for j in range(32) ]
        for i in range(32):
            for j in range(32):
                one_point_fixed[i][j][0] = (i/31) *2 -1
                one_point_fixed[i][j][1] = (j/31) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(1024,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    else:
        print('This up_ratio ('+str(up_ratio)+') is not supported now. You can try the random mode!')
        exit()
#### Sample uv uniformly in (-1,1). #### 
def uniform_random_sample(thisbatchsize, num_point, up_ratio):
    # return : randomly and uniformly sampled uv_coors   |   Its shape should be : thisbatchsize, num_point, up_ratio, 2
    res_ = torch.rand(thisbatchsize*num_point, 4*up_ratio, 3)*2-1
    res_ = res_.cuda()
    res_[:,:,2:]*=0
    furthest_point_index = pn2_utils.furthest_point_sample(res_,up_ratio)
    uniform_res_ = pn2_utils.gather_operation(res_.permute(0, 2, 1).contiguous(), furthest_point_index)
    uniform_res_ = uniform_res_.permute(0,2,1).contiguous()
    uniform_res_ = uniform_res_[:,:,:2].view(thisbatchsize, num_point, up_ratio, 2)
    return uniform_res_
#### Compute the grad ####
def cal_grad(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad = False, device = outputs.device)
    points_grad = grad(
        outputs = outputs,
        inputs = inputs,
        grad_outputs = d_points,
        create_graph = True,
        retain_graph = True,
        only_inputs = True)[0]
    return points_grad



######## TODO: END PART: OUR OWN NETWORK ########
