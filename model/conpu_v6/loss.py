import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
import sys
import time
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../code/')
import cv2 as cv
from PIL import Image
from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()
import glob
import trimesh
import random
import numpy as np
import math
from math import ceil
import time
import cv2
from PIL import Image
#from options import TestOptions
#import trimesh
import struct
import pickle
from pointnet2 import pointnet2_utils as pn2_utils


import torch_tensor_functions

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        
    def loss_on_cd(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        output = 0
        dist1, dist2 = chamfer_dist(deformation_p, p1)
        output += (torch.sum(dist1) + torch.sum(dist2))*0.5
        return output/thisbatchsize
    
    def loss_on_proj(self, p0, p1):
        # p0 : B, M, 3
        # p1 : B, N, 3
        thisbatchsize = p0.size()[0]
        output = 0
        dis_map = torch_tensor_functions.compute_sqrdis_map(p0, p1)   # B, M, N

        neighbour_id_01 = torch.topk(dis_map, k=5, dim=-1, largest= False)[1]
        neighbour_dis_01 = torch.topk(dis_map, k=5, dim=-1, largest= False)[0]
        neighbour_id_01 = neighbour_id_01[:,:,1:]
        neighbour_coor_01 = torch_tensor_functions.indexing_neighbor(p1, neighbour_id_01)
        neighbour_dis_01 = neighbour_dis_01[:,:,1:]
        neighbour_weight_01 = neighbour_dis_01.detach() * 1000
        neighbour_weight_01 = torch.exp(-1*neighbour_weight_01)
        neighbour_weight_01 = neighbour_weight_01/(torch.sum(neighbour_weight_01, dim=-1, keepdim=True)+0.00001)
        dis_01 = p0.view(thisbatchsize,-1,1,3) - neighbour_coor_01
        dis_01 = torch.sum(torch.mul(dis_01, dis_01), dim=-1, keepdim=False)
        pro_dis_01 = torch.mul(neighbour_weight_01, dis_01)
        output += 0.5 * torch.sum(pro_dis_01)

        neighbour_id_10 = torch.topk(dis_map, k=5, dim=1, largest= False)[1].transpose(2,1)
        neighbour_dis_10 = torch.topk(dis_map, k=5, dim=1, largest= False)[0].transpose(2,1)
        neighbour_id_10 = neighbour_id_10[:,:,1:]
        neighbour_coor_10 = torch_tensor_functions.indexing_neighbor(p0, neighbour_id_10)
        neighbour_dis_10 = neighbour_dis_10[:,:,1:]
        neighbour_weight_10 = neighbour_dis_10.detach() * 1000
        neighbour_weight_10 = torch.exp(-1*neighbour_weight_10)
        neighbour_weight_10 = neighbour_weight_10/(torch.sum(neighbour_weight_10, dim=-1, keepdim=True)+0.00001)
        dis_10 = p1.view(thisbatchsize,-1,1,3) - neighbour_coor_10
        dis_10 = torch.sum(torch.mul(dis_10, dis_10), dim=-1, keepdim=False)
        pro_dis_10 = torch.mul(neighbour_weight_10, dis_10)
        output += 0.5 * torch.sum(pro_dis_10)

        return output/thisbatchsize

    
    def loss_on_normal(self, p0, p1, n0, n1):
        # p0 : B, M, 3 ; n0 : B, M, 3
        # p1 : B, N, 3 ; n1 : B, N, 3
        thisbatchsize = p0.size()[0]
        output = 0
        dis_map = torch_tensor_functions.compute_sqrdis_map(p0, p1)   # B, M, N

        neighbour_id_01 = torch.topk(dis_map, k=5, dim=-1, largest= False)[1]
        neighbour_dis_01 = torch.topk(dis_map, k=5, dim=-1, largest= False)[0]
        neighbour_id_01 = neighbour_id_01[:,:,1:]
        neighbour_normal_01 = torch_tensor_functions.indexing_neighbor(n1, neighbour_id_01)
        neighbour_dis_01 = neighbour_dis_01[:,:,1:]
        neighbour_weight_01 = neighbour_dis_01.detach() * 1000
        neighbour_weight_01 = torch.exp(-1*neighbour_weight_01)
        neighbour_weight_01 = neighbour_weight_01/(torch.sum(neighbour_weight_01,   dim=-1, keepdim=True)+0.00001)
        dis_01 = n0.view(thisbatchsize,-1,1,3) - neighbour_normal_01
        dis_01 = torch.sum(torch.mul(dis_01, dis_01), dim=-1, keepdim=False)
        dis_01_ = n0.view(thisbatchsize,-1,1,3) + neighbour_normal_01
        dis_01_ = torch.sum(torch.mul(dis_01_, dis_01_), dim=-1, keepdim=False)
        bar_ = torch.sign(dis_01 - dis_01_)
        dis_01_min = torch.mul((bar_+1)*0.5, dis_01_) + torch.mul((1-bar_)*0.5, dis_01)
        dis_01_min = torch.mul(neighbour_weight_01, dis_01_min)
        output += 0.5 * torch.sum(dis_01_min)

        return output/thisbatchsize
    
    def loss_on_reg(self, gen_points_batch, train_points_sparse_batch):
        thisbatchsize = gen_points_batch.size()[0]
        output = 0
        up_ratio_here = gen_points_batch.size()[1]//train_points_sparse_batch.size()[1]
        gen_points_batch_ = gen_points_batch.view(thisbatchsize,-1,up_ratio_here,3)
        train_points_sparse_batch_ = train_points_sparse_batch.view(thisbatchsize,-1,1,3)
        dis = train_points_sparse_batch_ - gen_points_batch_
        squdis = torch.sum(torch.mul(dis,dis),dim=-1,keepdim=True)
        squdis_bar = squdis.detach()*0+0.04
        squdis_sign = torch.sign(squdis.detach() - squdis_bar)*0.5+1
        squdis = torch.mul(squdis,squdis_sign)
        output += torch.sum(squdis)
        return output/thisbatchsize
    
    def loss_on_arap(self, gen_points_batch, uv_sampling_coors):
        thisbatchsize = gen_points_batch.size()[0]
        output = 0
        gen_points_batch_ = gen_points_batch.reshape(thisbatchsize*self.args.num_point, -1 ,3)
        uv_sampling_coors_ = uv_sampling_coors.reshape(thisbatchsize*self.args.num_point, -1 ,2).detach()
        uv_sampling_coors_ = torch.cat((uv_sampling_coors_, uv_sampling_coors_[:,:,:1]),dim=-1)
        uv_sampling_coors_[:,:,2:]*=0
        neighbour_indexes = torch_tensor_functions.get_neighbor_index(uv_sampling_coors_, 4) 
        uv_neibour_points_ = torch_tensor_functions.indexing_neighbor(uv_sampling_coors_, neighbour_indexes)
        gen_neibour_points_ = torch_tensor_functions.indexing_neighbor(gen_points_batch_, neighbour_indexes)
        uv_dis = uv_neibour_points_ - uv_sampling_coors_.view(thisbatchsize*self.args.num_point, -1 ,1, 3)
        gen_dis = gen_neibour_points_ - gen_points_batch_.view(thisbatchsize*self.args.num_point, -1 ,1, 3)
        uv_squ_dis = torch.sqrt( torch.sum(torch.mul(uv_dis, uv_dis),dim=-1) + 0.00000001 )
        gen_squ_dis = torch.sqrt( torch.sum(torch.mul(gen_dis, gen_dis),dim=-1) + 0.00000001 )
        uv_sum_dis = torch.sum(uv_squ_dis)
        gen_sum_dis = torch.sum(gen_squ_dis).detach()
        uv_squ_dis *= gen_sum_dis / uv_sum_dis
        delta = uv_squ_dis - gen_squ_dis
        output += torch.sum(torch.mul(delta, delta))
        return output/thisbatchsize

    def loss_on_overlap(self, gen_points_batch, train_points_sparse_batch):
        thisbatchsize = gen_points_batch.size()[0]
        output = 0
        gen_points_batch_ = gen_points_batch.reshape(thisbatchsize*self.args.num_point, -1 ,3)
        neighbour_indexes = torch_tensor_functions.get_neighbor_index(train_points_sparse_batch, 6) 
        sparse_neibour_points_ = torch_tensor_functions.indexing_neighbor(train_points_sparse_batch, neighbour_indexes)
        sparse_neibour_points_ = sparse_neibour_points_.reshape(thisbatchsize*self.args.num_point, -1, 3)
        cross_dis = torch_tensor_functions.compute_sqrdis_map(sparse_neibour_points_, gen_points_batch_)
        dis = torch.sum(torch.min(cross_dis,dim=-1)[0])
        output += dis
        return output/thisbatchsize


    def loss_on_ndirection(self, gen_points_batch, uv_sampling_coors, gen_normals_batch):
        thisbatchsize = gen_points_batch.size()[0]
        output = 0
        # gen_points_batch_ = gen_points_batch.reshape(thisbatchsize*self.args.num_point, -1 ,3)
        gen_normals_batch_ = gen_normals_batch.reshape(thisbatchsize*self.args.num_point, -1 ,3)
        uv_sampling_coors_ = uv_sampling_coors.reshape(thisbatchsize*self.args.num_point, -1 ,2).detach()
        uv_sampling_coors_ = torch.cat((uv_sampling_coors_, uv_sampling_coors_[:,:,:1]),dim=-1)
        uv_sampling_coors_[:,:,2:]*=0
        neighbour_indexes = torch_tensor_functions.get_neighbor_index(uv_sampling_coors_, 4) 
        uv_neibour_points_ = torch_tensor_functions.indexing_neighbor(uv_sampling_coors_, neighbour_indexes)
        # gen_neibour_points_ = torch_tensor_functions.indexing_neighbor(gen_points_batch_, neighbour_indexes)
        gen_neibour_normals_ = torch_tensor_functions.indexing_neighbor(gen_normals_batch_, neighbour_indexes)
        gen_normals_batch_ = gen_normals_batch_.view(thisbatchsize*self.args.num_point, -1 ,1, 3)
        gen_neibour_normals_delta_ = gen_neibour_normals_ - gen_normals_batch_
        gen_neibour_normals_delta_squ = torch.mul(gen_neibour_normals_delta_, gen_neibour_normals_delta_)

        normals_delta_squ_bar = gen_neibour_normals_delta_squ.detach()*0+1
        normals_delta_squ_sign = torch.sign(gen_neibour_normals_delta_squ.detach() - normals_delta_squ_bar)*0.5+1
        gen_neibour_normals_delta_squ = torch.mul(gen_neibour_normals_delta_squ, normals_delta_squ_sign)

        output += torch.sum(gen_neibour_normals_delta_squ)
        
        return output/thisbatchsize


    
    def forward(self, gen_points_batch, gen_normals_batch, uv_sampling_coors, train_points_sparse_batch, train_normals_sparse_batch, train_points_dense_batch, train_normals_dense_batch):
        thisbatchsize = gen_points_batch.size()[0]
        loss = torch.mean(torch.zeros((1),dtype = torch.float, device=gen_points_batch.device))
        zero_tensor = torch.mean(torch.zeros((1),dtype = torch.float, device=gen_points_batch.device))
        loss_stages=[]
        
        if self.args.weight_cd > 0:
            # L^{cd}  # n*3, n*3
            loss_cd = 0 
            loss_cd += self.loss_on_cd(gen_points_batch, train_points_dense_batch)
            loss += loss_cd * self.args.weight_cd
            loss_stages.append(loss_cd)
        else:
            loss_stages.append(zero_tensor)    

        if self.args.weight_reg > 0:
            # L^{reg}  # n*3, n*3
            loss_reg = 0 
            loss_reg += self.loss_on_reg(gen_points_batch, train_points_sparse_batch)
            loss += loss_reg * self.args.weight_reg
            loss_stages.append(loss_reg)
        else:
            loss_stages.append(zero_tensor) 

        if self.args.weight_arap > 0:
            # L^{arap}  # 
            loss_arap = 0 
            loss_arap += self.loss_on_arap(gen_points_batch, uv_sampling_coors)
            loss += loss_arap * self.args.weight_arap
            loss_stages.append(loss_arap)
        else:
            loss_stages.append(zero_tensor) 


        if self.args.weight_overlap > 0:
            # L^{overlap}  # 
            loss_overlap = 0 
            loss_overlap += self.loss_on_overlap(gen_points_batch, train_points_sparse_batch)
            loss += loss_overlap * self.args.weight_overlap
            loss_stages.append(loss_overlap)
        else:
            loss_stages.append(zero_tensor) 
           
        
        if self.args.weight_proj > 0:
            # L^{proj}  # 
            loss_proj = 0 
            loss_proj += self.loss_on_proj(gen_points_batch, train_points_dense_batch)
            loss += loss_proj * self.args.weight_proj
            loss_stages.append(loss_proj)
        else:
            loss_stages.append(zero_tensor) 

        if self.args.weight_normal > 0:
            # L^{normal}  # 
            loss_normal = 0 
            loss_normal += self.loss_on_normal(gen_points_batch, train_points_dense_batch, gen_normals_batch, train_normals_dense_batch)
            loss += loss_normal * self.args.weight_normal
            loss_stages.append(loss_normal)
        else:
            loss_stages.append(zero_tensor) 


        if self.args.weight_ndirection > 0:
            # L^{ndirection}  # 
            loss_ndirection = 0 
            loss_ndirection += self.loss_on_ndirection(gen_points_batch, uv_sampling_coors, gen_normals_batch)
            loss += loss_ndirection * self.args.weight_ndirection
            loss_stages.append(loss_ndirection)
        else:
            loss_stages.append(zero_tensor) 
           
            
        return loss, loss_stages
