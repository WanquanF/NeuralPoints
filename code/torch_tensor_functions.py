#### Author : Wanquan Feng (University of Science and Technology of China)
#### Description : Some operations of the point cloud based on the pytorch tensor
#### Data : 2021-10-16


import os
import sys
import torch
import numpy
import mesh_operations



def compute_sqrdis_map(points_x, points_y):
    ## The shape of the input and output ##
    # points_x : batchsize * M * 3
    # points_y : batchsize * N * 3
    # output   : batchsize * M * N
    thisbatchsize = points_x.size()[0]
    pn_x = points_x.size()[1]
    pn_y = points_y.size()[1]
    x_sqr = torch.sum(torch.mul(points_x, points_x), dim=-1).view(thisbatchsize, pn_x, 1).expand(-1,-1,pn_y)
    y_sqr = torch.sum(torch.mul(points_y, points_y), dim=-1).view(thisbatchsize, 1, pn_y).expand(-1,pn_x,-1)
    inner = torch.bmm(points_x, points_y.transpose(1,2))
    sqrdis = x_sqr + y_sqr - 2*inner
    return sqrdis

def draw_tensor_point_xyz_with_normal(save_path, torch_tensor_points, torch_tensor_normals=torch.ones([1])):
    ## The shape of the input ##
    # torch_tensor_points : M * 3
    # torch_tensor_normals (optional) : M * 3 
    if len(torch_tensor_points.size())!=2:
        print('The size of the point tensor should be 2. Exit here.')
        exit()
    if torch_tensor_points.size()[1]!=3:
        print('The dim of the point tensor is not correct. It should be (num_point, 3).')
        exit()
    numpy_points = torch_tensor_points.cpu().numpy()
    numpy_normals = torch_tensor_normals.cpu().numpy()
    mesh_operations.write_xyz_(save_path, numpy_points, numpy_normals)


def draw_tensor_point_xyz_with_normal_by_threshold(save_path, torch_tensor_points, torch_anchor, torch_tensor_normals=torch.ones([1]), threshold=0.95, ):
    ## The shape of the input ##
    # torch_tensor_points : M * 3
    # torch_tensor_normals (optional) : M * 3 
    # threshold : a float value < 1
    if len(torch_tensor_points.size())!=2:
        print('The size of the point tensor should be 2. Exit here.')
        exit()
    if torch_tensor_points.size()[1]!=3:
        print('The dim of the point tensor is not correct. It should be (num_point, 3).')
        exit()
    torch_tensor_points_norm = torch.sum(torch.mul(torch_tensor_points, torch_tensor_points), dim=1)

    numpy_points = torch_tensor_points.cpu().numpy()
    numpy_normals = torch_tensor_normals.cpu().numpy()
    mesh_operations.write_xyz_(save_path, numpy_points, numpy_normals)

def draw_tensor_point_obj_with_color(save_path, torch_tensor_points, torch_tensor_color=torch.ones([1])):
    ## The shape of the input ##
    # torch_tensor_points : M * 3
    # torch_tensor_color (optional) : M * 3 
    if len(torch_tensor_points.size())!=2:
        print('The size of the point tensor should be 2. Exit here.')
        exit()
    if torch_tensor_points.size()[1]!=3:
        print('The dim of the point tensor is not correct. It should be (num_point, 3).')
        exit()
    numpy_points = torch_tensor_points.cpu().numpy()
    numpy_color = torch_tensor_color.cpu().numpy()
    mesh_operations.write_obj_(save_path, numpy_points, color_=torch_tensor_color.cpu().numpy())


def draw_tensor_point_batch_xyz_with_normal(save_batch_path, torch_tensor_points_batch, torch_tensor_normals_batch=torch.ones([1])):
    ## The shape of the input ##
    # torch_tensor_points : B * M * 3
    # torch_tensor_normals (optional) : B * M * 3 
    if not os.path.exists(save_batch_path):os.mkdir(save_batch_path)
    thisbatchsize = len(torch_tensor_points_batch)
    for bi in range(thisbatchsize):
        bi_path = save_batch_path+'/'+str(bi)+'.xyz'
        torch_tensor_points = torch_tensor_points_batch[bi]
        if len(torch_tensor_normals_batch.size())==1: torch_tensor_normals = torch.ones([1])
        else:torch_tensor_normals = torch_tensor_normals_batch[bi]
        draw_tensor_point_xyz_with_normal(bi_path, torch_tensor_points, torch_tensor_normals)
    

def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))



def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int):
    # Return: (bs, vertice_num, neighbor_num)
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k= neighbor_num + 1, dim= -1, largest= False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, query_vertice_num, neighbor_num)" ):
    # Return: (bs, query_vertice_num, neighbor_num, dim)
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed


def indexing_by_id(tensor: "(bs, vertice_num, dim)", index: "(bs, query_num, neighbor_num)" ):
    # Return: (bs, query_num, neighbor_num, dim)
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed