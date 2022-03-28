import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import math
import numpy as np
import torch.nn.init as init
import struct
import os
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../code/')
#import drawer
import time
import mesh_operations
import torch_tensor_functions
import colormap
import random
from pointnet2 import pointnet2_utils as pn2_utils

# from torch_geometric.data import Data
# from torch_geometric.transforms.generate_mesh_normals  import *
# from torch_scatter import scatter_add


######  The network and loss are figured out here  ###### 
from loss import Loss, chamfer_dist
###### ------ ######


from utils.config import parse_args
import time
import igl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The parameter that controls the overfitting.
# over_fitting_id = 0
# if_over_fitting_this_time = False
# if_only_test = False
# if_only_test_max_num = 3




# Set the GradScaler
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# all the args. They can be set in another .py file.
args = parse_args()
print ('args:')
print (args)


exec('from network import '+args.network_name)

over_fitting_id = args.over_fitting_id
if_over_fitting_this_time = args.if_over_fitting_this_time
if_only_test = args.if_only_test
if_only_test_max_num = args.if_only_test_max_num

# The color map for visualization.
# The points generated from a same source point should share the same color. 
rb_colormap = np.array(colormap.rb_colormap_list_little).reshape(8,3) 

batch_size=args.batchsize
# It is used to control the training process.
train_max_samples = args.train_max_samples  

# The path of the packed dataset.
pack_path=args.pack_path
print('The packed data path is : ',pack_path)

# The point number of the sparse and dense patch, respectively.
num_point = args.num_point
gt_num_point = args.gt_num_point

# The path of the training data
train_points_normals_sparse_path = pack_path+'/training_points_normals_'+str(num_point)+'.bin'
train_points_normals_dense_path = pack_path+'/training_points_normals_'+str(gt_num_point)+'.bin'

# The path of the testing data
test_points_normals_sparse_path = pack_path+'/testing_points_normals_'+str(num_point)+'.bin'
test_center_scale_sparse_path = pack_path+'/testing_center_scale_'+str(num_point)+'.bin'
test_points_normals_dense_path = pack_path+'/testing_points_normals_'+str(gt_num_point)+'.bin'

# READ in train_points_normals : sparse
train_points_normals_sparse = np.fromfile(train_points_normals_sparse_path, dtype = np.float32).reshape(-1,num_point,6)
# READ in train_points_normals : dense
train_points_normals_dense = np.fromfile(train_points_normals_dense_path, dtype = np.float32).reshape(-1,gt_num_point,6)
# READ in test_points_normals : sparse
test_points_normals_sparse = np.fromfile(test_points_normals_sparse_path, dtype = np.float32).reshape(-1,num_point,6)
test_center_scale_sparse = np.fromfile(test_center_scale_sparse_path, dtype = np.float32).reshape(-1,4)
# READ in test_points_normals : dense
test_points_normals_dense = np.fromfile(test_points_normals_dense_path, dtype = np.float32).reshape(-1,gt_num_point,6)



# The pair number of the training and testing pairs, respectively.
train_pair_num = train_points_normals_sparse.shape[0]
test_pair_num = test_points_normals_sparse.shape[0]

train_points_normals_sparse_tensor = torch.from_numpy(train_points_normals_sparse).float()
test_points_normals_sparse_tensor = torch.from_numpy(test_points_normals_sparse).float()
train_points_normals_dense_tensor = torch.from_numpy(train_points_normals_dense).float()
test_points_normals_dense_tensor = torch.from_numpy(test_points_normals_dense).float()

# All the torch-tensors used for the input and ground truth. 
train_points_sparse_tensor = train_points_normals_sparse_tensor[:,:,:3]
train_normals_sparse_tensor = train_points_normals_sparse_tensor[:,:,3:]
test_points_sparse_tensor = test_points_normals_sparse_tensor[:,:,:3]
test_normals_sparse_tensor = test_points_normals_sparse_tensor[:,:,3:]
train_points_dense_tensor = train_points_normals_dense_tensor[:,:,:3]
train_normals_dense_tensor = train_points_normals_dense_tensor[:,:,3:]
test_points_dense_tensor = test_points_normals_dense_tensor[:,:,:3]
test_normals_dense_tensor = test_points_normals_dense_tensor[:,:,3:]

# All the batch-data used for the input and ground truth. 
train_points_sparse_batch = torch.zeros([batch_size,num_point,3],dtype=torch.float,requires_grad=False).cuda()
train_normals_sparse_batch = torch.zeros([batch_size,num_point,3],dtype=torch.float,requires_grad=False).cuda()
test_points_sparse_batch = torch.zeros([batch_size,num_point,3],dtype=torch.float,requires_grad=False).cuda()
test_normals_sparse_batch = torch.zeros([batch_size,num_point,3],dtype=torch.float,requires_grad=False).cuda()
train_points_dense_batch = torch.zeros([batch_size,gt_num_point,3],dtype=torch.float,requires_grad=False).cuda()
train_normals_dense_batch = torch.zeros([batch_size,gt_num_point,3],dtype=torch.float,requires_grad=False).cuda()
test_points_dense_batch = torch.zeros([batch_size,gt_num_point,3],dtype=torch.float,requires_grad=False).cuda()
test_normals_dense_batch = torch.zeros([batch_size,gt_num_point,3],dtype=torch.float,requires_grad=False).cuda()


def update_test_cache(used_samples_num, model, loss_obj, args):
    print('updating cache for used_samples_num = ' + str(used_samples_num))
    test_cache_file='./'+args.out_baseline+'/result_cache.txt'
    loss_sum_, loss_stages_=compute_test_loss_values(model, loss_obj, args)
    print('the test loss: ',loss_sum_, loss_stages_)
    cf=open(test_cache_file,'a+')
    # The first number is the iteration times.
    cf.write(str(used_samples_num//batch_size)+' ')
    cf.write(str(loss_sum_)+' ')
    for i in range(len(loss_stages_)):
        cf.write(str(loss_stages_[i])+' ')
    cf.write('\n')
    cf.close()
    update_pics()
    if args.visualization_while_testing:
        update_visualization(model,  args)
        if if_only_test==True:exit()
    
def update_pics():
    test_cache_file='./'+args.out_baseline+'/result_cache.txt'
    cf=open(test_cache_file,'r')
    lines=cf.readlines()
    x=[]
    y_sum=[]
    y_cd=[]
    y_reg = []
    y_arap = []
    y_overlap = []
    y_proj = []
    y_normal = []
    y_ndirection = []
    for i in range(len(lines)):
        if i%1==0:
            index = int(lines[i].split(' ')[0])
            sum_loss = float(lines[i].split(' ')[1])
            cd_loss = float(lines[i].split(' ')[2])
            reg_loss = float(lines[i].split(' ')[3])
            arap_loss = float(lines[i].split(' ')[4])
            overlap_loss = float(lines[i].split(' ')[5])
            proj_loss = float(lines[i].split(' ')[6])
            normal_loss = float(lines[i].split(' ')[7])
            ndirection_loss = float(lines[i].split(' ')[8])
            iter_index=index
            x.append(iter_index)
            y_sum.append(sum_loss)
            y_cd.append(cd_loss)
            y_reg.append(reg_loss)
            y_arap.append(arap_loss)
            y_overlap.append(overlap_loss)
            y_proj.append(proj_loss)
            y_normal.append(normal_loss)
            y_ndirection.append(ndirection_loss)
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The sum loss')
    plt.xlabel('iteration')
    plt.ylabel('sum loss')
    plt.plot(x, y_sum, c='r', ls='-')
    plt.savefig('./'+args.out_baseline+'/loss_sum.png')
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on cd')
    plt.xlabel('iteration')
    plt.ylabel('cd loss')
    plt.plot(x, y_cd, c='#526922', ls='-')
    plt.savefig('./'+args.out_baseline+'/loss_cd.png')

    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on reg')
    plt.xlabel('iteration')
    plt.ylabel('reg loss')
    plt.plot(x, y_reg, c='#526922', ls='-')
    plt.savefig('./'+args.out_baseline+'/loss_reg.png')

    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on arap')
    plt.xlabel('iteration')
    plt.ylabel('arap loss')
    plt.plot(x, y_arap, c='#526922', ls='-')
    plt.savefig('./'+args.out_baseline+'/loss_arap.png')

    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on overlap')
    plt.xlabel('iteration')
    plt.ylabel('overlap loss')
    plt.plot(x, y_overlap, c='#526922', ls='-')
    plt.savefig('./'+args.out_baseline+'/loss_overlap.png')

    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on proj')
    plt.xlabel('iteration')
    plt.ylabel('proj loss')
    plt.plot(x, y_proj, c='#526922', ls='-')
    plt.savefig('./'+args.out_baseline+'/loss_proj.png')

    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on normal')
    plt.xlabel('iteration')
    plt.ylabel('normal loss')
    plt.plot(x, y_normal, c='#526922', ls='-')
    plt.savefig('./'+args.out_baseline+'/loss_normal.png')

    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on ndirection')
    plt.xlabel('iteration')
    plt.ylabel('ndirection loss')
    plt.plot(x, y_ndirection, c='#526922', ls='-')
    plt.savefig('./'+args.out_baseline+'/loss_ndirection.png')
    
def update_visualization(model,  args):
    global test_center_scale_sparse

    global train_points_sparse_tensor
    global train_normals_sparse_tensor
    global test_points_sparse_tensor
    global test_normals_sparse_tensor
    global train_points_dense_tensor
    global train_normals_dense_tensor
    global test_points_dense_tensor
    global test_normals_dense_tensor

    global train_points_sparse_batch
    global train_normals_sparse_batch
    global test_points_sparse_batch
    global test_normals_sparse_batch
    global train_points_dense_batch
    global train_normals_dense_batch
    global test_points_dense_batch
    global test_normals_dense_batch
    
    test_cache_file='./'+args.out_baseline+'/result_cache.txt'
    cf=open(test_cache_file,'r')
    lines=cf.readlines()
    last_line = lines[len(lines)-1]
    iter_num = int(last_line.split(' ')[0])
    visual_folder = './'+args.out_baseline+'/visual_'+str(iter_num*batch_size)
    if not os.path.exists(visual_folder):
        os.mkdir(visual_folder)
    print('Satrt to visualize the results now.')
    # To be finished. Draw whatever you wanna observe here.
    visual_sample_num = min(batch_size,5)
    if if_only_test==True:
        visual_sample_num = test_pair_num
        testing_anchor_num = args.testing_anchor_num
        testing_model_num = test_pair_num // testing_anchor_num
        if if_only_test_max_num>=0 and if_only_test_max_num<testing_model_num: testing_model_num=if_only_test_max_num
        visual_sample_num = testing_model_num*testing_anchor_num
    if not if_over_fitting_this_time:over_fitting_id_here=0
    else:over_fitting_id_here=args.over_fitting_id
    for si in range(over_fitting_id_here, over_fitting_id_here+visual_sample_num):
        this_sample_path = visual_folder+'/sample_'+str(si)
        if not os.path.exists(this_sample_path):os.mkdir(this_sample_path)
        a_points_sparse_tensor = test_points_sparse_tensor[si:si+1].cuda()
        a_normals_sparse_tensor = test_normals_sparse_tensor[si:si+1].cuda()
        a_points_dense_tensor = test_points_dense_tensor[si:si+1].cuda()
        a_normals_dense_tensor = test_normals_dense_tensor[si:si+1].cuda()
        # get the generated results.
        model.eval()
        # with torch.no_grad():
        if True:
            a_points_gen_tensor, a_normals_gen_tensor, _, a_querying_points_3d, a_querying_points_n_3d, a_glued_points, a_glued_normals = model(a_points_sparse_tensor)
        
        # save the points : format-xyz, with normal.
        torch_tensor_functions.draw_tensor_point_xyz_with_normal(this_sample_path+'/query.xyz', a_points_gen_tensor[0].detach(),torch_tensor_normals=a_normals_gen_tensor[0].detach())
        torch_tensor_functions.draw_tensor_point_xyz_with_normal(this_sample_path+'/query_3d.xyz', a_querying_points_3d[0].detach(), torch_tensor_normals=a_querying_points_n_3d[0].detach())
        torch_tensor_functions.draw_tensor_point_xyz_with_normal(this_sample_path+'/glued.xyz', a_glued_points[0].detach())
        torch_tensor_functions.draw_tensor_point_xyz_with_normal(this_sample_path+'/sparse.xyz', a_points_sparse_tensor[0])
        torch_tensor_functions.draw_tensor_point_xyz_with_normal(this_sample_path+'/dense.xyz', a_points_dense_tensor[0])

        # the color tensor of the sparse points
        num_point_here = a_points_sparse_tensor.size()[1]
        a_points_sparse_color_tensor = torch.from_numpy(rb_colormap).float().cuda()
        while a_points_sparse_color_tensor.size()[0]<num_point_here:a_points_sparse_color_tensor = torch.cat((a_points_sparse_color_tensor,a_points_sparse_color_tensor),dim=0)
        a_points_sparse_color_tensor = a_points_sparse_color_tensor[:num_point_here]
        
        # the color tensor of the generated points
        up_ratio_here = a_points_gen_tensor.size()[1]//a_points_sparse_tensor.size()[1]
        a_points_gen_color_tensor = a_points_sparse_color_tensor.clone().view(1,-1,3)
        while a_points_gen_color_tensor.size()[0]<up_ratio_here:a_points_gen_color_tensor = torch.cat((a_points_gen_color_tensor,a_points_gen_color_tensor),dim=0)
        a_points_gen_color_tensor = a_points_gen_color_tensor[:up_ratio_here].transpose(1,0)
        a_points_gen_color_tensor = a_points_gen_color_tensor.reshape(-1,3)
        
        # save the points : format-obj, with color.
        torch_tensor_functions.draw_tensor_point_obj_with_color(this_sample_path+'/query.obj', a_points_gen_tensor[0].detach(),torch_tensor_color=a_points_gen_color_tensor)
        torch_tensor_functions.draw_tensor_point_obj_with_color(this_sample_path+'/sparse.obj', a_points_sparse_tensor[0],torch_tensor_color=a_points_sparse_color_tensor)
    
    # if if_only_test==True : Test all the testing models. 
    if if_only_test==True:
        tested_mesh_path = visual_folder + '/0tested_models'
        if not os.path.exists(tested_mesh_path):os.mkdir(tested_mesh_path)
        for model_i in range(testing_model_num):
            all_patches_points = []
            one_tested_mesh_obj_path = tested_mesh_path+'/test_model_'+str(model_i)+'.obj'
            for anchor_i in range(testing_anchor_num):
                this_sample_id = model_i*testing_anchor_num + anchor_i
                this_sample_path = visual_folder+'/sample_'+str(this_sample_id)
                v_, n_ = mesh_operations.read_xyz_(this_sample_path+'/glued.xyz')
                v_ = v_[:,:3]
                this_center_scale = test_center_scale_sparse[this_sample_id]
                this_center = this_center_scale[:3].reshape(1,3)
                this_scale = this_center_scale[3]
                v_ = v_ * this_scale
                v_ = v_ + this_center
                all_patches_points.append(v_)
            all_patches_points = np.concatenate(all_patches_points,axis=0)
            all_patches_points_torch = torch.from_numpy(all_patches_points).float().cuda().view(1,-1,3)
            fps_id = pn2_utils.furthest_point_sample(all_patches_points_torch.contiguous(), 2000*args.testing_up_ratio)
            new_xyz = pn2_utils.gather_operation(all_patches_points_torch.permute(0, 2, 1).contiguous(), fps_id)
            all_patches_points = new_xyz.permute(0,2,1).view(-1,3).cpu().numpy().astype(np.float32)
            mesh_operations.write_obj_(one_tested_mesh_obj_path, all_patches_points)
                







    
    
def stophere():
    while True:
        continue

def run_train_val(model, optimizer, loss_obj,  args):
    global train_points_sparse_tensor
    global train_normals_sparse_tensor
    global test_points_sparse_tensor
    global test_normals_sparse_tensor
    global train_points_dense_tensor
    global train_normals_dense_tensor
    global test_points_dense_tensor
    global test_normals_dense_tensor

    global train_points_sparse_batch
    global train_normals_sparse_batch
    global test_points_sparse_batch
    global test_normals_sparse_batch
    global train_points_dense_batch
    global train_normals_dense_batch
    global test_points_dense_batch
    global test_normals_dense_batch
    
    used_samples_num=args.last_sample_id
    start_pos=used_samples_num % train_pair_num

    if used_samples_num==0 or if_only_test==True:
        update_test_cache(used_samples_num, model, loss_obj,  args)
    
    while used_samples_num<train_max_samples:
        while True:
            end_pos=start_pos+batch_size
            print('Training with pair samples: '+str(start_pos)+'~'+str(end_pos))
            train_one_batch(model, optimizer, loss_obj, start_pos, end_pos, args) ############## train one batch
            used_samples_num+=end_pos-start_pos
            if used_samples_num%(args.test_blank)==0:
                update_test_cache(used_samples_num, model, loss_obj, args) ############## test once
                print('Test here, at '+str(used_samples_num))
                torch.save(model.state_dict(), './'+args.out_baseline+'/sample_'+str(used_samples_num)+'.pt')
            if end_pos>=train_pair_num:
                start_pos=end_pos - train_pair_num
            else:
                start_pos=end_pos
            print(used_samples_num,train_max_samples)
            if used_samples_num >= train_max_samples:
                break
    
    
    
def train_one_batch(model, optimizer, loss_obj, start_pos, end_pos, args):
    global train_points_sparse_tensor
    global train_normals_sparse_tensor
    global test_points_sparse_tensor
    global test_normals_sparse_tensor
    global train_points_dense_tensor
    global train_normals_dense_tensor
    global test_points_dense_tensor
    global test_normals_dense_tensor

    global train_points_sparse_batch
    global train_normals_sparse_batch
    global test_points_sparse_batch
    global test_normals_sparse_batch
    global train_points_dense_batch
    global train_normals_dense_batch
    global test_points_dense_batch
    global test_normals_dense_batch
    
    print(start_pos, end_pos)
    if end_pos<=train_pair_num:
        train_points_sparse_batch = train_points_sparse_tensor[start_pos:end_pos].cuda()
        train_normals_sparse_batch = train_normals_sparse_tensor[start_pos:end_pos].cuda()
        train_points_dense_batch = train_points_dense_tensor[start_pos:end_pos].cuda()
        train_normals_dense_batch = train_normals_dense_tensor[start_pos:end_pos].cuda()
    else:
        bottom = train_pair_num - start_pos
        top = end_pos - train_pair_num
        
        train_points_sparse_batch[:bottom] = train_points_sparse_tensor[start_pos:].cuda()
        train_normals_sparse_batch[:bottom] = train_normals_sparse_tensor[start_pos:].cuda()
        train_points_dense_batch[:bottom] = train_points_dense_tensor[start_pos:].cuda()
        train_normals_dense_batch[:bottom] = train_normals_dense_tensor[start_pos:].cuda()
        
        
        train_points_sparse_batch[bottom:] = train_points_sparse_tensor[:top].cuda()
        train_normals_sparse_batch[bottom:] = train_normals_sparse_tensor[:top].cuda()
        train_points_dense_batch[bottom:] = train_points_dense_tensor[:top].cuda()
        train_normals_dense_batch[bottom:] = train_normals_dense_tensor[:top].cuda()
    
    # For over-fitting!!
    if if_over_fitting_this_time==True:
        train_points_sparse_batch = test_points_sparse_tensor[0+over_fitting_id:end_pos-start_pos+over_fitting_id].cuda()
        train_normals_sparse_batch = test_normals_sparse_tensor[0+over_fitting_id:end_pos-start_pos+over_fitting_id].cuda()
        train_points_dense_batch = test_points_dense_tensor[0+over_fitting_id:end_pos-start_pos+over_fitting_id].cuda()
        train_normals_dense_batch = test_normals_dense_tensor[0+over_fitting_id:end_pos-start_pos+over_fitting_id].cuda()

        # torch_tensor_functions.draw_tensor_point_batch_xyz_with_normal('./train_sparsepoint_shows', train_points_sparse_batch, train_normals_sparse_batch)
        # torch_tensor_functions.draw_tensor_point_batch_xyz_with_normal('./train_densepoint_shows', train_points_dense_batch, train_normals_dense_batch)
    if if_over_fitting_this_time==False:
        pi_ = 3.14159265
        all_rot_matrix_ = None
        for b in range(train_points_sparse_batch.size()[0]):
            euler_x = random.randint(0,10000)/10000
            euler_y = random.randint(0,10000)/10000
            euler_z = random.randint(0,10000)/10000
            euler_angle = torch.tensor([[-pi_+2*pi_*euler_x, -pi_+2*pi_*euler_y, -pi_+2*pi_*euler_z]], dtype=torch.float32).cuda()
            a_rot_matrix_ = torch_tensor_functions.euler2rot(euler_angle)
            if b==0:all_rot_matrix_ = a_rot_matrix_
            else:all_rot_matrix_ = torch.cat((all_rot_matrix_,a_rot_matrix_),dim=0)
        train_points_sparse_batch = torch.bmm(train_points_sparse_batch, all_rot_matrix_)
        train_normals_sparse_batch = torch.bmm(train_normals_sparse_batch, all_rot_matrix_)
        train_points_dense_batch = torch.bmm(train_points_dense_batch, all_rot_matrix_)
        train_normals_dense_batch = torch.bmm(train_normals_dense_batch, all_rot_matrix_)


    for train_times in range(1):
        optimizer.zero_grad()    
        model.train()
        gen_points_batch, gen_normals_batch, uv_sampling_coors, _, _, glued_points, glued_normals = model(train_points_sparse_batch)
        
        conpu_loss, conpu_loss_stages  = loss_obj(gen_points_batch, gen_normals_batch, uv_sampling_coors, train_points_sparse_batch, train_normals_sparse_batch, train_points_dense_batch, train_normals_dense_batch)
        print('cd:',conpu_loss_stages[0])
        print('reg:',conpu_loss_stages[1])
        print('arap:',conpu_loss_stages[2])
        print('overlap:',conpu_loss_stages[3])
        print('proj:',conpu_loss_stages[4])
        print('normal:',conpu_loss_stages[5])
        print('ndirection:',conpu_loss_stages[6])
        
        model.zero_grad()
        if True:
            with torch.autograd.set_detect_anomaly(True): scaler.scale(conpu_loss).backward()
            if_have_nan = False
            if if_have_nan==False:
                scaler.unscale_(optimizer)                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                print('optimizer.lr : ',optimizer.state_dict()['param_groups'][0]['lr'])
                scheduler.step()
                scaler.update()
            else:
                print('The grad is dirty!!!')
        else:
            conpu_loss.backward()
            optimizer.step()


    
def test_one_batch(model, loss_obj, start_pos, end_pos, args):
    global train_points_sparse_tensor
    global train_normals_sparse_tensor
    global test_points_sparse_tensor
    global test_normals_sparse_tensor
    global train_points_dense_tensor
    global train_normals_dense_tensor
    global test_points_dense_tensor
    global test_normals_dense_tensor

    global train_points_sparse_batch
    global train_normals_sparse_batch
    global test_points_sparse_batch
    global test_normals_sparse_batch
    global train_points_dense_batch
    global train_normals_dense_batch
    global test_points_dense_batch
    global test_normals_dense_batch

    
#    model.eval()
    
#    print(start_pos, end_pos)
    if end_pos<=test_pair_num:
        test_points_sparse_batch = test_points_sparse_tensor[start_pos:end_pos].cuda()
        test_normals_sparse_batch = test_normals_sparse_tensor[start_pos:end_pos].cuda()
        test_points_dense_batch = test_points_dense_tensor[start_pos:end_pos].cuda()
        test_normals_dense_batch = test_normals_dense_tensor[start_pos:end_pos].cuda()
    else:
        bottom = test_pair_num - start_pos
        top = end_pos - test_pair_num
        
        test_points_sparse_batch[:bottom] = test_points_sparse_tensor[start_pos:].cuda()
        test_normals_sparse_batch[:bottom] = test_normals_sparse_tensor[start_pos:].cuda()
        test_points_dense_batch[:bottom] = test_points_dense_tensor[start_pos:].cuda()
        test_normals_dense_batch[:bottom] = test_normals_dense_tensor[start_pos:].cuda()
        
        
        test_points_sparse_batch[bottom:] = test_points_sparse_tensor[:top].cuda()
        test_normals_sparse_batch[bottom:] = test_normals_sparse_tensor[:top].cuda()
        test_points_dense_batch[bottom:] = test_points_dense_tensor[:top].cuda()
        test_normals_dense_batch[bottom:] = test_normals_dense_tensor[:top].cuda()
        
    # For over-fitting!!
    if if_over_fitting_this_time==True:
        test_points_sparse_batch = test_points_sparse_tensor[0+over_fitting_id:end_pos-start_pos+over_fitting_id].cuda()
        test_normals_sparse_batch = test_normals_sparse_tensor[0+over_fitting_id:end_pos-start_pos+over_fitting_id].cuda()
        test_points_dense_batch = test_points_dense_tensor[0+over_fitting_id:end_pos-start_pos+over_fitting_id].cuda()
        test_normals_dense_batch = test_normals_dense_tensor[0+over_fitting_id:end_pos-start_pos+over_fitting_id].cuda()
    
    
    model.eval()
    # with torch.no_grad():
    if True:
        gen_points_batch, gen_normals_batch, uv_sampling_coors, _, _, glued_points, glued_normals = model(test_points_sparse_batch)

        conpu_loss, conpu_loss_stages = loss_obj(gen_points_batch, gen_normals_batch, uv_sampling_coors, test_points_sparse_batch, test_normals_sparse_batch, test_points_dense_batch, test_normals_dense_batch)

    return conpu_loss, conpu_loss_stages
    
    
    

    
def compute_test_loss_values(model, loss_obj, args):
    start_pos=0
    loss_sum=0.0
    loss_stages=[]
    batch_cnt=0.0
    print('Computing the testing loss on the testing set:')
    for s in range(0, 2, batch_size):
        start_pos = s
        end_pos = s + batch_size
        if end_pos > test_pair_num:
            end_pos = test_pair_num
        this_batch_size = end_pos - start_pos
        lsum,lstages = test_one_batch(model, loss_obj, start_pos, end_pos, args)
        if start_pos==0:
            loss_sum=lsum.item()*this_batch_size
            for i in range(len(lstages)):
                loss_stages.append(lstages[i].item()*this_batch_size)
        else:
            loss_sum+=lsum.item()*this_batch_size
            for i in range(len(lstages)):
                loss_stages[i]+=lstages[i].item()*this_batch_size
        batch_cnt += this_batch_size
    loss_sum/=batch_cnt
    for i in range(len(loss_stages)):
        loss_stages[i]/=batch_cnt
    return loss_sum, loss_stages

def show_parameter_by_name(net_name, layer_name):
    for name, parameters in net_name.named_parameters():
        if name==layer_name:
            return parameters
    return None
    
def get_para_of_one_layer_from_another_net(net_source, net_to_be_changed, layer_name):
    a = show_parameter_by_name(net_source, layer_name)
#    print(show_parameter_by_name(net_to_be_changed, layer_name))
    for name, parameters in net_to_be_changed.named_parameters():
        if name==layer_name:
            parameters.data = a.data
            return None
    print('No matching for layer: ',layer_name)
#    print(show_parameter_by_name(net_to_be_changed, layer_name))
    
    

if __name__=='__main__':
    exec('conpu_net = '+str(args.network_name)+'(args).cuda()')
    if False:
        print('#parameters:', sum(param.numel() for param in conpu_net.parameters())*4/(1024*1024),' Mb')
        exit()
    if args.last_sample_id==0:
        if os.path.exists('./'+args.out_baseline):
            os.system('rm -rf ./'+args.out_baseline)
        os.makedirs('./'+args.out_baseline)
        if len(args.pretrained)>=1: conpu_net.load_state_dict(torch.load(args.pretrained),True)
        torch.save(conpu_net.state_dict(), './'+args.out_baseline+'/sample_0.pt')
        # os.system('cp ./out_baseline_5/sample_600000.pt ./'+args.out_baseline+'/sample_0.pt')
        
    if args.if_only_test==True: conpu_net.load_state_dict(torch.load(args.pretrained),True)
    else: conpu_net.load_state_dict(torch.load('./'+args.out_baseline+'/sample_'+str(args.last_sample_id)+'.pt'),True)
    
    # setup optimizer
    optimizer = optim.AdamW(conpu_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.learning_rate, total_steps = (args.train_max_samples - args.last_sample_id)//args.batchsize, pct_start=0.03, cycle_momentum=False, anneal_strategy='linear')
    scaler = GradScaler(enabled=args.mixed_precision)
    
    # setup loss object
    loss_obj = Loss(args)
    
    # run train and test
    run_train_val(conpu_net, optimizer, loss_obj,  args)
    print('Done.')
