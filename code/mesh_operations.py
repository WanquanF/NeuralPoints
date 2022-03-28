#### Author : Wanquan Feng (University of Science and Technology of China)
#### Description : Some operations of the mesh/pc based on the numpy array
#### Data : 2021-10-16

import os
import sys
import numpy
import igl


#  off format
def read_off_(off_file_name):
    v,f,_ = igl.read_off(off_file_name)
    return v,f
def write_off_(off_file_name,v,face_=numpy.zeros((1))):
    fout = open(off_file_name,'w')
    fout.write('OFF\n')
    fout.write(str(v.shape[0])+' '+str(face_.shape[0])+' 0\n')
    for i in range(v.shape[0]):
        fout.write(str(v[i][0])+' '+str(v[i][1])+' '+str(v[i][2])+'\n')
    if face_.shape[0]<2:return None
    for i in range(face_.shape[0]):
        fout.write('3 '+str(face_[i][0])+' '+str(face_[i][1])+' '+str(face_[i][2])+'\n')
    fout.close()
    return None

# obj format
def write_obj_(obj_write_name,v,face_=numpy.zeros((1)),color_=numpy.zeros((1)),normal_=numpy.zeros((1))):
    f=open(obj_write_name,'w')
    vnum = v.shape[0]
    for vid in range(vnum):
        f.write('v '+str(v[vid][0])+' '+str(v[vid][1])+' '+str(v[vid][2]))
        if color_.shape[0]<vnum: f.write('\n')
        else:f.write(' '+str(color_[vid][0])+' '+str(color_[vid][1])+' '+str(color_[vid][2])+'\n')
        if normal_.shape[0]==vnum:
            f.write('vn '+str(normal_[vid][0])+' '+str(normal_[vid][1])+' '+str(normal_[vid][2])+'\n')
    if face_.shape[0]<2:
        f.close()
        return None
    fnum = face_.shape[0]
    for fid in range(fnum):
        f.write('f '+str(face_[fid][0]+1)+' '+str(face_[fid][1]+1)+' '+str(face_[fid][2]+1)+'\n')
    f.close()
    return None
def read_obj_(obj_write_name):
    v, _, n, f, _, _ = igl.read_obj(obj_write_name)
    return v, f, n

# xyz format
def write_xyz_(xyz_write_name,v,normal_=numpy.zeros((1))):
    f = open(xyz_write_name, 'w')
    vnum = v.shape[0]
    for i in range(vnum):
        f.write(str(v[i][0])+' '+str(v[i][1])+' '+str(v[i][2]))
        if normal_.shape[0]<vnum: f.write('\n')
        else:f.write(' '+str(normal_[i][0])+' '+str(normal_[i][1])+' '+str(normal_[i][2])+'\n')
    f.close()
    return None
def read_xyz_(xyz_name):
    v_ = []
    n_ = []
    ff = open(xyz_name)
    lines = ff.readlines()
    for i, aline in enumerate(lines):
        words = aline.split(' ')
        x,y,z = float(words[0]), float(words[1]), float(words[2])
        v_.append([x,y,z])
        if len(words)>=6:
            nx,ny,nz = float(words[3]), float(words[4]), float(words[5])
            n_.append([nx,ny,nz])
    v_ = numpy.array(v_).astype(numpy.float32)
    n_ = numpy.array(n_).astype(numpy.float32)
    if n_.shape[0] < v_.shape[0]:
        n_ = None
    return v_, n_

# format converting
def convert_obj_to_off_(obj_path_in, off_path_out):
    v,face_,_ = read_obj_(obj_path_in)
    write_off_(off_path_out, v, face_)
    return None
    
# normalize the points to sphere
def normalize_points_to_sphere_(v_in):
    v_out = v_in.copy()
    center = numpy.mean(v_out,axis=0,keepdims=True)
    v_out = v_out-center
    factor = numpy.sum(v_out*v_out, axis=-1, keepdims=True).max()**0.5
    v_out /= factor
    return v_out, center, factor

# normalize the points to sphere with given center and factor
def normalize_points_to_sphere_with_given_center_and_factor_(v_in, center, factor):
    v_out = v_in.copy()
    v_out = v_out-center
    v_out /= factor
    return v_out, center, factor
                
