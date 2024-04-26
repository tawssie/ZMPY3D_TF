# MIT License
#
# Copyright (c) 2024 Jhih-Siang Lai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf
import pickle
import argparse
import os
import sys

import ZMPY_TF as z

# Full procedure to calculate superposition for a single graph component.
@tf.function
def core(Voxel3D,Corner,GridWidth,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear,MaxOrder,s_id,n,l,m,mu,k,IsNLM_Value):

    Dimension_BBox_scaled=tf.shape(Voxel3D)
    X_sample = tf.range(Dimension_BBox_scaled[0] + 1,dtype=tf.float64)
    Y_sample = tf.range(Dimension_BBox_scaled[1] + 1,dtype=tf.float64)
    Z_sample = tf.range(Dimension_BBox_scaled[2] + 1,dtype=tf.float64)
    
    [VolumeMass,Center,_]=z.calculate_bbox_moment(Voxel3D,1,X_sample,Y_sample,Z_sample)

    [AverageVoxelDist2Center,MaxVoxelDist2Center]=z.calculate_molecular_radius(Voxel3D,Center,VolumeMass,1.80) # Param['default_radius_multiplier'] == 1.80

    Center_scaled=Center*GridWidth+Corner
    
    ##################################################################################
    # You may add any preprocessing on the voxel before applying the Zernike moment. #
    ##################################################################################
                
    Sphere_X_sample, Sphere_Y_sample, Sphere_Z_sample=z.get_bbox_moment_xyz_sample(Center,AverageVoxelDist2Center,Dimension_BBox_scaled)
    
    _,_,SphereBBoxMoment=z.calculate_bbox_moment(Voxel3D
                                      ,MaxOrder
                                      ,Sphere_X_sample
                                      ,Sphere_Y_sample
                                      ,Sphere_Z_sample)
    
    ZMoment_scaled,ZMoment_raw=z.calculate_bbox_moment_2_zm(MaxOrder
                                       , GCache_complex
                                       , GCache_pqr_linear
                                       , GCache_complex_index
                                       , CLMCache3D
                                       , SphereBBoxMoment)

    ABList_2=z.calculate_ab_rotation_all(ZMoment_raw, 2)
    ABList_3=z.calculate_ab_rotation_all(ZMoment_raw, 3)
    ABList_4=z.calculate_ab_rotation_all(ZMoment_raw, 4)
    ABList_5=z.calculate_ab_rotation_all(ZMoment_raw, 5)
    ABList_6=z.calculate_ab_rotation_all(ZMoment_raw, 6)

    ABList_all=tf.concat([tf.reshape(ABList_2,[-1,2]),tf.reshape(ABList_3,[-1,2]),tf.reshape(ABList_4,[-1,2]),tf.reshape(ABList_5,[-1,2]),tf.reshape(ABList_6,[-1,2])],axis=0)
    ZMList_all=z.calculate_zm_by_ab_rotation(ZMoment_raw, BinomialCache, ABList_all, MaxOrder, CLMCache,s_id,n,l,m,mu,k,IsNLM_Value)
    ZMList_all = tf.boolean_mask(ZMList_all, ~tf.math.is_nan(tf.math.real(ZMList_all)))

    # Based on ABList_all, it is known in advance that Order 6 will definitely have 96 pairs of AB, which means 96 vectors.
    ZMList_all = tf.reshape(ZMList_all, [-1,96])

    return Center_scaled, ABList_all,ZMList_all

@tf.function
def CalMatrix(data1, data2):
    Center_scaled_A=data1[0]
    ABList_A=data1[1]
    ZMList_A=data1[2]

    Center_scaled_B=data2[0]
    ABList_B=data2[1]
    ZMList_B=data2[2]

    M = tf.math.abs(tf.linalg.matmul(tf.transpose(tf.math.conj(ZMList_A)), ZMList_B)) # square matrix A^T*B 
    MaxValueIndex = tf.where(M == tf.math.reduce_max(M))  # MaxValueIndex is a tuple that contains an nd array.
    
    i,j=MaxValueIndex[0,0], MaxValueIndex[0,1]
    
    RotM_A=z.get_transform_matrix_from_ab_list(ABList_A[i,0],ABList_A[i,1],Center_scaled_A)
    RotM_B=z.get_transform_matrix_from_ab_list(ABList_B[j,0],ABList_B[j,1],Center_scaled_B)
    
    TargetRotM = tf.linalg.solve(RotM_B, RotM_A)

    return TargetRotM


def ZMPY_TF_CLI_BatchSuperA2B(PDBFileNameA, PDBFileNameB):
    
    MaxOrder=int(6) 
    GridWidth= 1.00

    BinomialCacheFilePath = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_data'), 'BinomialCache.pkl')
    with open(BinomialCacheFilePath, 'rb') as file: # Used at the entry point, it requires __file__ to identify the package location
    # with open('./cache_data/BinomialCache.pkl', 'rb') as file: # Can be used in ipynb, but not at the entry point. 
        BinomialCachePKL = pickle.load(file)

    LogCacheFilePath=os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_data'), 'LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder))
    with open(LogCacheFilePath, 'rb') as file: # Used at the entry point, it requires __file__ to identify the package location
    # with open('./cache_data/LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder), 'rb') as file: # Can be used in ipynb, but not at the entry point. 
        CachePKL = pickle.load(file)  

    # Extract all cached variables from pickle. These will be converted into a tensor/cupy objects for ZMPY_CP and ZMPY_TF.
    BinomialCache= tf.convert_to_tensor(BinomialCachePKL['BinomialCache'], dtype=tf.float64)
    
    # GCache, CLMCache, and all RotationIndex
    GCache_pqr_linear= tf.convert_to_tensor(CachePKL['GCache_pqr_linear'])
    GCache_complex= tf.convert_to_tensor(CachePKL['GCache_complex'])
    GCache_complex_index= tf.convert_to_tensor(CachePKL['GCache_complex_index'])
    CLMCache3D= tf.convert_to_tensor(CachePKL['CLMCache3D'],dtype=tf.complex128)
    CLMCache= tf.convert_to_tensor(CachePKL['CLMCache'], dtype=tf.float64)
    
    RotationIndex=CachePKL['RotationIndex']

    # RotationIndex is a structure, must be [0,0] to accurately obtain the s_id ... etc, within RotationIndex.
    s_id=tf.convert_to_tensor(np.squeeze(RotationIndex['s_id'][0,0])-1, dtype=tf.int64)
    n   =tf.convert_to_tensor(np.squeeze(RotationIndex['n'] [0,0]), dtype=tf.int64)
    l   =tf.convert_to_tensor(np.squeeze(RotationIndex['l'] [0,0]), dtype=tf.int64)
    m   =tf.convert_to_tensor(np.squeeze(RotationIndex['m'] [0,0]), dtype=tf.int64)
    mu  =tf.convert_to_tensor(np.squeeze(RotationIndex['mu'][0,0]), dtype=tf.int64)
    k   =tf.convert_to_tensor(np.squeeze(RotationIndex['k'] [0,0]), dtype=tf.int64)
    IsNLM_Value=tf.convert_to_tensor(np.squeeze(RotationIndex['IsNLM_Value'][0,0])-1, dtype=tf.int64)
    
    VD_A = (z.pdb_2_voxel_xyz_weight_dataset(PDBFileNameA, GridWidth)
           .map(lambda v, xyz, w, c: core(v, c, GridWidth, BinomialCache, CLMCache, CLMCache3D, 
                                          GCache_complex, GCache_complex_index, 
                                          GCache_pqr_linear, MaxOrder, s_id, n, l, m, mu, 
                                          k, IsNLM_Value),
                num_parallel_calls=tf.data.AUTOTUNE)
           .prefetch(10))

    VD_B = (z.pdb_2_voxel_xyz_weight_dataset(PDBFileNameB, GridWidth)
           .map(lambda v, xyz, w, c: core(v, c, GridWidth, BinomialCache, CLMCache, CLMCache3D, 
                                          GCache_complex, GCache_complex_index, 
                                          GCache_pqr_linear, MaxOrder, s_id, n, l, m, mu, 
                                          k, IsNLM_Value),
                num_parallel_calls=tf.data.AUTOTUNE)
           .prefetch(10))

    VD_AB = tf.data.Dataset.zip((VD_A, VD_B)).map( CalMatrix  ,num_parallel_calls=tf.data.AUTOTUNE)

    # TargetRotMList=[]
    # for m in VD_AB:
    #     TargetRotMList.append(m)

    TargetRotM_tensor = zip(*VD_AB)
    TargetRotMList = tf.stack(list(TargetRotM_tensor),axis=1)

    return TargetRotMList



def main():
    if len(sys.argv) != 2:
        print('Usage: ZMPY_TF_CLI_BatchSuperA2B PDBFileList.txt')
        print('       This function takes a list of paired PDB structure file paths to generate transformation matrices.')
        print("Error: You must provide exactly one input file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Process input file that contains paths to .pdb or .txt files.')
    parser.add_argument('input_file', type=str, help='The input file that contains paths to .pdb or .txt files.')

    args = parser.parse_args()

    input_file = args.input_file
    if not input_file.endswith('.txt'):
        parser.error("File must end with .txt")
    
    if not os.path.isfile(input_file):
        parser.error("File does not exist")

    with open(input_file, 'r') as file:
        lines = file.readlines()

    file_list_1 = []
    file_list_2 = []
    for line in lines:

        files = line.strip().split()
        if len(files) != 2:
            print(f"Error: Each line must contain exactly two file paths, but got {len(files)}.")
            sys.exit(1)
        file1, file2 = files
        
        for file in [file1, file2]:
            if not (file.endswith('.pdb') or file.endswith('.txt')):
                print(f"Error: File {file} must end with .pdb or .txt.")
                sys.exit(1)
            if not os.path.isfile(file):
                print(f"Error: File {file} does not exist.")
                sys.exit(1)
        file_list_1.append(file1)
        file_list_2.append(file2)

    TargetRotM=ZMPY_TF_CLI_BatchSuperA2B(file_list_1, file_list_2)

    for M in TargetRotM:
        print(M)

if __name__ == '__main__':
    main()