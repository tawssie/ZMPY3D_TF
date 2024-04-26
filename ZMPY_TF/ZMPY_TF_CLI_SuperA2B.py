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

@tf.function
def ZMCal(Voxel3D,Corner,GridWidth,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear,MaxOrder,s_id,n,l,m,mu,k,IsNLM_Value):

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


def ZMPY_TF_CLI_SuperA2B(PDBFileNameA, PDBFileNameB):
        
    Param=z.get_global_parameter()

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

    ResidueBox=z.get_residue_gaussian_density_cache(Param)

    [XYZ_A,AA_NameList_A]=z.get_pdb_xyz_ca(PDBFileNameA)
    [Voxel3D_A,Corner_A]=z.fill_voxel_by_weight_density(XYZ_A,AA_NameList_A,Param['residue_weight_map'],GridWidth,ResidueBox[GridWidth])
    Voxel3D_A=tf.convert_to_tensor(Voxel3D_A,dtype=tf.float64)

    [XYZ_B,AA_NameList_B]=z.get_pdb_xyz_ca(PDBFileNameB)
    [Voxel3D_B,Corner_B]=z.fill_voxel_by_weight_density(XYZ_B,AA_NameList_B,Param['residue_weight_map'],GridWidth,ResidueBox[GridWidth])
    Voxel3D_B=tf.convert_to_tensor(Voxel3D_B,dtype=tf.float64)

    Center_scaled_A,ABList_A,ZMList_A=ZMCal(Voxel3D_A,Corner_A,GridWidth,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear, MaxOrder,s_id,n,l,m,mu,k,IsNLM_Value)
    Center_scaled_B,ABList_B,ZMList_B=ZMCal(Voxel3D_B,Corner_B,GridWidth,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear, MaxOrder,s_id,n,l,m,mu,k,IsNLM_Value)

    M = tf.math.abs(tf.linalg.matmul(tf.transpose(tf.math.conj(ZMList_A)), ZMList_B)) # square matrix A^T*B 
    MaxValueIndex = tf.where(M == tf.math.reduce_max(M))  # MaxValueIndex is a tuple that contains an nd array.
    
    i,j=MaxValueIndex[0,0], MaxValueIndex[0,1]
    
    RotM_A=z.get_transform_matrix_from_ab_list(ABList_A[i,0],ABList_A[i,1],Center_scaled_A)
    RotM_B=z.get_transform_matrix_from_ab_list(ABList_B[j,0],ABList_B[j,1],Center_scaled_B)
    
    TargetRotM = tf.linalg.solve(RotM_B, RotM_A)
        
    return TargetRotM


def main():
    if len(sys.argv) != 3:
        print('Usage: ZMPY_TF_CLI_SuperA2B PDB_A PDB_B')
        print('    This function generates a transformation matrix to superimpose structure A onto B, i.e., the matrix is for A’s coordinates.')
        print('Error: You must provide exactly two input files.')
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Process two .pdb or .txt files.')
    parser.add_argument('input_file1', type=str, help='The first input file to process (must end with .pdb or .txt) with old PDB text format')
    parser.add_argument('input_file2', type=str, help='The second input file to process (must end with .pdb or .txt) with old PDB text format')

    args = parser.parse_args()

    # Perform validation checks directly after parsing arguments
    input_files = [args.input_file1, args.input_file2]
    for input_file in input_files:
        if not (input_file.endswith('.pdb') or input_file.endswith('.txt')):
            parser.error("File must end with .pdb or .txt")
        
        if not os.path.isfile(input_file):
            parser.error("File does not exist")

    TargetRotM=ZMPY_TF_CLI_SuperA2B(args.input_file1,args.input_file2)

    print('the matrix is for A’s coordinates.')
    print(TargetRotM)

if __name__ == '__main__':
    main()