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

# Full procedure to calculate ZM and moment in a single graph component
@tf.function
def core(Voxel3D,XYZ,TotalResidueWeight, MaxOrder, BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear,s_id,n,l,m,mu,k,IsNLM_Value):

    Dimension_BBox_scaled=tf.shape(Voxel3D)
    X_sample = tf.range(Dimension_BBox_scaled[0] + 1,dtype=tf.float64)
    Y_sample = tf.range(Dimension_BBox_scaled[1] + 1,dtype=tf.float64)
    Z_sample = tf.range(Dimension_BBox_scaled[2] + 1,dtype=tf.float64)

    [VolumeMass,Center,_]=z.calculate_bbox_moment(Voxel3D,1,X_sample,Y_sample,Z_sample)

    [AverageVoxelDist2Center,_]=z.calculate_molecular_radius(Voxel3D,Center,VolumeMass,1.80) # Param['DEFAULT_RADIUS_MULTIPLIER'] == 1.80

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


    ZMList = []
    ZMList.append(z.get_3dzd_121_descriptor(ZMoment_scaled))

    MaxTargetOrder2NormRotate=5

    for TargetOrder2NormRotate in range(2, MaxTargetOrder2NormRotate+1):
        ABList=z.calculate_ab_rotation(ZMoment_raw, TargetOrder2NormRotate)
        ZM=z.calculate_zm_by_ab_rotation(ZMoment_raw, BinomialCache, ABList, MaxOrder, CLMCache,s_id,n,l,m,mu,k,IsNLM_Value)
        ZM_mean, _ = z.get_mean_invariant(ZM)
        ZMList.append(ZM_mean)

    MomentInvariant = tf.concat([tf.reshape(tf.boolean_mask(z, ~tf.math.is_nan(z)), [-1]) for z in ZMList], axis=0)

    [Prctile_list,STD_XYZ_dist2center,S,K]=z.get_ca_distance_info(XYZ)
    
    GeoDescriptor = tf.concat([tf.reshape(AverageVoxelDist2Center,[-1])
                               , tf.reshape(TotalResidueWeight,[-1])
                               , tf.reshape(Prctile_list,[-1])
                               , STD_XYZ_dist2center, S, K], axis=0)        

    return MomentInvariant, GeoDescriptor


@tf.function
def CalZM_Geo_ScoreScaled(data1, data2,ZMIndex,ZMWeight,GeoWeight):
    MomentInvariantRawA=data1[0]
    MomentInvariantRawB=data2[0]

    GeoDescriptorA=data1[1]
    GeoDescriptorB=data2[1]

    # Calculating ZMScore
    ZMScore = tf.reduce_sum(tf.abs(tf.gather(MomentInvariantRawA, ZMIndex) - tf.gather(MomentInvariantRawB, ZMIndex)) * ZMWeight)

    # Calculating GeoScore
    GeoScore = tf.reduce_sum( GeoWeight * (2 * tf.abs(GeoDescriptorA - GeoDescriptorB) / (1 + tf.abs(GeoDescriptorA) + tf.abs(GeoDescriptorB))))

    # # Calculating paper loss
    # Paper_Loss = ZMScore + GeoScore
    
    # # Scaled scores, fitted to shape service
    GeoScoreScaled = (6.6 - GeoScore) / 6.6 * 100.0
    ZMScoreScaled = (9.0 - ZMScore) / 9.0 * 100.0

    return GeoScoreScaled, ZMScoreScaled

def ZMPY_TF_CLI_BatchShapeScore(PDBFileNameA, PDBFileNameB,GridWidth):

    MaxOrder=int(20)
    
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

    P=z.get_descriptor_property()

    ZMIndex   = tf.concat([P['ZMIndex0'], P['ZMIndex1'], P['ZMIndex2'], P['ZMIndex3'], P['ZMIndex4']], axis=0)
    ZMWeight  = tf.concat([P['ZMWeight0'], P['ZMWeight1'], P['ZMWeight2'], P['ZMWeight3'], P['ZMWeight4']], axis=0)

    GeoWeight = tf.reshape(tf.convert_to_tensor(P['GeoWeight'],dtype=tf.float64),[-1])


    VD_A = z.pdb_2_voxel_xyz_weight_dataset(PDBFileNameA, GridWidth) \
        .map(
            lambda v, xyz, w, c: core(
                v, xyz, w,
                MaxOrder, BinomialCache, 
                CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, 
                GCache_pqr_linear, s_id, n, l, m, mu, k, IsNLM_Value
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ) \
        .prefetch(10)

    VD_B = z.pdb_2_voxel_xyz_weight_dataset(PDBFileNameB, GridWidth) \
        .map(
            lambda v, xyz, w, c: core(
                v, xyz, w,
                MaxOrder, BinomialCache, 
                CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, 
                GCache_pqr_linear, s_id, n, l, m, mu, k, IsNLM_Value
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ) \
        .prefetch(10)

    VD_AB = tf.data.Dataset.zip((VD_A, VD_B)).map(lambda a,b: CalZM_Geo_ScoreScaled(a,b,ZMIndex,ZMWeight,GeoWeight),num_parallel_calls=tf.data.AUTOTUNE)

    # # Evaluate iteratively, building a list of tensors.
    # GeoScoreScaledList=[]
    # ZMScoreScaledList =[]
    # for geo, zm in VD_AB:
    #     GeoScoreScaledList.append(geo)
    #     ZMScoreScaledList.append(zm)
    # return GeoScoreScaledList, ZMScoreScaledList


    geo_tensor, zm_tensor = zip(*VD_AB)

    GeoScoreScaledTensor = tf.stack(list(geo_tensor))
    ZMScoreScaledTensor = tf.stack(list(zm_tensor))
    return GeoScoreScaledTensor, ZMScoreScaledTensor


def main():
    if len(sys.argv) != 3:
        print('Usage: ZMPY_TF_CLI_BatchShapeScore PDBFileList.txt GridWidth')
        print('       This function takes a list of paired PDB structure file paths and a grid width to generate shape analysis scores.')
        print("Error: You must provide exactly one input file and a grid width.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Process input file that contains paths to .pdb or .txt files and a grid width.')
    parser.add_argument('input_file', type=str, help='The input file that contains paths to .pdb or .txt files.')
    parser.add_argument('grid_width', type=str, help='The grid width (must be 0.25, 0.50 or 1.0)')

    args = parser.parse_args()

    # Perform validation checks directly after parsing arguments
    input_file = args.input_file
    if not input_file.endswith('.txt'):
        parser.error("PDB file list must end with .txt")
    
    if not os.path.isfile(input_file):
        parser.error("File does not exist")

    try:
        GridWidth = float(args.grid_width)
    except ValueError:
        parser.error("GridWidth cannot be converted to a float.")
    
    if GridWidth not in [0.25, 0.50, 1.0]:
        parser.error("grid width must be either 0.25, 0.50, or 1.0")

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

    GeoScoreScaled, ZMScoreScaled=ZMPY_TF_CLI_BatchShapeScore(file_list_1,file_list_2,GridWidth)

    # print('Left, the scaled score for the geometric descriptor.')
    # print('Right, the scaled score for the Zernike moments.')
    for geo_score, zm_score in zip(GeoScoreScaled, ZMScoreScaled):
        print(f'GeoScore {geo_score:.2f} TotalZMScore {zm_score:.2f}')

if __name__ == "__main__":
    main()