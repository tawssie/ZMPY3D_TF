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
def ZMCal(Voxel3D,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear,MaxOrder,s_id,n,l,m,mu,k,IsNLM_Value):
    
    Dimension_BBox_scaled=tf.shape(Voxel3D)
    X_sample = tf.range(Dimension_BBox_scaled[0] + 1,dtype=tf.float64)
    Y_sample = tf.range(Dimension_BBox_scaled[1] + 1,dtype=tf.float64)
    Z_sample = tf.range(Dimension_BBox_scaled[2] + 1,dtype=tf.float64)
    
    [VolumeMass,Center,_]=z.calculate_bbox_moment(Voxel3D,1,X_sample,Y_sample,Z_sample)

    [AverageVoxelDist2Center,MaxVoxelDist2Center]=z.calculate_molecular_radius(Voxel3D,Center,VolumeMass,1.80) # Param['default_radius_multiplier'] == 1.80

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

    return MomentInvariant, AverageVoxelDist2Center


@tf.function
def GeoCal(XYZ,AA_NameList,Param,AverageVoxelDist2Center):
    TotalResidueWeight=z.get_total_residue_weight(AA_NameList,Param['residue_weight_map'])
    TotalResidueWeight=tf.convert_to_tensor(TotalResidueWeight,dtype=tf.float64)
    [Prctile_list,STD_XYZ_dist2center,S,K]=z.get_ca_distance_info(tf.convert_to_tensor(XYZ))
    GeoDescriptor = tf.concat([tf.reshape(AverageVoxelDist2Center,[-1])
                               , tf.reshape(TotalResidueWeight,[-1])
                               , tf.reshape(Prctile_list,[-1])
                               , STD_XYZ_dist2center, S, K], axis=0)        
    return GeoDescriptor


def ZMPY_TF_CLI_ShapeScore(PDBFileNameA, PDBFileNameB,GridWidth):

    Param=z.get_global_parameter()
    
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
    
    ResidueBox=z.get_residue_gaussian_density_cache(Param)

    [XYZ_A,AA_NameList_A]=z.get_pdb_xyz_ca(PDBFileNameA)
    [Voxel3D_A,_]=z.fill_voxel_by_weight_density(XYZ_A,AA_NameList_A,Param['residue_weight_map'],GridWidth,ResidueBox[GridWidth])
    Voxel3D_A=tf.convert_to_tensor(Voxel3D_A,dtype=tf.float64)

    [XYZ_B,AA_NameList_B]=z.get_pdb_xyz_ca(PDBFileNameB)
    [Voxel3D_B,_]=z.fill_voxel_by_weight_density(XYZ_B,AA_NameList_B,Param['residue_weight_map'],GridWidth,ResidueBox[GridWidth])
    Voxel3D_B=tf.convert_to_tensor(Voxel3D_B,dtype=tf.float64)

    MomentInvariantRawA, AverageVoxelDist2CenterA = ZMCal(Voxel3D_A,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear,MaxOrder,s_id,n,l,m,mu,k,IsNLM_Value)
    MomentInvariantRawB, AverageVoxelDist2CenterB = ZMCal(Voxel3D_B,BinomialCache, CLMCache, CLMCache3D, GCache_complex, GCache_complex_index, GCache_pqr_linear,MaxOrder,s_id,n,l,m,mu,k,IsNLM_Value)

    GeoDescriptorA=GeoCal(XYZ_A,AA_NameList_A,Param,AverageVoxelDist2CenterA)
    GeoDescriptorB=GeoCal(XYZ_B,AA_NameList_B,Param,AverageVoxelDist2CenterB)

    P=z.get_descriptor_property()

    ZMIndex=tf.concat(
        [ tf.convert_to_tensor(P['ZMIndex0'])
         ,tf.convert_to_tensor(P['ZMIndex1'])
         ,tf.convert_to_tensor(P['ZMIndex2'])
         ,tf.convert_to_tensor(P['ZMIndex3'])
         ,tf.convert_to_tensor(P['ZMIndex4'])],axis=0)

    ZMWeight=tf.concat(
        [ tf.convert_to_tensor(P['ZMWeight0'])
         ,tf.convert_to_tensor(P['ZMWeight1'])
         ,tf.convert_to_tensor(P['ZMWeight2'])
         ,tf.convert_to_tensor(P['ZMWeight3'])
         ,tf.convert_to_tensor(P['ZMWeight4'])],axis=0)

    # Calculating ZMScore
    ZMScore = tf.reduce_sum(tf.abs(tf.gather(MomentInvariantRawA, ZMIndex) - tf.gather(MomentInvariantRawB, ZMIndex)) * ZMWeight)

    # Calculating GeoScore
    GeoScore = tf.reduce_sum( tf.reshape(tf.convert_to_tensor(P['GeoWeight'],dtype=tf.float64),[-1])* (2 * tf.abs(GeoDescriptorA - GeoDescriptorB) / (1 + tf.abs(GeoDescriptorA) + tf.abs(GeoDescriptorB))))

    # # Calculating paper loss
    # Paper_Loss = ZMScore + GeoScore
    
    # # Scaled scores, fitted to shape service
    GeoScoreScaled = (6.6 - GeoScore) / 6.6 * 100.0
    ZMScoreScaled = (9.0 - ZMScore) / 9.0 * 100.0

    return GeoScoreScaled, ZMScoreScaled


def main():
    if len(sys.argv) != 4:
        print('Usage: ZMPY_TF_CLI_ShapeScore PDB_A PDB_B GridWidth')
        print('    This function takes two PDB structures and a grid width to generate shape analysis scores.')
        print('Error: You must provide exactly two input files and an input grid width.')
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Process two .pdb or .txt files.')
    parser.add_argument('input_file1', type=str, help='The first input file to process (must end with .pdb or .txt) with old PDB text format')
    parser.add_argument('input_file2', type=str, help='The second input file to process (must end with .pdb or .txt) with old PDB text format')
    parser.add_argument('grid_width', type=str, help='The third input file to process (must 0.25, 0.50 or 1.0)')

    args = parser.parse_args()

    # Perform validation checks directly after parsing arguments
    input_files = [args.input_file1, args.input_file2]
    for input_file in input_files:
        if not (input_file.endswith('.pdb') or input_file.endswith('.txt')):
            parser.error("File must end with .pdb or .txt")
        
        if not os.path.isfile(input_file):
            parser.error("File does not exist")

    try:
        GridWidth = float(args.grid_width)
    except ValueError:
        print("GridWidth cannot be converted to a float.")   

    if GridWidth not in [0.25, 0.50, 1.0]:
        parser.error("grid width must be either 0.25, 0.50, or 1.0")

    GeoScoreScaled, ZMScoreScaled=ZMPY_TF_CLI_ShapeScore(args.input_file1,args.input_file2,GridWidth)

    print('The scaled score for the geometric descriptor is calculated.')
    print(f'GeoScore {GeoScoreScaled:.2f}')

    print('The scaled score for the Zernike moments is calculated.')
    print(f'TotalZMScore {ZMScoreScaled:.2f}')

if __name__ == '__main__':
    main()