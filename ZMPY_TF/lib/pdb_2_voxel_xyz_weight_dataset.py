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



import tensorflow as tf

from .get_global_parameter02 import *
from .get_residue_gaussian_density_cache02 import *
from .get_pdb_xyz_ca02 import *
from .fill_voxel_by_weight_density04 import *
from .get_total_residue_weight import *

class pdb_2_voxel_xyz_weight_dataset(tf.data.Dataset):
    def __new__(cls, pdb_file_name_list, grid_width=1.0):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=None, dtype=tf.float64),
                tf.TensorSpec(shape=None, dtype=tf.float64),
                tf.TensorSpec(shape=None, dtype=tf.float64),
                tf.TensorSpec(shape=None, dtype=tf.float64)
            ),
            args=(pdb_file_name_list, grid_width))

    @staticmethod
    def _generator(pdb_file_name_list, grid_width):
        param = get_global_parameter02()
        rg_cache = get_residue_gaussian_density_cache02(param)
        residue_box = rg_cache[grid_width]
        residue_weight_map = param['residue_weight_map']

        for file in pdb_file_name_list:
            [xyz, aa_name_list] = get_pdb_xyz_ca02(file)
            [voxel3d, corner] = fill_voxel_by_weight_density04(xyz, aa_name_list, residue_weight_map, grid_width, residue_box)

            voxel3d_tensor = tf.convert_to_tensor(voxel3d, tf.float64)
            corner_tensor = tf.convert_to_tensor(corner, tf.float64)

            total_residue_weight = get_total_residue_weight(aa_name_list, residue_weight_map)
            total_residue_weight_tensor = tf.convert_to_tensor(total_residue_weight, dtype=tf.float64)
            xyz_tensor = tf.convert_to_tensor(xyz, tf.float64)

            yield voxel3d_tensor, xyz_tensor, total_residue_weight_tensor, corner_tensor



