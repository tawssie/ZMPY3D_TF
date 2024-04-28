# GPL3.0 License, JSL
#
# Copyright (C) 2024  Jhih-Siang Lai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import tensorflow as tf

@tf.function
def calculate_molecular_radius_04_tf(voxel3d, center, volume_mass, default_radius_multiplier):
    has_weight = voxel3d > 0

    has_weight.set_shape([None, None, None])

    voxel_list = tf.boolean_mask(voxel3d, has_weight)

    voxel_list_xyz = tf.cast(tf.where(has_weight), dtype=tf.float64)

    voxel_dist2center_squared = tf.reduce_sum(tf.square(voxel_list_xyz - center), axis=1)

    average_voxel_mass2center_squared = tf.reduce_sum(voxel_dist2center_squared * voxel_list) / volume_mass

    average_voxel_dist2center = tf.sqrt(average_voxel_mass2center_squared) * default_radius_multiplier
    max_voxel_dist2center = tf.sqrt(tf.reduce_max(voxel_dist2center_squared))

    return average_voxel_dist2center, max_voxel_dist2center
