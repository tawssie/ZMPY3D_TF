# GPL3.0 License, JSL from ZM, this is my originality, to calculate 3D bbox moment by tensordot
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
import tensorflow.experimental.numpy as tnp

@tf.function
def calculate_bbox_moment_07_tf(voxel3d, max_order, x_sample, y_sample, z_sample):
    # When MaxOrder is fixed, p, q, r do not need to be recalculated
    p, q, r = tf.meshgrid(tf.range(1, max_order + 2, dtype=tf.float64), 
                          tf.range(1, max_order + 2, dtype=tf.float64), 
                          tf.range(1, max_order + 2, dtype=tf.float64), indexing='ij')

    extend_voxel3d = tf.pad(voxel3d, [[0, 1], [0, 1], [0, 1]])

    diff_extend_voxel3d = tnp.diff(tnp.diff(tnp.diff(extend_voxel3d, axis=0), axis=1), axis=2)

    x_sample = x_sample[1:, tf.newaxis]
    xpower = tf.pow(x_sample, tf.range(1, max_order + 2, dtype=tf.float64))

    y_sample = y_sample[1:, tf.newaxis]
    ypower = tf.pow(y_sample, tf.range(1, max_order + 2, dtype=tf.float64))

    z_sample = z_sample[1:, tf.newaxis]
    zpower = tf.pow(z_sample, tf.range(1, max_order + 2, dtype=tf.float64))

    bbox_moment = tf.tensordot(zpower, tf.tensordot(ypower, tf.tensordot(xpower, diff_extend_voxel3d, axes=([0], [0])), axes=([0], [1])), axes=([0], [2]))
    bbox_moment = -tf.transpose(bbox_moment, (2, 1, 0)) / p / q / r
    
    volume_mass = bbox_moment[0, 0, 0]
    center = [bbox_moment[1, 0, 0], bbox_moment[0, 1, 0], bbox_moment[0, 0, 1]] / volume_mass

    return volume_mass, center, bbox_moment
