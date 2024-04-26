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
import numpy as np

@tf.function
def tf_complex_nan():
    c = np.zeros(1, dtype=np.complex128)
    c[0] = np.nan
    return tf.convert_to_tensor(c, dtype=tf.complex128)

@tf.function
def calculate_bbox_moment_2_zm05_tf(max_order, gcache_complex, gcache_pqr_linear, gcache_complex_index, clm_cache3d, bbox_moment):
    max_n = max_order + 1

    bbox_moment = tf.reshape(tf.transpose(bbox_moment, perm=[2, 1, 0]), [-1])
    bbox_moment = tf.cast(bbox_moment, tf.complex128)
    
    zm_geo = gcache_complex * tf.gather(bbox_moment, gcache_pqr_linear - 1)

    zm_geo_sum = tf.zeros([max_n * max_n * max_n, 1], dtype=tf.complex128)
    zm_geo_sum = tf.tensor_scatter_nd_add(zm_geo_sum, gcache_complex_index - 1, zm_geo)
    
    zm_geo_sum = tf.where(zm_geo_sum == 0.0, tf_complex_nan(), zm_geo_sum)
    
    zmoment_raw = zm_geo_sum * (3.0 / (4.0 * np.pi))
    zmoment_raw = tf.reshape(zmoment_raw, [max_n, max_n, max_n])
    zmoment_raw = tf.transpose(zmoment_raw, perm=[2, 1, 0])

    zmoment_scaled = zmoment_raw * clm_cache3d

    return zmoment_scaled, zmoment_raw

