# GPL 3.0 License, JSL from BIOZ
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
def get_3dzd_121_descriptor_02_tf(zmoment_scaled):
    zmoment_scaled = tf.where(
        tf.math.is_nan(tf.math.real(zmoment_scaled)), 
        tf.zeros_like(zmoment_scaled), 
        zmoment_scaled
    )
    
    zmoment_scaled_norm = tf.math.abs(zmoment_scaled) ** 2

    zmoment_scaled_norm_positive = tf.reduce_sum(zmoment_scaled_norm, axis=2)

    zero_matrix = tf.zeros_like(zmoment_scaled_norm[:, :, 0:1])
    part_matrix = zmoment_scaled_norm[:, :, 1:]
    zmoment_scaled_norm = tf.concat([zero_matrix, part_matrix], axis=2)

    zmoment_scaled_norm_negative = tf.reduce_sum(zmoment_scaled_norm, axis=2)

    zm_3dzd_invariant = tf.sqrt(zmoment_scaled_norm_positive + zmoment_scaled_norm_negative)

    zm_3dzd_invariant = tf.where(
        zm_3dzd_invariant < 1e-20, 
        tf.fill(zm_3dzd_invariant.shape, tf.constant(np.nan, dtype=tf.float64)), 
        zm_3dzd_invariant
    )

    return zm_3dzd_invariant
