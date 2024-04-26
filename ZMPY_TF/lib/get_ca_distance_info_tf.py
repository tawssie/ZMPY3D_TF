# GPL3.0 License, JSLai
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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def get_ca_distance_info_tf(xyz):
    # xyz == Nx3 tf matrix
    xyz_center = tf.reduce_mean(xyz, axis=0)
    xyz_diff = xyz - xyz_center
    xyz_dist2center = tf.sqrt(tf.reduce_sum(tf.square(xyz_diff), axis=1))

    percentiles_for_geom = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    prctile_list = tfp.stats.percentile(xyz_dist2center, percentiles_for_geom, interpolation='linear') 
    prctile_list = tf.reshape(prctile_list, [-1, 1]) 

    std_xyz_dist2center = tf.math.reduce_std(xyz_dist2center)


    n = tf.shape(xyz_dist2center)

    mean_distance = tf.math.reduce_mean(xyz_dist2center)
    std_xyz_dist2center = tf.math.reduce_std(xyz_dist2center) * tf.sqrt(n / (n - 1))
    s = (n / ((n - 1) * (n - 2))) * tf.math.reduce_sum(((xyz_dist2center - mean_distance) / std_xyz_dist2center) ** 3)

    fourth_moment = tf.math.reduce_sum(((xyz_dist2center - mean_distance) / std_xyz_dist2center) ** 4)

    k = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * fourth_moment -
         3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    return prctile_list, std_xyz_dist2center, s, k
