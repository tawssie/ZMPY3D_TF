# GPL3.0 License, JSL, ZM
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
def calculate_zm_by_ab_rotation01_tf(z_moment_raw, binomial_cache, ab_list, max_order, clm_cache, s_id, n, l, m, mu, k, is_nlm_value):

    a = ab_list[:, 0]
    b = ab_list[:, 1]


    aac = tf.cast(tf.math.real(a * tf.math.conj(a)), tf.complex128)
    bbc = tf.cast(tf.math.real(b * tf.math.conj(b)), tf.complex128)
    bbcaac = -bbc / aac
    abc = -(a / tf.math.conj(b))
    ab = a / b
    
    bbcaac_pow_k_list = tf.math.log(bbcaac)[:, tf.newaxis] * tf.cast(tf.range(max_order + 1), dtype=tf.complex128)
    aac_pow_l_list = tf.math.log(aac)[:, tf.newaxis] * tf.cast(tf.range(max_order + 1), dtype=tf.complex128)
    ab_pow_m_list = tf.math.log(ab)[:, tf.newaxis] * tf.cast(tf.range(max_order + 1), dtype=tf.complex128)
    abc_pow_mu_list = tf.math.log(abc)[:, tf.newaxis] * tf.cast(tf.range(-max_order, max_order + 1), dtype=tf.complex128)
    
    f_exp = tf.zeros(tf.shape(s_id), dtype=tf.complex128)
    

    cond1 = mu >= 0
    cond2 = (mu < 0) & (tf.math.mod(mu, 2) == 0)
    cond3 = (mu < 0) & (tf.math.mod(mu, 2) != 0)
    
    indices1 = tf.boolean_mask(tf.stack([n, l, mu], axis=1), cond1)
    indices2 = tf.boolean_mask(tf.stack([n, l, -mu], axis=1), cond2)
    indices3 = tf.boolean_mask(tf.stack([n, l, -mu], axis=1), cond3)
    
    f_exp_values1 = tf.gather_nd(z_moment_raw, indices1)
    f_exp_values2 = tf.math.conj(tf.gather_nd(z_moment_raw, indices2))
    f_exp_values3 = -tf.math.conj(tf.gather_nd(z_moment_raw, indices3))
    
    f_exp = tf.tensor_scatter_nd_update(f_exp, tf.where(cond1), f_exp_values1)
    f_exp = tf.tensor_scatter_nd_update(f_exp, tf.where(cond2), f_exp_values2)
    f_exp = tf.tensor_scatter_nd_update(f_exp, tf.where(cond3), f_exp_values3)
    

    f_exp = tf.math.log(f_exp)

    max_n = max_order + 1
    clm = tf.gather(clm_cache, l * max_n + m)
    clm = tf.cast(clm, tf.complex128)
    clm = tf.reshape(clm, [-1])


    indices_bin1 = tf.stack([l - mu, k - mu], axis=1)
    indices_bin2 = tf.stack([l + mu, k - m], axis=1)
    
    bin_part1 = tf.gather_nd(binomial_cache, indices_bin1)
    bin_part2 = tf.gather_nd(binomial_cache, indices_bin2)
    
    bin = tf.cast(bin_part1, tf.complex128) + tf.cast(bin_part2, tf.complex128)

    al = tf.gather(aac_pow_l_list, l, axis=1)
    abpm = tf.gather(ab_pow_m_list, m, axis=1)
    amu = tf.gather(abc_pow_mu_list, max_order + mu, axis=1)
    bbk = tf.gather(bbcaac_pow_k_list, k, axis=1)

    nlm = f_exp + clm + bin + al + abpm + amu + bbk

    exp_nlm = tf.exp(nlm)
    z_nlm = tf.zeros(is_nlm_value.shape, dtype=tf.complex128)
    z_nlm = tf.reshape(z_nlm, [-1, 1])
    z_nlm = tf.tile(z_nlm, [1, tf.size(a)])
    exp_nlm = tf.transpose(exp_nlm)
    z_nlm = tf.tensor_scatter_nd_add(z_nlm, tf.expand_dims(s_id, 1), exp_nlm)


    zm = tf.fill([tf.reduce_prod(tf.shape(z_moment_raw)), tf.size(a)], tf.cast(np.nan, dtype=tf.complex128))
    zm = tf.tensor_scatter_nd_update(zm, tf.reshape(is_nlm_value, [-1, 1]), z_nlm)
    num_of_ab = tf.shape(ab_list)[0]
    zm = tf.reshape(zm, [max_n, max_n, max_n, num_of_ab])
    zm = tf.transpose(zm, (2, 1, 0, 3))

    return zm
