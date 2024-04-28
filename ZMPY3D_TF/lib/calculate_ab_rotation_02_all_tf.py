# GPL3.0, JSL, ZM, bioz
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

from .eigen_root_tf import *
from .eigen_root_tf2 import *


@tf.function
def calculate_ab_rotation_02_all_tf(z_moment_raw, target_order_2_norm_rotate):
    if target_order_2_norm_rotate % 2 == 0:
        abconj_coef = tf.stack([
            z_moment_raw[target_order_2_norm_rotate, 2, 2],
            -z_moment_raw[target_order_2_norm_rotate, 2, 1],
            z_moment_raw[target_order_2_norm_rotate, 2, 0],
            tf.math.conj(z_moment_raw[target_order_2_norm_rotate, 2, 1]),
            tf.math.conj(z_moment_raw[target_order_2_norm_rotate, 2, 2])
        ], 0)

        n_abconj = 4
    else:
        abconj_coef = tf.stack([
            z_moment_raw[target_order_2_norm_rotate, 1, 1],
            -z_moment_raw[target_order_2_norm_rotate, 1, 0],
            -tf.math.conj(z_moment_raw[target_order_2_norm_rotate, 1, 1])
        ], 0)

        n_abconj = 2

    abconj_coef = tf.expand_dims(abconj_coef, 0)
    abconj_sol = eigen_root_tf(abconj_coef)

    @tf.function
    def get_ab_list_by_ind_real(ind_real):
        k_re = tf.math.real(abconj_sol)
        k_im = tf.math.imag(abconj_sol)
        k_im2 = tf.math.square(k_im)
        k_re2 = tf.math.square(k_re)
        k_im3 = k_im * k_im2
        k_im4 = tf.math.square(k_im2)
        k_re4 = tf.math.square(k_re2)

        f20 = tf.math.real(z_moment_raw[ind_real, 2, 0])
        f21 = z_moment_raw[ind_real, 2, 1]
        f22 = z_moment_raw[ind_real, 2, 2]

        f21_im = tf.math.imag(f21)
        f21_re = tf.math.real(f21)
        f22_im = tf.math.imag(f22)
        f22_re = tf.math.real(f22)

        coef4 = (
            4 * f22_re * k_im * (-1 + k_im2 - 3 * k_re2) -
            4 * f22_im * k_re * (1 - 3 * k_im2 + k_re2) -
            2 * f21_re * k_im * k_re * (-3 + k_im2 + k_re2) +
            2 * f20 * k_im * (-1 + k_im2 + k_re2) +
            f21_im * (1 - 6 * k_im2 + k_im4 - k_re4)
        )

        coef3 = (
            2 * (-4 * f22_im * (k_im + k_im3 - 3 * k_im * k_re2) +
            f21_re * (-1 + k_im4 + 6 * k_re2 - k_re4) +
            2 * k_re * (f22_re * (2 + 6 * k_im2 - 2 * k_re2) +
            f21_im * k_im * (-3 + k_im2 + k_re2) +
            f20 * (-1 + k_im2 + k_re2)))
        )

        bimbre_coef = tf.transpose(tf.stack([coef4, coef3, tf.zeros_like(coef4), coef3, -coef4]))

        bimbre_sol_real = tf.math.real(eigen_root_tf2(tf.cast(bimbre_coef, tf.complex128)))

        is_abs_bimre_good = tf.math.abs(bimbre_sol_real) > 1e-7

        bre = 1 / tf.math.sqrt((1 + tf.math.pow(bimbre_sol_real, 2)) * tf.expand_dims((1 + k_im2 + k_re2), 1))
        bim = bimbre_sol_real * bre

        b = tf.complex(bre, bim)
        a = tf.math.conj(b) * tf.expand_dims(abconj_sol, 1)

        ab_list = tf.stack([a[is_abs_bimre_good], b[is_abs_bimre_good]], axis=1)

        return ab_list

    ind_real_all = tf.range(2, tf.shape(z_moment_raw)[0] + 1, delta=2)
    ab_list_all = tf.map_fn(get_ab_list_by_ind_real, ind_real_all, fn_output_signature=tf.complex128)

    return ab_list_all

