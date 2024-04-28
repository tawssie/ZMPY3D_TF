# MIT License, BIOZ
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


@tf.function
def get_transform_matrix_from_ab_list02_tf(A, B, center_scaled):
    a2pb2 = tf.math.square(A) + tf.math.square(B)
    a2mb2 = tf.math.square(A) - tf.math.square(B)

    m33_linear = tf.stack([
        tf.math.real(a2pb2),
        -tf.math.imag(a2mb2),
        2 * tf.math.imag(A * B),
        tf.math.imag(a2pb2),
        tf.math.real(a2mb2),
        -2 * tf.math.real(A * B),
        2 * tf.math.imag(A * tf.math.conj(B)),
        2 * tf.math.real(A * tf.math.conj(B)),
        tf.math.real(A * tf.math.conj(A)) - tf.math.real(B * tf.math.conj(B))
    ])

    s = 1.0


    m33 = tf.reshape(m33_linear, [3, 3])

    center_scaled = tf.reshape(center_scaled, [3, -1])  

    m34 = tf.concat([m33, center_scaled], axis=1)  
    m14 = tf.reshape(tf.convert_to_tensor([0, 0, 0, 1], dtype=tf.float64), [-1, 4])  
    m44 = tf.concat([m34, m14], axis=0)

    transform = tf.linalg.inv(m44)

    return transform



