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

@tf.function
def eigen_root_tf2(poly_coefficient_list):

    num_of_rows = tf.shape(poly_coefficient_list)[0]
    num_of_cols = tf.shape(poly_coefficient_list)[1]

    companion_matrix = tf.eye(num_rows=num_of_cols - 2, num_columns=num_of_cols - 1,
                              batch_shape=[num_of_rows], dtype=tf.complex128)

    col_1st = tf.reshape(poly_coefficient_list[:, 0], [-1, 1])
    row_1st = -poly_coefficient_list[:, 1:] / col_1st

    row_1st = tf.expand_dims(row_1st, axis=1)
    
    full_matrix = tf.concat([row_1st, companion_matrix], axis=1)

    return tf.reverse(tf.linalg.eigvals(full_matrix), axis=[1])
