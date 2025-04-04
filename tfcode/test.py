import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras


import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util

# Construct graph
x = tf.compat.v1.placeholder(tf.float32, [None, 9, 9, 3])

gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
    h_input='Z2', h_output='D4', in_channels=3, out_channels=64, ksize=3)
w = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
y = gconv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME',
            gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)

gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
    h_input='D4', h_output='D4', in_channels=64, out_channels=64, ksize=3)
w = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
y = gconv2d(input=y, filter=w, strides=[1, 1, 1, 1], padding='SAME',
            gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)

# Compute
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)
y = sess.run(y, feed_dict={x: np.random.randn(10, 9, 9, 3)})
sess.close()

print(y.shape)  # (10, 9, 9, 512)