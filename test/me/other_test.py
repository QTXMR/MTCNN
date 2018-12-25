#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf

print(tf.Session().run(tf.where(tf.less(-1, 0), 1, 0)))
list_a = [[1, 2, 3, 4], [4, 5, 6, 7]]
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

print(np.array(list_a, dtype=np.float32).reshape(4, -1))

# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
# tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]

# import os
#
# file_path = "E:/tt/abc.py"
# filepath, fullflname = os.path.split(file_path)
# fname, ext = os.path.splitext(fullflname)
#
# print(filepath, fullflname, fname, ext)
#
# list_a = [1.0, 2, 3, 4, 5, 6, 7, 78, 9]
# # print(len(list_a))
# print(np.random.choice(len(list_a), size=2, replace=True))

assert 2 > 1

# with open(os.path.join("lala", 'pos_12.txt'), 'w') as f:
#     print(f.write("Hello"))
#
# with open(os.path.join("lala", 'pos_12.txt'), 'r') as f:
#     a = f.readlines()

# print(a)
# print(f.write("appends"))

# L, R = map(int, input().split())
# sum = 0
# for i in range(L, R + 1):
#     print(i)

import tensorflow as tf

# x = tf.constant([[1., 2., 3.],
#                  [4., 5., 6.]])
#
# x = tf.reshape(x, [1, 2, 3, 1])  # give a shape accepted by tf.nn.max_pool
# print(tf.Session().run(x))
#
# valid_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
# same_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#
# print(valid_pad.get_shape())
# print(same_pad.get_shape())

# TODO:SAME 和 VALID 的区别
"""
SAME是余下的不够pooling会用0来填充；VALID不够就会舍弃剩余的列，不填充数据
"""

# slim = tf.contrib.slim
# # trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
# weights = slim.variable('weights',
#                         shape=[10,10,3, 3],  # 形状
#                         # 参数初始化
#                         initializer=tf.truncated_normal_initializer(stddev=0.1),
#                         regularizer=slim.l2_regularizer(0.05)
#                         )
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(weights))
