#!/usr/bin/python
# -*- coding: utf-8 -*-

# import tensorflow as tf
#
# tow_node = tf.constant(2.)
# three_node = tf.constant(3)
#
# input_placeholder = tf.placeholder(tf.int32)
#
# variable = tf.get_variable("a", [])
#
# sum_node = tf.add(input_placeholder, three_node)
#
# assign_node = tf.assign(variable, tow_node)
#
# with tf.Session() as sess:
#     # _sum, _, var = sess.run([sum_node, variable, assign_node], feed_dict={input_placeholder: 13})
#     # sess.run(assign_node)
#     var, nu, = sess.run([variable, assign_node, ])
# print(var)

# import tensorflow as tf
# count_variable = tf.get_variable("count", [])
# zero_node = tf.constant(0.)
# assign_node = tf.assign(count_variable, zero_node)
#
# sess = tf.Session()
# sess.run(assign_node)
# print(sess.run(count_variable))



import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
### this new copy of two_node is not on the computation path, so nothing prints!
print_two_node = tf.Print(two_node, [two_node, three_node, sum_node])
sess = tf.Session()
print(sess.run(sum_node))
print(sess.run(print_two_node))
