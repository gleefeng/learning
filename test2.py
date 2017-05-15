# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:58:17 2017

@author: 95
"""

import tensorflow as tf

files = tf.train.match_filenames_once("train.*")
with tf.Session() as sess:
#    tf.initialize_all_variables().run()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
#    tf.initialize_variables(files)
    sess.run(files)