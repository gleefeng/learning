# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:13:47 2017

@author: 95
"""
import matplotlib.pyplot as plt

import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("1.jpg",'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    a = img_data.eval()
    plt.imshow(img_data.eval())
    plt.show()              
#    print(a)
    img_data = tf.image.convert_image_dtype(img_data,dtype = tf.uint8)
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("aaa.jpg","wb") as f:
        f.write(encoded_image.eval())