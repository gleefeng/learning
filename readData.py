# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:33:25 2017

@author: 95
"""
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
reader = tf.TFRecordReader()
files = tf.train.match_filenames_once("./*.tfrecords")
#filename_queue = tf.train.string_input_producer(["./train.tfrecords"])
filename_queue = tf.train.string_input_producer(files,shuffle= False)
_,serialized_example = reader.read(filename_queue)
features= tf.parse_single_example(serialized_example,
                                  features={
                                          'img_raw':tf.FixedLenFeature([],tf.string),
                                          'label':tf.FixedLenFeature([],tf.int64),
                                          })
images=tf.decode_raw(features['img_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
#images=features['image_raw']
#labels =features['labels']
#pixels =features['pixels']
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord = coord)
for i in range(10):
    image,label = sess.run([images,labels])
#    print(image.shape)
#    image.shape=(30,30,3)
#    print(image.shape)
    img =image.reshape(30,30,3)
    plt.imshow(img)
    plt.show()
#    print(label)