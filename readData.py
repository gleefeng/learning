# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:33:25 2017

@author: 95
"""
import tensorflow as tf

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["./out.tfrecords"])
_,serialized_example = reader.read(filename_queue)
features= tf.parse_single_example(serialized_example,
                                  features={
                                          'image_raw':tf.FixedLenFeature([],tf.string),
                                          'pixels':tf.FixedLenFeature([],tf.int64),
                                          'labels':tf.FixedLenFeature([],tf.int64),
                                          })
images=tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['labels'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)

#images=features['image_raw']
#labels =features['labels']
#pixels =features['pixels']


sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord = coord)
for i in range(10):
    image,label,pixel = sess.run([images,labels,pixels])
    print(image.type)
    print(label)
    print(pixel)