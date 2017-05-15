# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:37:27 2017

@author: 95
"""
import tensorflow as tf

def read_and_decode(filename):
#    files = tf.train.match_filenames_once(filename)     
    files =[filename]
    print(files)
    filename_queue = tf.train.string_input_producer(files,shuffle= False)
    reader= tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                   features={
                                    "img_raw":tf.FixedLenFeature([],tf.string),
                                    "label":tf.FixedLenFeature([],tf.int64)        
                                             })
    decoded_image = tf.decode_raw(features["img_raw"],tf.uint8)
    decoded_image = tf.reshape(decoded_image,[30,30,3])
    label = tf.cast(features["label"],tf.int32)
    return decoded_image,label
def main(_):
    img,label = read_and_decode("./train.tfrecords")
    img_batch,label_batch = tf.train.shuffle_batch([img,label],batch_size=100,capacity=2000,
                                                   min_after_dequeue=1000)
    with tf.Session() as sess:
        init =tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        for i in range(1):
#            val, l= sess.run([img_batch, label_batch])
            val, l= sess.run([img, label])
            print(val.shape)
        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__': 
    tf.app.run(main)
    
    
    

