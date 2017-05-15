import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value =[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

cwd = os.getcwd()
writer = tf.python_io.TFRecordWriter("./train.tfrecords")
classes =["blue","yellow","green","white"]
for index, name in enumerate(classes):
    class_path = "I:/" + name+"/"
    dirnum = len(os.listdir(class_path))
    print(dirnum)
    for img_name in os.listdir(class_path):
        img_path = class_path+img_name
#        img=tf.image.decode_jpeg(img_path)
#        plt.imshow(img.eval())
        img = Image.open(img_path)
        img = img.resize((30,30))
        #plt.imshow(img)
        img_raw=img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
              'label':_int64_feature(index),
              'img_raw':_bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())
writer.close()
#        img = img.resize((224, 224))        
#        plt.show()
#        os.system("pause")
#        print(img_path)
#writer = tf.python_io.TFRecordWriter("./platecolor.tfrecords")
#mnist = input_data.read_data_sets("./data",dtype = tf.uint8,one_hot= True)
#images= mnist.train.images
#labels=mnist.train.labels

#shape = images.shape
#pixels = images.shape[1]
#num_examples = mnist.train.num_examples

#filename ="./out.tfrecords"

#writer = tf.python_io.TFRecordWriter(filename)
#for index in range(num_examples):
#    image_raw=images[index].tostring()
#    example = tf.train.Example(features= tf.train.Features(feature={
#            'pixels':_int64_feature(pixels),
#            'labels':_int64_feature(np.argmax(labels[index])),
#            'image_raw' :_bytes_feature(image_raw)}))
#    writer.write(example.SerializeToString())
#writer.close()
