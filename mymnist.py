# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:53:57 2017

@author: 95
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def biase_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides= [1,2,2,1],padding='SAME')
def mydeepnn(x):
  x_image = tf.reshape(x,[-1,28,28,1])
  
  w_conv1 = weight_variable([5,5,1,32]) # 5乘以5的核，输入1通道，输出32通道
  b_conv1 = biase_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
  
def main(argv=None):
    mnist = input_data.read_data_sets("./data", one_hot=True)
    x= tf.placeholder(tf.float32,[None,784],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,10],name='y-input')
    
    
    
    
    print("main test!!!\n")

if __name__ =='__main__':
    tf.app.run()
    