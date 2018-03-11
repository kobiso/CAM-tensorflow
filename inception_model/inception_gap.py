"""'inception_gap.py' is for constructing GoogLeNet-GAP model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inception_model import inception_v3, inception_utils

slim = tf.contrib.slim

"""Add global pooling layer (GAP) on the inception_v3 network.
"""
def inception_gap(X,
                 output_dim,
                 input_dim=None,                                                  
                 is_training=False,
                 dropout_keep_prob=0.8,
                 weight_decay=None):
    
    with slim.arg_scope(inception_v3.inception_v3_arg_scope(weight_decay=weight_decay)):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):        
            net, _ = inception_v3.inception_v3_base(X, final_endpoint='Mixed_7c', scope='InceptionV3')
            
            with tf.variable_scope('gap'):
                #conv_first = slim.conv2d(net, 1024, [3, 3], stride=1, padding='SAME', scope='conv_first')
                net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_average_pooling')
                #print ("gap dimension: ", gap.get_shape().as_list())
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                #batch_norm_l = slim.batch_norm(gap, scope='batch_norm')
                conv_second = slim.conv2d(net, output_dim, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_second')
                logits = tf.squeeze(conv_second, [1, 2], name='squeeze_logits')
                print ("logits dimension: ", logits.get_shape().as_list())
                Y_pred = slim.softmax(logits, scope='softmax_prediction')
                print ("Y_pred dimension: ", Y_pred.get_shape().as_list())
                
            W = tf.get_default_graph().get_tensor_by_name("gap/conv_second/weights:0")
            
    return logits, Y_pred

