from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# slim主要是做代码瘦身-2016年开始
slim = tf.contrib.slim


# 5 x Inception-Resnet-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Build the 35 x 35 resnet block"""
    # 用于管理一个graph钟变量的名字，避免变量之间的命名冲突
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat(axis=3,values=[tower_conv,tower_conv1_1,tower_conv2_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# 10 x Inception-Resnet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Build the 17x17 resnet block"""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1,7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7,1],
                                        scope='Conv2d_0c_7x1')
            mixed = tf.concat(axis=3, values=[tower_conv,tower_conv1_2])
            up = slim.conv2d(mixed,net.get_shape()[3], 1, normalizer_fn=None,
                             activation_fn=None, scope='Conv2d_1x1')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net


# 5 x Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1,3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1,256, [3,1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat(axis=3, values=[tower_conv,tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-ResNet-V1
def inception_resnet_v2_base(inputs,
                             final_endpoint='Conv2d_7b_1x1',
                             output_stride=16,
                             align_feature_maps=False,
                             scope=None):
    if output_stride != 8 and output_stride != 16:
        raise ValueError('output_stride must be 8 or 16.')

    padding = 'SAME' if align_feature_maps else 'VALID'

    end_points = {}

    def add_and_check_final(name, net):
        end_points[name] = net
        return name == final_endpoint

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):

            # 149 x 149 x 32
            net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,
                              scope='Conv2d_1a_3x3')
            if add_and_check_final('Conv2d_1a_3x3',net):return net, end_points

            # 147 x 147 x 32
            net = slim.conv2d(net, 32, 3, padding=padding,
                              scope='Conv2d_2a_3x3')
            if add_and_check_final('Conv2d_2b_3x3',net):return net, end_points

            # 147 x 147 x 64
            net = slim.conv2d(net, 64, 3, padding=padding,
                              scope='Conv2d_2b_3x3')
            if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points

            # 73 x 73 x 64
            net = slim.max_pool2d(net, 3, strides=2, padding=padding,
                                  scope='MaxPool_3a_3x3')
            if add_and_check_final('Conv2d_3a_3x3', net):return net, end_points

            # 73 x 73 x 80
            net = slim.conv2d(net, 80, 1, padding=padding,
                              scope='Conv2d_3b_1x1')
            if add_and_check_final('Conv2d_3b_1x1',net):return net, end_points

            # 71 x 71 x 192
            net = slim.conv2d(net, 192, 3, padding=padding,
                              scope='Conv2d_4a_3x3')
            if add_and_check_final('Conv2d_4a_3x3',net):return net,end_points

            # 35 x 35 x 192
            net = slim.max_pool2d(net, 3, strides=2, padding=padding,
                                  scope='MaxPool_5a_3x3')
            if add_and_check_final('MaxPool_5a_3x3',net):return net, end_points

            # 35 x 35 x 320
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv2d(net, 96,1,scope='Conv2d_1x1')
                with tf.variable_scope('Branch_1'):
                    tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                                scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                    tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                                scope='Conv2d_0b_3x3')
                    tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                                scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    tower_pool = slim.avg_pool2d(net, 3, strides=1, padding='SAME',
                                                 scope='Avgpool_0a_3x3')
                    tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                               scope='Conv2d_ob_1x1')
                net = tf.concat(
                    [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1],3
                )

            if add_and_check_final('Mixed_5b', net):return net, end_points

            # TODO(alemi):Register intermediate endpoints

            net = slim.repeat(net, 10, block35(), scale=0.17)






