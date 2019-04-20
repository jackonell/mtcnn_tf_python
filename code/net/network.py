import tensorflow as tf


def logtf(fn):
    """
    用于记录日志的装饰器
    """
    def logsth(*args,**kwargs):
        layer = fn(*args,**kwargs)
        print(layer.get_shape())
        return layer

    return logsth

class cnnbox:

    def __init__(self,stdev=0.01):
        self.stdev = stdev

    @logtf
    def conv2d(self, pre_layer, name ,out_channels, stride=[1,1,1,1], filter_size=[3,3], padding='VALID', activation_fn=None):
        with tf.variable_scope(name):
            N,H,W,C = tf.shape(pre_layer)
            in_channels = C

            w_c = tf.get_variable(shape=[filter_size[0],filter_size[1],in_channels,out_channels],initializer=tf.contrib.layers.xavier_initializer(),name="weight")
            b_c = tf.get_variable(shape=[out_channels],initializer=tf.contrib.layers.xavier_initializer(),name="bias")

            conv = tf.nn.conv2d(pre_layer,w_c,stride,padding=padding)
            cf  = tf.nn.bias_add(conv,b_c)

            if activation_fn is not None:
                cf = activation_fn(cf)

            return cf

    @logtf
    def max_pool2d(self, pre_layer, name, stride=[1,2,2,1], filter_size=[1,2,2,1], padding='VALID'):
        with tf.variable_scope(name):
            maxp = tf.nn.max_pool(pre_layer,ksize=filter_size,strides=stride,padding=padding)
            return maxp

    @logtf
    def fc(self,pre_layer,name,out_size, activation_fn=None):
        with tf.variable_scope(name):
            pre_layer = tf.contrib.layers.flatten(pre_layer)
            print(pre_layer.get_shape())

            N,C = tf.shape(pre_layer)
            in_size = C

            w_f = tf.get_variable(shape=[in_size,out_size],initializer=tf.contrib.layers.xavier_initializer(),name="weight")
            b_f = tf.get_variable(shape=[out_size],initializer=tf.contrib.layers.xavier_initializer(),name="bias")

            mul = tf.matmul(pre_layer,w_f)
            fc = tf.nn.bias_add(mul,b_f)

            if activation_fn is not  None:
                fc = activation_fn(fc)

            return fc

    def prelu(self,inputs,name=None):
        alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs-abs(inputs))*0.5
        return pos + neg
