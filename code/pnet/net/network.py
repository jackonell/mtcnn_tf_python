import tensorflow as tf



class cnnbox:

    def __init__(self,stdev=0.01):
        self.stdev = stdev

    def conv2d(self, pre_layer, name , in_channels, out_channels, stride=[1,1,1,1], filter_size=[3,3], padding='VALID', activation_fn=None):
        with tf.variable_scope(name):
            # filter = tf.truncated_normal([filter_size[0],filter_size[1],in_channels,out_channels],0.0,self.stdev)
            # w_c = tf.Variable(filter,name='weight')
            w_c = tf.get_variable(shape=[filter_size[0],filter_size[1],in_channels,out_channels],initializer=tf.contrib.layers.xavier_initializer(),name="weight")

            # bs = tf.truncated_normal([out_channels],0.0,self.stdev)
            # b_c = tf.Variable(bs,name='bias');
            b_c = tf.get_variable(shape=[out_channels],initializer=tf.contrib.layers.xavier_initializer(),name="bias")

            conv = tf.nn.conv2d(pre_layer,w_c,stride,padding=padding)
            cf  = tf.nn.bias_add(conv,b_c)

            if activation_fn is not None:
                cf = activation_fn(cf)

            return cf

    def max_pool2d(self, pre_layer, name, stride=[1,2,2,1], filter_size=[1,2,2,1], padding='SAME'):
        with tf.variable_scope(name):
            maxp = tf.nn.max_pool(pre_layer,ksize=filter_size,strides=stride,padding=padding)
            return maxp

    def fc(self,pre_layer,name,in_size,out_size, activation_fn=None):
        with tf.variable_scope(name):
            inc = tf.truncated_normal([in_size,out_size],0.0,self.stdev)
            # inc = tf.truncated_normal([in_size,out_size])
            w_f = tf.Variable(inc,name='weight')

            outc = tf.truncated_normal([out_size],0.0,self.stdev)
            # outc = tf.truncated_normal([out_size])
            b_f = tf.Variable(outc,name='bias')

            x = tf.reshape(pre_layer,[-1,in_size])
            mul = tf.matmul(x,w_f)
            fc = tf.nn.bias_add(mul,b_f)

            if activation_fn is not  None:
                fc = activation_fn(fc)

            return fc

    def prelu(self,x,name=None):
        if name is None:
            name = "alpha"
        _alpha = tf.get_variable(name,shape=x.get_shape(),initializer=tf.constant_initializer(0.0),dtype=x.dtype)
        return tf.maximum(_alpha*x,x)
