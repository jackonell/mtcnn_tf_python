import tensorflow as tf

class cnnbox:

    def __init__(self,activation_fn = None):
        if activation_fn is None:
            self.activation_fn = prelu
        else:
            self.activation_fn = activation_fn

    '''
      卷积例子如下：
      with tf.variable_scope('layer1-conv1'):
            w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]),name='weight')
            b_c1 = tf.Variable(b_alpha*tf.random_normal([32]),name='bias')
            relu1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    '''
    def conv2d(self, pre_layer, name , in_channels, out_channels, stride=[1,1,1,1], filter_size=[3,3], padding='SAME', activation_fn=None):
        with tf.variable_scope(name):
            filter = tf.truncated_normal([filter_size[0],filter_size[1],in_channels,out_channels],0.0,0.001)
            w_c = tf.Variable(filter,name='weight')

            bs = tf.truncated_normal([out_channels],0.0,0.001)
            b_c = tf.Variable(bs,name='bias');

            conv = tf.nn.conv2d(pre_layer,w_c,stride,padding=padding)
            cf  = tf.nn.bias_add(conv,b_c)

            activivation
            if activivation_fn == None:
                activivation = self.activation_fn(cf)
            else:
                activation = activation_fn(cf)

            return activation

    def max_pool2d(self, pre_layer, name, stride=[[1,2,2,1], filter_size=[1,2,2,1]], padding='SAME'):
        with tf.variable_scope(name):
            tf.nn.max_pool(pre_layer,ksize=filter_size,strides=stride,padding=padding)

    def fc(self,pre_layer,name,in_size,out_size, activation_fn=None):
        with tf.variable_scope(name):
            inc = tf.truncated_normal([in_size,out_size],0.0,0.001)
            w_f = tf.Variable(inc,name='weight')

            outc = tf.truncated_normal([out_size],0.0,0.001)
            b_f = tf.Variable(outc,name='bias')

            x = tf.reshape(pre_layer,[-1,in_size])
            mul = tf.matmul(x,w_f)
            fc = tf.nn.bias_add(mul,b_f)

            activation
            if activation_fn != None:
                activation = activation_fn(fc)

            return activation

    def prelu(x,name=None):
        if name is None:
            name = "alpha"
        _alpha = tf.get_variable(name,shape=x.get_shape(),initializer=tf.constant_initializer(0.0),dtype=x.dtype)
        return tf.maximum(_alpha*x,x)
