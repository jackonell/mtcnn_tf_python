import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32,[None,5,5,3],name="X")

with tf.variable_scope('layer1-conv1'):
    kernel = np.random.randint(5,size=(2,2,3,1)) 
    print("kernel: %r"%np.squeeze(kernel))
    w_c1 = tf.cast(tf.Variable(kernel,name="weight"),tf.float32)
    con = tf.nn.conv2d(X,w_c1,strides=[1,1,1,1],padding="SAME")
    ops = 1-X

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = np.random.randint(5,size=(1,5,5,3)) 
    print("input: %r"%np.squeeze(x))
    Y,p = sess.run([con,ops],feed_dict={X:x}) 
    print("conv result: %r"%np.squeeze(np.array(Y)))
    print(p)
