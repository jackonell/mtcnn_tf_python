import tensorflow as tf

# #our NN's output
# logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
# #step1:do softmax
# y=tf.nn.softmax(logits)
# #true label
# y_=tf.constant([[0.0,0.5,0.5],[0.0,0.0,1.0],[0.0,0.0,1.0]])
# #step2:do cross_entropy
# # cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# cross_entropy = -(y_*tf.log(y))
# #do cross_entropy just one step
# # cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))#dont forget tf.reduce_sum()!!
# cross_entropy2=(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))#dont forget tf.reduce_sum()!!

# with tf.Session() as sess:
    # softmax=sess.run(y)
    # c_e = sess.run(cross_entropy)
    # c_e2 = sess.run(cross_entropy2)
    # print("step1:softmax result=")
    # print(softmax)
    # print("step2:cross_entropy result=")
    # print(c_e)
    # print("Function(softmax_cross_entropy_with_logits) result=")
    # print(c_e2)

import tensorflow as tf
import numpy as np

logits = tf.constant([7.78562784e-01, 8.81508589e-02])
labels = tf.constant([0.0,1.0])

_logits = tf.nn.softmax(logits)
cross_entropy = -labels*tf.log(_logits)

loss = (tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

with tf.Session() as sess:
    sf = sess.run(_logits)
    ce = sess.run(cross_entropy)
    lossv = sess.run([loss])
    print(sf)
    print(ce)
    print(lossv)
