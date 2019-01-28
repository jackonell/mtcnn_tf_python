import numpy as np
import tensorflow as tf

def train_net_wise(current_net,ratio):
    """
    训练网络
    :current_net: TODO
    :ratio: TODO
    :returns: TODO
    """
    IMG = tf.placeholder(tf.float32,[None,size,size,3],name="IMG")
    CLS = tf.placeholder(tf.float32,[None],name="CLS")
    BBX = tf.placeholder(tf.float32,[None,size,size,4],name="BBX")
    LANDMARK = tf.placeholder(tf.float32,[None,size,size,10],name="LANDMARK")

    fcls_pred,bbr_pred,landmark_pred = current_net(IMG)

    fcls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fcls_pred,labels=CLS))
    bbx_loss = tf.reduce_mean(tf.square(bbr_pred-BBX))
    landmark_loss = tf.reduce_mean(tf.square(landmark_pred-LANDMARK))

    loss = ratio[0]*fcls_loss+ratio[1]*bbx_loss+ratio[2]*landmark_loss
    optimizer = tf.train.MomentumOptimizer().minimize(loss)

    return loss,optimizer


def train(arg1):
    """TODO: Docstring for train_pnet.

    :arg1: TODO
    :returns: TODO

    """
    pass
