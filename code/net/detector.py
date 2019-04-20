import tensorflow as tf
import numpy as np

import sys
import os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

class Detector(object):

    """
    用于检测图片中的人脸:
    1.pnet使用fcn_predict()
    2.rnet与onet使用slide_predict()

    因为pnet是全卷积网络，所以可以接收任意大小的输入图片，
    但是rnet与onet有全连接层，所以要使用滑动窗口的方式来检测人脸
    """

    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():

            img = tf.placeholder(tf.float32,shape=[None,None,None,3],name="IMG")
            img = (tf.cast(img,tf.float32)-127.5)/128

            self.fcls_pred,self.bbr_pred,self.landmark_pred = PNet(img)
            self.fcls_pred = tf.nn.softmax(self.fcls_pred)

            self.sess = tf.Session()
            ckpt = tf.train.get_checkpoint_state(cfg.PNET_MODEL_PATH)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess,ckpt.model_checkpoint_path)

    def fcn_predict(self, img):
        """
        用于pnet预测结果
        """
        img = img[np.newaxis,:,:,:]
        cls,bbr,landmark = self.sess.run([self.fcls_pred,self.bbr_pred,self.landmark_pred],feed_dict={"IMG:0":img})
        return cls,bbr,landmark

    def slide_predict(self, img,bbx):
        """
        用于rnet或者onet，预测是否人脸

        :img: TODO
        :returns: TODO

        """
        cls = []
        bbr = []
        landmark = []


        

