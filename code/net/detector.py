import tensorflow as tf
import numpy as np
import cv2

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
    def __init__(self,xnet,xnet_model_path,img_size=None):
        self.size = img_size
        graph = tf.Graph()
        with graph.as_default():

            img = tf.placeholder(tf.float32,shape=[None,img_size,img_size,3],name="IMG")
            img = (tf.cast(img,tf.float32)-127.5)/128

            self.fcls_pred,self.bbr_pred,self.landmark_pred = xnet(img)
            self.fcls_pred = tf.nn.softmax(self.fcls_pred)

            self.sess = tf.Session()
            ckpt = tf.train.get_checkpoint_state(xnet_model_path)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess,ckpt.model_checkpoint_path)

    def fcn_predict(self, img):
        """
        预测结果
        全卷积的方式，用于pnet
        """
        img = img[np.newaxis,:,:,:]
        cls,bbr,landmark = self.sess.run([self.fcls_pred,self.bbr_pred,self.landmark_pred],feed_dict={"IMG:0":img})
        return cls,bbr,landmark

    def slide_predict(self, img, bbxs):
        """
        预测结果
        滑动窗口方式，用于rnet与onet
        """
        all_cls = []
        all_bbr = []
        all_landmark = []
        all_bbx = []

        for bbx in bbxs:
            x,y,w,h = list(map(int,bbx))

            #对于x或者y小于0的情况,考虑处理方式
            #1.padding
            #2.过滤掉
            #3.设为0
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            patch = img[x:x+w,y:y+h]
            ch,cw,_ = np.shape(patch)

            if cw*ch == 0:
                continue

            patch = cv2.resize(patch,(self.size,self.size))
            patch = patch[np.newaxis,:,:,:]

            cls,bbr,landmark = self.sess.run([self.fcls_pred,self.bbr_pred,self.landmark_pred],feed_dict={"IMG:0":patch})

            all_cls.append(cls)
            all_bbr.append(bbr)
            all_landmark.append(landmark)
            all_bbx.append(bbx)

        return all_bbx,all_cls,all_bbr,all_landmark


