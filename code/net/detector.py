import tensorflow as tf
import numpy as np
import cv2

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

    def slide_predict(self, patches,size):
        """
        预测结果
        滑动窗口方式，用于rnet与onet
        几个细节：
        1.建议框不一定是方形，但是网络只接受方形：
           1.1 直接将建议框中的图片resize为方形?(我采用的方式)
           1.2 建议框按长边扩展为方形?
        2.建议框有可能超出了边界
           2.1 padding?
           2,2 忽略这些框？(我采用的方式)
           2.3 将这些框超出边界的部分舍弃？
        3.bbxs可使用批量预测，不可使用循环逐个预测，速度太慢
        """
        all_cls = []
        all_bbr = []
        all_landmark = []

        #对patches考虑使用批量预测（batch size根据数据量确定）
        batch_size = int(2048*12/size)

        step = np.shape(patches)[0]/batch_size
        step = int(np.ceil(step))
        print("patch shape:%r batch_size:%d step:%d"%(np.shape(patches),batch_size,step))

        start = 0
        end = batch_size
        for idx in range(step):
            one_batch = patches[start:end]
            cls,bbr,landmark = self.sess.run([self.fcls_pred,self.bbr_pred,self.landmark_pred],feed_dict={"IMG:0":one_batch})

            all_cls.extend(cls)
            all_bbr.extend(bbr)
            all_landmark.extend(landmark)

            start = end
            end = end+batch_size

        return all_cls,all_bbr,all_landmark


