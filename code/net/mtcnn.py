import tensorflow as tf
import numpy as np
import cv2
import time
from net.utils import nms,crop_img_by_bbxs,timecost
from net.detector import Detector
from net.cfgs import cfg
from net.component_nets import PNet,RNet,ONet

class Mtcnn(object):

    """
    使用三个网络对图片上的人脸进行检测
    可以根据需要调节每个网络的得分阈值
    可以根据需要只选择部分网络进行预测
    """

    def __init__(self,output_net,thresholds):
        self.output_net = output_net
        self.thresholds = thresholds
        self.pnet_detector = None
        self.rnet_detector = None
        self.onet_detector = None

    def calc_real_coordinate(self,ratio,bbox,cls):
        """
        根据当前图像和缩放比例，计算出bbox中的数值对应的真实坐标值
        """
        field,jump,start = np.array([12,2,6.0])/ratio

        cls = np.squeeze(cls)

        cls = cls[:,:,0]
        cls = np.squeeze(cls)
        bbox = np.squeeze(bbox)

        mask = np.where(cls >= self.thresholds[0])

        cls = cls[mask]

        bbox = bbox[mask]
        bbox = bbox*field
        mask = np.array(mask,dtype=np.float64)
        xy = mask*jump

        bbox[:,:2] = bbox[:,:2]+xy.T
        bbox[:,2:] = bbox[:,2:]+field

        return cls,bbox

    @timecost
    def detect_pnet(self, img):
        """
        预测图片上的人脸位置与特征点
        """
        height,width,_ = img.shape
        resize_ratio = 0.79
        if self.pnet_detector is None:
            self.pnet_detector = Detector(PNet,cfg.MODEL_PATH%cfg.PNET_DIR)

        cls_proposal = []
        bbx_proposal = []
        landmark_proposal = []

        ratio = 1
        while min(width,height) > 24 :
            cls,bbr,landmark = self.pnet_detector.fcn_predict(img)

            #计算出预测正例的真实坐标值
            keep_cls,keep_boxes = self.calc_real_coordinate(ratio,bbr,cls)

            if np.shape(keep_boxes)[0] != 0:
                remain_idxs = nms(keep_boxes,keep_cls,0.5)
                bbx_proposal.extend(list(keep_boxes[remain_idxs]))
                cls_proposal.extend(list(keep_cls[remain_idxs]))

            ratio  = ratio*resize_ratio
            width  = width*resize_ratio
            height = height*resize_ratio

            img = cv2.resize(img,(int(width),int(height)))

        if len(bbx_proposal) == 0:
            return cls_proposal,bbx_proposal,landmark_proposal

        cls_proposal = np.array(cls_proposal)
        bbx_proposal = np.array(bbx_proposal)
        remain = nms(bbx_proposal,cls_proposal,0.7)

        bbx_proposal = bbx_proposal[remain]
        cls_proposal = cls_proposal[remain]

        return cls_proposal,bbx_proposal,landmark_proposal

    @timecost
    def detect_rnet(self, img,bbxs):
        """
        判断pnet预测的bbx中，有多少是真正的人脸
        """
        size = 24
        if self.rnet_detector is None:
            self.rnet_detector = Detector(RNet,cfg.MODEL_PATH%cfg.RNET_DIR,size)

        #首先要处理bbxs长宽不一致和超出边界的问题
        patches,bbxs = crop_img_by_bbxs(img,bbxs,size)

        cls,bbr,_ = self.rnet_detector.slide_predict(patches,size)

        cls = np.array(cls)
        bbr = np.array(bbr)

        cls = np.squeeze(cls)
        bbr = np.squeeze(bbr)
        cls = cls[:,0]

        thresh = self.thresholds[1]

        mask = np.where(cls > thresh)

        cls = cls[mask]
        bbr = bbr[mask]
        bbxs = bbxs[mask]

        wh = bbxs[:,2:]
        wh0 = np.tile(wh,2)

        bbxs = bbr*wh0+bbxs

        return cls,bbxs,[]

    @timecost
    def detect_onet(self, img,bbxs):
        """
        判断rnet过滤后的bbx中，有多少真正的人脸
        """
        size = 48
        if self.onet_detector is None:
            self.onet_detector = Detector(ONet,cfg.MODEL_PATH%cfg.ONET_DIR,size)


        #首先要处理bbxs长宽不一致和超出边界的问题
        patches,bbxs = crop_img_by_bbxs(img,bbxs,size)

        cls,bbr,landmark = self.onet_detector.slide_predict(patches,size)

        cls = np.array(cls)
        bbr = np.array(bbr)
        landmark = np.array(landmark)

        cls = np.squeeze(cls)
        bbr = np.squeeze(bbr)
        landmark = np.squeeze(landmark)
        cls = cls[:,0]

        thresh = self.thresholds[2]

        mask = np.where(cls > thresh)

        cls = cls[mask]
        bbr = bbr[mask]
        bbxs = bbxs[mask]
        landmark = landmark[mask]

        xy = bbxs[:,:2]
        wh = bbxs[:,2:]

        wh0 = np.tile(wh,2)
        bbxs = bbr*wh0+bbxs

        xy1 = np.tile(xy,5)
        wh1 = np.tile(wh,5)
        landmark = landmark*wh1+xy1

        # landmark = [landmark[:,i*2:i*2+2]*wh+xy for i in range(5)]
        # landmark = np.concatenate(landmark,axis=1) #上一种写法性能更加,但是会占用更过的内存

        return cls,bbxs,landmark

    def detect(self, img):
        """
        使用三个阶段来检测人脸
        """
        cls = []
        bbxs = []
        landmark = []

        cls,bbxs,landmark = self.detect_pnet(img)
        if self.output_net == "PNet":
            return cls,bbxs,landmark

        cls,bbxs,landmark = self.detect_rnet(img,bbxs)
        if self.output_net == "RNet":
            return cls,bbxs,landmark

        cls,bbxs,landmark = self.detect_onet(img,bbxs)

        return cls,bbxs,landmark


if __name__ == "__main__":
    mtcnn = Mtcnn("RNet",[0.3,0.3,0.0])
    img = cv2.imread("0_Parade_Parade_0_106.jpg",cv2.IMREAD_COLOR)

    cls,bbr,landmark = mtcnn.detect_rnet(img,[[684,269,37,37]])

