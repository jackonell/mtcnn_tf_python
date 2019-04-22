import tensorflow as tf
import numpy as np
import cv2
from net.utils import nms
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
        25
        1-15 4-18 7-21 10-24
        19 12 6.0 2
        1-12 3-14 5-16 7-18
        """
        field,jump,start = np.array([12,2,6.0])/ratio

        cls = np.squeeze(cls)
        height,width,_ = cls.shape
        index_arr = np.zeros_like(cls)

        cls = cls[:,:,0]
        cls = np.squeeze(cls)
        bbox = np.squeeze(bbox)

        for i in range(height):
            for j in range(width):
                index_arr[i][j] = [j,i]

        mask = np.where(cls >= self.thresholds[0])

        cls = cls[mask]
        bbox = bbox[mask]
        index_arr = index_arr[mask]


        bbox = bbox*field

        index_arr = index_arr*jump
        wh_arr = np.zeros_like(index_arr)
        wh_arr = wh_arr+field
        receptive_field = np.hstack((index_arr,wh_arr))

        bbox = bbox+receptive_field

        return cls,bbox

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

    def detect_rnet(self, img,bbxs):
        """
        判断pnet预测的bbx中，有多少是真正的人脸

        """
        if self.rnet_detector is None:
            self.rnet_detector = Detector(RNet,cfg.MODEL_PATH%cfg.RNET_DIR,24)

        #暂时只保存大于24的人脸,至于是否要将小于的人脸resize,留到以后再说
        # mask = np.min(bbxs[:,2:],axis=1) >= 24
        # bbxs = bbxs[mask]

        bbxs,cls,bbr,landmark = self.rnet_detector.slide_predict(img,bbxs)

        cls = np.array(cls)
        bbr = np.array(bbr)
        cls = np.squeeze(cls)
        bbr = np.squeeze(bbr)
        cls = cls[:,0]

        cls_refine = []
        bbx_refine = []
        landmark_refine = []

        thresh = self.thresholds[1]

        for idx in range(len(bbxs)):
            if cls[idx] > thresh:
                x,y,w,h = bbxs[idx]

                #由于预测的结果是相对于resize后的图片，
                #所以真实的bbx，有两种计算方式:
                #1.只将结果看成相对于resize前的图片
                #2.按相对于resize前的图片计算真实坐标值后，按resize比例缩放
                #咱是按方法一处理

                #计算真正的bbx
                # base = np.array([w,h,w,h])
                # bbx_tmp = np.multiply(base,bbr[idx])+bbxs[idx]

                # bbx_refine.append(bbx_tmp)

                bbx_refine.append(bbxs[idx])

                #保存cls
                cls_refine.append(cls[idx])

        return cls_refine,bbx_refine,landmark_refine

    def detect_onet(self, img,bbxs):
        """
        判断rnet过滤后的bbx中，有多少真正的人脸
        """
        if self.rnet_detector is None:
            self.rnet_detector = Detector(RNet,cfg.MODEL_PATH%cfg.ONET_DIR,48)

        cls,bbr,landmark = detector.slide_predict(img,bbxs)

        cls_output = []
        bbx_output = []
        landmark_output = []

        thresh = self.thresholds[2]

        for idx in range(len(bbxs)):
            if cls[idx] > thresh:
                x,y,w,h = bbxs[idx]

                base = np.array([w,h])

                #计算真正的landmark
                landmark_tmp = np.array(landmark[idx]).reshape((-1,2))
                landmark_tmp = landmark_tmp*base+np.array([x,y])
                landmark_output.append(landmark_tmp)

                #计算真正的bbx
                base = np.array([w,h,w,h])
                bbx_tmp = base*bbr+bbxs[idx]
                bbx_output.append(bbx_tmp)

                #保存cls
                cls_output.append(cls[idx])

        return cls_output,bbx_output,landmark_output

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

