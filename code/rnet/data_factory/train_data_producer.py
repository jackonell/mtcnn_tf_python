import tensorflow as tf
import numpy as np
import math
import cv2
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg
from mtcnn_utils import IOU
from mtcnn_utils import NMS
from net.mtcnn import PNet

class Detector(object):

    """用于产生rnet与onet所需的训练数据"""

    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():

            # IMG = tf.placeholder(tf.float32,name="IMG")
            # WIDTH = tf.placeholder(tf.int32,name="WIDTH")
            # HEIGHT = tf.placeholder(tf.int32,name="HEIGHT")

            # img = tf.reshape(IMG,[1,HEIGHT,WIDTH,3])
            img = tf.placeholder(tf.float32,shape=[None,None,None,3],name="IMG") #是否有更简单的方式实现输入不同大小的图片
            img = (tf.cast(img,tf.float32)-127.5)/128

            self.fcls_pred,self.bbr_pred,self.landmark_pred = PNet(img)
            self.fcls_pred = tf.nn.softmax(self.fcls_pred)

            self.sess = tf.Session()
            ckpt = tf.train.get_checkpoint_state(cfg.PNET_MODEL_PATH)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess,ckpt.model_checkpoint_path)

    def predict(self, img):
        """
        预测结果
        """
        img = img[np.newaxis,:,:,:]
        cls,bbr,landmark = self.sess.run([self.fcls_pred,self.bbr_pred,self.landmark_pred],feed_dict={"IMG:0":img})
        return cls,bbr,landmark

def calc_receptive_filed():
    """
    计算感受野与第一个点坐标，jump等
    """
    field = 12
    jump = 2
    start = 6.0

    return field,jump,start

def calc_real_coordinate(ratio,bbox,cls):
    """
    根据当前图像和缩放比例，计算出bbox中的数值对应的真实坐标值
    25
    1-15 4-18 7-21 10-24
    19 12 6.0 2
    1-12 3-14 5-16 7-18
    """
    field,jump,start = calc_receptive_filed()

    field = (field/ratio)
    jump  = (jump/ratio)
    start = (start/ratio)

    cls = np.squeeze(cls)
    bbox = np.squeeze(bbox)

    height,width,_ = cls.shape

    bbox = bbox*field
    remain = []
    for i in range(height):
        for j in range(width):
           centerx = start + j*jump
           centery = start + i*jump
           x = centerx - field/2
           y = centery - field/2

           #得分大于0.5，则为候选框
           if cls[i,j,0] >= 0.5:
               x = x+bbox[i,j,0]
               y = y+bbox[i,j,1]
               w = field+bbox[i,j,2]
               h = field+bbox[i,j,3]
               if w > 24 and h > 24:
                   remain.append([x,y,w,h,cls[i,j,0]])
               # remain.append([x,y,field,field,cls[i,j,0]])

    return remain

def produce_rnet_detection_train_dataset():
    """
    产生用于训练rnet的detection数据集:
    注意到大量的neg,少量的par以及极少的pos,那么如何保证3:1:1:2的数据量呢？
    得分大于0.5，则为候选框
    只对neg部分进行nms
    """
    fneg = open(cfg.RNET_TRAIN_NEG_TXT_PATH,"w")
    fpos = open(cfg.RNET_TRAIN_POS_TXT_PATH,"w")
    fpar = open(cfg.RNET_TRAIN_PART_TXT_PATH,"w")

    num_neg = 0
    num_par = 0
    num_pos = 0

    resize_ratio = 0.79

    with open(cfg.PNET_TRAIN_FORMATTER_TXT_PATH,"r") as f:
        data = f.readlines();

    detector = Detector()
    flag = 0
    total_num = len(data)

    for line in data:
        # if flag > 0:
            # break
        flag = flag + 1

        annotations = line.strip().split()
        img_path = cfg.ORIGINAL_IMG_PATH+annotations[0]
        print(img_path)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)

        true_boxes = list(map(float,annotations[1:]))
        true_boxes = np.array(true_boxes,dtype="float32").reshape(-1,4)

        height,width,_ = img.shape

        ratio = 1
        #临时存放每张图片上产生的所有输出框
        boxes_cls_temp = []
        while min(width,height) > 24 :
            cls,bbr,_ = detector.predict(img)

            #计算出预测正例的真实坐标值
            keep_boxes = calc_real_coordinate(ratio,bbr,cls)
            boxes_cls_temp.extend(keep_boxes)

            ratio  = ratio*resize_ratio
            width  = width*resize_ratio
            height = height*resize_ratio

            img = cv2.resize(img,(int(width),int(height)))

        if len(boxes_cls_temp) == 0:
            continue

        boxes_cls_temp = np.array(boxes_cls_temp)
        # remain = NMS(boxes_cls_temp[:,:4],boxes_cls_temp[:,4],0.5)
        # boxes_cls_temp = boxes_cls_temp[remain]

        # 查看图片
        # img0 = cv2.imread(img_path,cv2.IMREAD_COLOR)
        # for rec in boxes_cls_temp:
            # rec = list(map(int,rec))
            # cv2.rectangle(img0,(rec[0],rec[1]),(rec[0]+rec[2],rec[1]+rec[3]),(0,255,0),2)
        # cv2.imwrite("%d.jpg"%flag,img0)

        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        np.random.shuffle(boxes_cls_temp)
        # 对剩余的框，为其打标签，pos,par,neg,并产生用于训练的图片
        neg_temp = 0
        par_temp = 0
        for box in boxes_cls_temp:
            box = box[:-1]
            x,y,w,h = box

            # 可以不写
            if w < 24 or h < 24 or x < 0 or y < 0:
                continue

            iou = IOU(box,true_boxes)
            max_idx = np.argmax(iou)
            miou = np.max(iou)

            # if miou >= 0.4:
            tx,ty,tw,th = true_boxes[max_idx]

            offset_x = (tx-x)/w
            offset_y = (ty-y)/h
            offset_w = (tw-w)/w
            offset_h = (th-h)/h

            box = list(map(int,box))
            x,y,w,h = box
            crop_img = img[y:y+h,x:x+w]
            resized_img = cv2.resize(crop_img,(24,24))

            #注意预测的图片不是方形
            if miou >= 0.65: #pos
                cv2.imwrite("%spos_%d.jpg"%(cfg.RNET_TRAIN_IMG_PATH,num_pos),resized_img)
                fpos.write("pos_%d.jpg 1 %f %f %f %f\n"%(num_pos,offset_x,offset_y,offset_w,offset_h))
                num_pos = num_pos+1
            elif miou >= 0.4 and par_temp < 100:
                cv2.imwrite("%spar_%d.jpg"%(cfg.RNET_TRAIN_IMG_PATH,num_par),resized_img)
                fpar.write("par_%d.jpg 0 %f %f %f %f\n"%(num_par,offset_x,offset_y,offset_w,offset_h))
                num_par = num_par+1
                par_temp = par_temp+1
            elif miou <= 0.3 and neg_temp < 80:
                cv2.imwrite("%sneg_%d.jpg\n"%(cfg.RNET_TRAIN_IMG_PATH,num_neg),resized_img)
                fneg.write("neg_%d.jpg -1"%(num_neg))
                num_neg = num_neg+1
                neg_temp = neg_temp+1

        num = num_neg+num_par+num_pos
        print("共需处理图片%d张，已经处理%d张，产生图片%d张：neg-%d张，par-%d张，pos-%d张"%(total_num,flag,num,num_neg,num_par,num_pos))
        if num_neg > 750000 and num_par > 250000 and num_pos > 250000:
            break

    fneg.close()
    fpar.close()
    fpos.close()

def produce_rnet_landmark_train_dataset():
    """
    产生用于训练rnet的landmark数据
    """
    flandmark = open(cfg.RNET_TRAIN_LANDMARK_TXT_PATH,"w")

    with open(cfg.ORIGINAL_LANDMARK_TXT_PATH,"r") as f:
        landmark_annotations = f.readlines()

    detector = Detector()
    num_landmark = 0

    resize_ratio = 0.79
    total_num = len(landmark_annotations)
    flag = 0

    for ma in landmark_annotations:
        # if flag > 0:
            # break
        flag = flag + 1

        ma = ma.strip().split()

        img_path = cfg.ORIGINAL_LANDMARK_IMG_PATH+ma[0].replace("\\","/")
        bbx = list(map(int,ma[1:5]))
        landmark = list(map(float,ma[5:]))
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)

        height,width,_ = img.shape

        bbx = np.array(bbx).reshape((-1,2))
        w,h = bbx[:,1]-bbx[:,0]
        x,y = bbx[:,0]

        true_boxes = np.array([x,y,w,h],dtype="float32").reshape(-1,4)

        landmark = np.array(landmark).reshape((-1,2))

        ratio = 1
        boxes_cls_temp = []
        while width > 24 and height > 24:
            cls,bbr,_ = detector.predict(img)

            #计算出预测正例的真实坐标值
            keep_boxes = calc_real_coordinate(ratio,bbr,cls)
            boxes_cls_temp.extend(keep_boxes)

            ratio  = ratio*resize_ratio
            width  = width*resize_ratio
            height = height*resize_ratio

            img = cv2.resize(img,(int(width),int(height)))

        if len(boxes_cls_temp) == 0:
            continue

        boxes_cls_temp = np.array(boxes_cls_temp)
        # remain = NMS(boxes_cls_temp[:,:4],boxes_cls_temp[:,4],0.5)
        # boxes_cls_temp = boxes_cls_temp[remain]

        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        height,width,_ = img.shape
        #对剩余的框，为pos框，记录landmark
        for box in boxes_cls_temp:
            box = box[:-1]
            rx,ry,rw,rh = box

            if rw < 24 or rh < 24 or rx < 0 or ry < 0:
                continue

            iou = IOU(box,true_boxes)
            miou = np.max(iou)
            is_flip = 0

            #如果是正例，则计算landmark
            while miou >= 0.65 and is_flip < 2:
                #归一化特征点
                landmark_temp = np.zeros_like(landmark)

                if is_flip == 1:
                    img = cv2.flip(img,1)
                    box[0] = width-rx-rw
                    landmark[:,0] = width-landmark[:,0]

                # print("%r %r %r %r"%(is_flip,img.shape,box,width))
                is_flip = is_flip+1

                landmark_temp[:,0] = (landmark[:,0]-rx)/rw
                landmark_temp[:,1] = (landmark[:,1]-ry)/rh

                box = list(map(int,box))
                rx,ry,rw,rh = box
                crop_img = img[ry:ry+rh,rx:rx+rw]
                resized_img = cv2.resize(crop_img,(24,24))

                cv2.imwrite("%slandmark_%d.jpg"%(cfg.RNET_TRAIN_IMG_PATH,num_landmark),resized_img)
                landmark_temp = landmark_temp.reshape((-1))
                landmark_temp = list(map(str,landmark_temp))
                flandmark.write("landmark_%s.jpg 2 %s\n"%(num_landmark," ".join(landmark_temp)))
                num_landmark = num_landmark+1

        print("共需处理图片%d张，已经处理%d张，产生landmark图片%d张"%(total_num,flag,num_landmark))

    flandmark.close()

if __name__ == "__main__":
    # produce_rnet_detection_train_dataset()
    produce_rnet_landmark_train_dataset()

