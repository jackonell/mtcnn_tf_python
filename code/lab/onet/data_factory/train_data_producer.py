import tensorflow as tf
import numpy as np
import math
import cv2
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from net.cfgs import cfg
from net.utils import iou,data_augmentation
from net.component_nets import PNet,RNet
from net.detector import Detector
from net.mtcnn import Mtcnn

def draw_bbx_on_img(img_path,cls,bbxs):
    img0 = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_name = img_path.split("/")[-1]
    for idx in range(len(cls)):
        if cls[idx] > 0.0:
            rec = list(map(int,bbxs[idx]))
            cv2.rectangle(img0,(rec[0],rec[1]),(rec[0]+rec[2],rec[1]+rec[3]),(0,255,0),2)
    cv2.imwrite(img_name,img0)

def produce_onet_detection_train_dataset():
    """
    产生用于训练onet的detection数据集:
    对于每张图片做如下处理：
    1.使用PNet检测，保留大于一定阈值的bbx
    2.使用RNet逐一检测这些bbx，保留剩下的大于一定阈值的bbx
    3.对这些bbx计算iou,分为part,neg,pos三类（对于数量较少的图片考虑增强）
    两个问题：
    1.两个网络的阈值多少？
    2.训练时与inference时，是否使用一样的阈值
    """
    fneg = open(cfg.ONET_TRAIN_NEG_TXT_PATH,"w")
    fpos = open(cfg.ONET_TRAIN_POS_TXT_PATH,"w")
    fpar = open(cfg.ONET_TRAIN_PART_TXT_PATH,"w")

    num_neg = 0
    num_par = 0
    num_pos = 0

    size = 48
    mtcnn = Mtcnn("RNet",[0.3,0.3,0.0])

    with open(cfg.ORIGINAL_FORMATTER_TXT_PATH,"r") as f:
        data = f.readlines();

    flag = 0
    total_num = len(data)

    for line in data:
        if flag > 7:
            break
        flag = flag + 1

        annotations = line.strip().split()
        img_path = cfg.ORIGINAL_IMG_PATH+annotations[0]
        print(img_path)

        true_boxes = list(map(float,annotations[1:]))
        true_boxes = np.array(true_boxes,dtype="float32").reshape(-1,4)

        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        cls,bbxs,_ = mtcnn.detect(img)

        # 查看图片
        draw_bbx_on_img(img_path,cls,bbxs)
        continue

        # 对剩余的框，为其打标签，pos,par,neg,并产生用于训练的图片
        neg_temp = 0
        par_temp = 0
        for box in bbxs:
            x,y,w,h = box

            # 可以不写
            if w < size or h < size or x < 0 or y < 0:
                continue

            ious = iou(box,true_boxes)
            max_idx = np.argmax(ious)
            miou = np.max(ious)

            if miou >= 0.4:
                tx,ty,tw,th = true_boxes[max_idx]

                offset_x = (tx-x)/w
                offset_y = (ty-y)/h
                offset_w = (tw-w)/w
                offset_h = (th-h)/h

            box = list(map(int,box))
            x,y,w,h = box
            crop_img = img[y:y+h,x:x+w]
            resized_img = cv2.resize(crop_img,(size,size))

            #注意预测的图片不是方形
            if miou >= 0.65: #pos
                #由于正例的缺乏，通过翻转、旋转来进行数据增强
                # for rotate_degree in np.arange(-10,15,5):
                    # is_flip = np.random.choice([False,True])
                for is_flip in [False,True]:
                    crop_img,_,bbx_regression = dataAugmentation(img,tbox,None,box,0,is_flip)

                    ch,cw,_ = np.shape(crop_img)

                    if cw*ch == 0:
                        continue

                    resized_img = cv2.resize(crop_img,(size,size))

                    offset_x,offset_y = bbx_regression[0]
                    offset_w,offset_h = bbx_regression[1]-bbx_regression[0]

                    cv2.imwrite("%spos_%d.jpg"%(cfg.RNET_TRAIN_IMG_PATH,num_pos),resized_img)
                    fpos.write("pos_%d.jpg 1 %f %f %f %f\n"%(num_pos,offset_x,offset_y,offset_w,offset_h))
                    num_pos = num_pos+1
            elif miou >= 0.4:
                cv2.imwrite("%spar_%d.jpg"%(cfg.RNET_TRAIN_IMG_PATH,num_par),resized_img)
                fpar.write("par_%d.jpg 0 %f %f %f %f\n"%(num_par,offset_x,offset_y,offset_w,offset_h))
                num_par = num_par+1
                par_temp = par_temp+1
            elif miou <= 0.3 and neg_temp < 80:
                cv2.imwrite("%sneg_%d.jpg"%(cfg.RNET_TRAIN_IMG_PATH,num_neg),resized_img)
                fneg.write("neg_%d.jpg -1\n"%(num_neg))
                num_neg = num_neg+1
                neg_temp = neg_temp+1

        num = num_neg+num_par+num_pos
        print("共需处理图片%d张，已经处理%d张，产生图片%d张：neg-%d张，par-%d张，pos-%d张"%(total_num,flag,num,num_neg,num_par,num_pos))
        # if num_neg > 750000 and num_par > 250000 and num_pos > 250000:
            # break

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

    num_landmark = 0
    size = 48
    flag = 0
    total_num = len(landmark_annotations)

    for ma in landmark_annotations:
        # if flag > 1000:
            # break
        flag = flag + 1

        ma = ma.strip().split()

        img_path = cfg.ORIGINAL_LANDMARK_IMG_PATH+ma[0].replace("\\","/")
        bbx = list(map(int,ma[1:5]))
        landmark = list(map(float,ma[5:]))

        bbx = np.array(bbx).reshape((-1,2))
        w,h = bbx[:,1]-bbx[:,0]
        x,y = bbx[:,0]

        true_boxes = np.array([x,y,w,h],dtype="float32").reshape(-1,4)
        landmark = np.array(landmark).reshape((-1,2))

        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        mtcnn = Mtcnn("RNet",[0.3,0.3,0.0])
        cls,bbxs,_ = mtcnn.detect(img)

        # 查看图片
        # draw_bbx_on_img(img_path,cls,bbxs)

        #对剩余的框，为pos框，记录landmark
        for box in boxes_cls_temp:
            box = box[:-1]
            rx,ry,rw,rh = box

            if rw < size or rh < size or rx < 0 or ry < 0:
                continue

            iou = iou(box,true_boxes)
            miou = np.max(iou)

            #如果是正例，则计算landmark
            if miou >= 0.65:
                for is_flip in np.array([False,True]):
                    for rotate_degree in np.arange(-10,15,5):
                        crop_img,landmark_regression,_ = dataAugmentation(img,None,landmark,box,rotate_degree,is_flip)

                        ch,cw,_ = np.shape(crop_img)

                        if cw*ch == 0:
                            continue

                        resized_img = cv2.resize(crop_img,(size,size))

                        cv2.imwrite("%slandmark_%d.jpg"%(cfg.RNET_TRAIN_IMG_PATH,num_landmark),resized_img)
                        landmark_regression = landmark_regression.reshape((-1))
                        landmark_regression = list(map(str,landmark_regression))
                        flandmark.write("landmark_%s.jpg 2 %s\n"%(num_landmark," ".join(landmark_regression)))
                        num_landmark = num_landmark+1

        print("共需处理图片%d张，已经处理%d张，产生landmark图片%d张"%(total_num,flag,num_landmark))

    flandmark.close()

if __name__ == "__main__":
    produce_onet_detection_train_dataset()
    # produce_rnet_landmark_train_dataset()

