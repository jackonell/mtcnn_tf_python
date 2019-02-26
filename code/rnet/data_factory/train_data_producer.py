import tensorflow as tf
import math
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg
from mtcnn_utils import IOU

class detector(object):

    """用于产生rnet与onet所需的训练数据"""

    def __init__(self):
        sess = tf.Session()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(cfg.PNET_MODEL_PATH)
        if ckpt and ckpt.model_checkpint_path:
            saver.restore(sess,ckpt.model_checkpint_path)

        width = tf.placeholder(tf.float32,name="width")
        height = tf.placeholder(tf.float32,name="height")
        img = tf.placeholder(tf.float16,shape=[None,width,height,3],name="IMG")

        self.fcls_pred,self.bbr_pred,landmark_pred = PNet(img)

    def predict(self, img):
        """
        预测结果
        """
        cls,bbr,landmark = sess.run([self.fcls_pred,self.bbr_pred,self.landmark_pred],feed_dict={"IMG:0":img})
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
    field,jump,start = calc_real_coordinate()

    field = (field/ratio)
    jump  = (jump/ratio)
    start = (start/ratio)

    output_shape = np.shape(bbox)
    height = output_shape[0]
    width = output_shape[1]

    bbox = bbox*field
    remain = []
    for i in range(height):
        for j in range(width):
           centerx = start + j*jump
           centery = start + i*jump
           x = centerx - field/2
           y = centery - field/2

           #得分大于0.5，判为正例,优化输出框
           if cls[i,j] > 0.5:
               x = x+bbox[i,j,0]
               y = y+bbox[i,j,1]
               w = field+bbox[i,j,2]
               h = filed+bbox[i,j,3]
               remain.append([x,y,w,h])

    return remain

def produce_rnet_detection_train_dataset():
    """
    产生用于训练rnet的detection数据集
    """
    fneg = open(cfg.RNET_TRAIN_NEG_TXT_PATH,"w")
    fpos = open(cfg.RNET_TRAIN_POS_TXT_PATH,"w")
    fpar = open(cfg.RNET_TRAIN_PART_TXT_PATH,"w")

    num_neg = 1
    num_par = 1
    num_pos = 1

    with open(cfg.PNET_TRAIN_FORMATTER_TXT_PATH,"r") as f:
        data = f.readlines();

    detector = detector()

    for line in data:
        annotations = line.strip().split()
        img_path = cfg.PNET_ORIGINAL_IMG_PATH+annotations[0]
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)

        true_boxes = list(map(float,annotations[1:]))
        true_boxes = np.array(bbxs,dtype="float32").reshape(-1,4)

        width = img.shape[1]
        height = img.shape[0]

        ratio = 1
        #临时存放每张图片上产生的所有输出框
        boxes_cls_temp = []
        while min(width,height) > 24 :
            cls,bbr,_ = detector.predict(img)

            #计算出预测正例的真实坐标值
            keep_boxes = calc_real_coordinate(ratio,bbr,cls)
            boxes_cls_temp.extend(keep_boxes)

            ratio  = ratio*0.79
            width  = width*0.79
            height = height*0.79

        remain = nms(boxes_cls_temp[:,:4],boxes_cls_temp[4],0.5)
        #对剩余的框，为其打标签，pos,par,neg,并产生用于训练的图片
        for i in range(len(remain)):
            box = remain[i]
            x,y,w,h = box

            if w < 24 || h < 24 || x < 0 || y < 0:
                continue

            iou = IOU(box,true_boxes)
            max_idx = np.argmax(iou)

            tx,ty,tw,th = true_boxes[max_idx]

            offset_x = (tx-x)/w
            offset_y = (ty-y)/h
            offset_w = (tw-w)/w
            offset_h = (th-h)/h

            crop_img = img[y:y+h,x:x+w]
            resized_img = cv2.resize(crop_img,(24,24))

            #注意预测的图片不是方形
            if np.max(iou) >= 0.65: #pos
                cv2.imwrite(cfg.RNET_TRAIN_POS_TXT_PATH+"pos_"+str(num_pos)+".jpg",resized_img)
                fpos.write("pos_%s.jpg 1 %f %f %f %f\n"%(num_pos,offset_x,offset_y,offset_w,offset_h))
                num_pos = num_pos+1
            elif np.max(iou) >= 0.4:
                cv2.imwrite(cfg.RNET_TRAIN_PART_TXT_PATH+"pos_"+str(num_par)+".jpg",resized_img)
                fpar.write("par_%s.jpg 0 %f %f %f %f\n"%(num_par,offset_x,offset_y,offset_w,offset_h))
                num_par = num_par+1
            elif np.max(iou) <= 0.3:
                cv2.imwrite(cfg.RNET_TRAIN_NEG_TXT_PATH+"pos_"+str(num_neg)+".jpg",resized_img)
                fneg.write("neg_%s.jpg -1 %f %f %f %f\n"%(num_neg,offset_x,offset_y,offset_w,offset_h))
                num_neg = num_neg+1


        num = num_neg+num_par+num_pos
        print("一共产生图片%d张：neg-%d张，par-%d张，pos-%d张"%(num,num_neg,num_par,num_pos))
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

    with open(cfg.PNET_ORIGINAL_LANDMARK_TXT_PATH,"r") as f:
        landmark_annotations = f.readlines()

    detector = detector()
    num_landmark = 1

    for ma in landmark_annotations:
        ma = ma.strip().split()

        img_path = cfg.PNET_ORIGINAL_LANDMARK_IMG_PATH+ma[0].replace("\\","/")
        bbx = list(map(int,ma[1:5]))
        landmark = list(map(float,ma[5:]))
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)

        width = img.shape[0]
        height = img.shape[1]

        bbx = np.array(bbx).reshape((-1,2))
        w,h = bbx[:,1]-bbx[:,0]
        x,y = bbx[:,0]

        landmark = np.array(landmark).reshape((-1,2))

        ratio = 0.79

        while width > 24 and height > 24:
            cls,bbr,_ = detector.predict(img)

            #计算出预测正例的真实坐标值
            keep_boxes = calc_real_coordinate(ratio,bbr,cls)
            boxes_cls_temp.extend(keep_boxes)

            ratio  = ratio*0.79
            width  = width*0.79
            height = height*0.79

        remain = nms(boxes_cls_temp[:,:4],boxes_cls_temp[4],0.5)
        #对剩余的框，为pos框，记录landmark
        for i in range(len(remain)):
            box = remain[i]
            rx,ry,rw,rh = box

            if rw < 24 || rh < 24 || rx < 0 || ry < 0:
                continue

            bbx = np.array([x,y,w,h])
            iou = IOU(np.array(box,bbx.reshap((-1,4)))

            #如果是正例，则计算landmark
            if np.max(iou) >= 0.65:
                #归一化特征点
                landmark[:,0] = (landmark[:,0]-rx)/rw
                landmark[:,1] = (landmark[:,1]-ry)/rh

                crop_img = img[ry:ry+rh,rx:rx+rw]
                resized_img = cv2.resize(crop_img,(24,24))

                resized_img = cv2.resize(crop_img,(12,12))
                cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"landmark_"+str(num_landmark)+".jpg",resized_img)
                landmark = landmark.reshape((-1))
                landmark = list(map(str,landmark))
                flandmark.write("landmark_%s.jpg 2 %s\n"%(num_landmark," ".join(landmark)))
                num_landmark = num_landmark+1

        print("一共产生landmark图片%d张"%(num_landmark-1))

    flandmark.close()

if __name__ == "__main__":
    produce_rnet_detection_train_dataset()
    # produce_rnet_landmark_train_dataset()

