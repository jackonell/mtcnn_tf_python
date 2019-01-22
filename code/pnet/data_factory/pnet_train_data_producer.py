import numpy as np
import cv2
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg
from mtcnn_utils import IOU

def detection_data():
    """
    产生用于训练pnet的数据:
    产生的框为方形
    """
    fneg = open(cfg.PNET_TRAIN_NEG_TXT_PATH,"w")
    fpos = open(cfg.PNET_TRAIN_POS_TXT_PATH,"w")
    fpar = open(cfg.PNET_TRAIN_PART_TXT_PATH,"w")
    flandmark = open(cfg.PNET_TRAIN_LANDMARK_TXT_PATH,"w")

    img_annotations = ""
    with open(cfg.PNET_TRAIN_FORMATTER_TXT_PATH,"r") as f:
        img_annotations = f.readlines()
    flag = 0

    for ia in img_annotations:
        if flag > 0:
            break;
        flag = flag+1;

        #计数所有图片
        idx = 0

        ia = ia.strip().split(" ")
        #图像路径
        img_path = cfg.PNET_ORIGINAL_IMG_PATH + ia[0]
        #gt框
        bbxs = list(map(float,ia[1:]))
        bbxs = np.array(bbxs,dtype="float32").reshape(-1,4)

        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img_width = img.shape[0]
        img_height = img.shape[1]

        #产生neg数据
        # idx_neg = 0
        # while idx_neg  < 50:
           # #产生多大的框合适呢？范围：[12,?],最大值先选取短边的一半
           # size = np.random.randint(12,min(img_width,img_height)/2)

           # nx = np.random.randint(0,img_width-size)
           # ny = np.random.randint(0,img_height-size)

           # nbox = np.array([nx,ny,size,size])
           # iou = IOU(nbox,bbxs)

           # if np.max(iou) < 0.3:
               # crop_img = img[ny:ny+size,nx:nx+size]
               # crop_img = cv2.resize(crop_img,(12,12))
               # cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"neg_"+str(idx)+".jpg",crop_img)
               # fneg.write("neg_%s.jpg 0\n"%idx)
               # idx = idx+1
               # idx_neg = idx_neg+1

        for bbx in bbxs:
           x,y,w,h = bbx
           print(bbx)

           # for i in range(5):
               # size = np.random.randint(12,min(img_width,img_height)/2)
               # #确保相交
               # nx = np.random.randint(max(0,x-size),min(x+w,img_width))
               # ny = np.random.randint(max(0,y-size),min(y+h,img_height))

               # nbox = np.array([nx,ny,size,size])
               # iou = IOU(nbox,bbxs)
               # print(nbox)
               # print(iou)
               # if np.max(iou) < 0.3:
                   # crop_img = img[ny:ny+size,nx:nx+size]
    # #               crop_img = cv2.resize(crop_img,(12,12))
                   # cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"neg_"+str(idx)+".jpg",crop_img)
                   # fneg.write("neg_%s.jpg 0\n"%idx)
                   # idx = idx+1

           for i in range(20):
               # # pos and part face size [minsize*0.8,maxsize*1.25]
               # size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

               # #print (box)
               # delta_x = np.random.randint(-w * 0.2, w * 0.2)
               # delta_y = np.random.randint(-h * 0.2, h * 0.2)

               # #show this way: nx1 = max(x1+w/2-size/2+delta_x)
               # # x1+ w/2 is the central point, then add offset , then deduct size/2
               # # deduct size/2 to make sure that the right bottom corner will be out of
               # nx = int(max(x + w / 2 + delta_x - size / 2, 0))
               # #show this way: ny1 = max(y1+h/2-size/2+delta_y)
               # ny = int(max(y + h / 2 + delta_y - size / 2, 0))
               #需要保证IOU的值较大
               size = np.random.randint(min(w,h)*0.8,max(w,h)*1.2)

               nx = np.random.randint(max(0,x-0.2*size),min(x+w-0.8*size,img_width))
               ny = np.random.randint(max(0,y-0.2*size),min(y+h-0.8*size,img_height))

               nbox = np.array([nx,ny,size,size])
               iou = IOU(nbox,bbxs)
               print(nbox)
               print(iou)
               if np.max(iou) > 0.65:
                   crop_img = img[ny:ny+size,nx:nx+size]
    #               crop_img = cv2.resize(crop_img,(12,12))
                   cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"pos_"+str(idx)+".jpg",crop_img)
                   fneg.write("pos_%s.jpg 0\n"%idx)
                   idx = idx+1
               elif np.max(iou) > 0.4:
                   crop_img = img[ny:ny+size,nx:nx+size]
    #               crop_img = cv2.resize(crop_img,(12,12))
                   cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"par_"+str(idx)+".jpg",crop_img)
                   fneg.write("par_%s.jpg 0\n"%idx)
                   idx = idx+1



if __name__ == "__main__":
    detection_data()



