import numpy as np
import cv2
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg
from mtcnn_utils import IOU

def detection_data():
    """
    产生用于训练pnet的数据
    """
    img_annotations = ""
    with open(cfg.PNET_FORMATTER_TXT_PATH,"r") as f:
        img_annotations = f.readlines()
    flag = 0

    for ia in img_annotations:
        if flag > 0:
            break;
        flag = flag+1;

        ia = ia.strip().split(" ")
        #图像路径
        img_path = cfg.PNET_ORIGINAL_IMG_PATH + ia[0]
        #gt框
        bbxs = list(map(float,ia[1:]))
        bbxs = np.array(bbxs,dtype="float32").reshape(-1,4)

        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img_width = img.shape[0]
        img_height = img.shape[1]

        #保证比例：3:1:1:2(neg:pos:part:landmark)
        #每张图片产生数据的方式：
        #   若有框X个，
        #   则先随机产生5X个neg
        #   随后在每个框附近产生5个pos,5个part
        #产生neg数据
        gt_num = bbxs.shape[0]
        idx = 0
        while idx < 5*gt_num:
            rb  = np.random.rand(4)
            #如果产生的框超出边界
            if rb[0]+rb[2] > 1 or rb[1]+rb[3]>1 :
                continue
            #如果产生的框太小
            if rb[2]*img_width < 12 or rb[3]*img_height < 12:
                continue

            rb[0] *= img_width
            rb[1] *= img_height
            rb[2] *= img_width
            rb[3] *= img_height
            rb[2] += rb[0]
            rb[3] += rb[1]

            score,_ = IOU(rb,bbxs)

            if score > 0.3:
                continue

            print(score,rb)

            left   = int(rb[0])
            right  = int(rb[2])
            top    = int(rb[1])
            bottom = int(rb[3])

            cimg = img[left:right,top:bottom]
            cv2.imwrite(str(idx)+".jpg",cimg)
            idx = idx+1


if __name__ == "__main__":
    detection_data()



