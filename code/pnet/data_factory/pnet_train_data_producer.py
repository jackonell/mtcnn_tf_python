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

        #产生neg数据
        idx = 0
        while idx < 50:
           #产生多大的框合适呢？范围：[12,?],最大值先选取短边的一半
           size = np.random.randint(12,np.minimum(img_width,img_height)/2)


if __name__ == "__main__":
    detection_data()



