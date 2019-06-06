import tensorflow as tf
import numpy as np
import cv2
import time
from net.utils import draw_bbx_on_img
from net.mtcnn import Mtcnn


if __name__ == "__main__":
    mtcnn = Mtcnn("ONet",[0.6,0.6,0.7])

    start = time.time()
    img_path = "/root/face/alignment/mtcnn_tf_python/code/6.png"
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    cls,bbxs,landmark = mtcnn.detect(img)
    print("时长：%r"%(time.time()-start))
    draw_bbx_on_img(img_path,cls,bbxs,landmark)
    print(bbxs)

