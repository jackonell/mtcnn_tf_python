import numpy as np
import cv2

def draw_landmark():
    with open("/root/face/alignment/mtcnn_tf_python/data/odata/train_landmark.txt","r") as f:
        landdata = f.readlines()

    landdata = landdata[:30]

    for land in landdata:
        an = land.split()
        imgname = an[0]
        anas = np.array(an[2:],dtype=np.float32)
        imgpath = "/root/face/alignment/mtcnn_tf_python/data/odata/images/%s"%imgname
        img = cv2.imread(imgpath)
        anas = anas*48
        anas = anas.reshape(-1,2)
        # for ma in anas:
        ma = anas[3]
        cv2.circle(img,(int(ma[0]),int(ma[1])),1,(0,0,225),-1)
        cv2.imwrite(imgname,img)


if __name__ == "__main__":
    draw_landmark()
