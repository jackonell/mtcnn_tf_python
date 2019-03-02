import numpy as np
import cv2
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg
from mtcnn_utils import IOU

def format_data_file():
    """
    将检测人脸的txt文件格式化
    """
    content = ""
    with open(cfg.ORIFINAL_TXT_PATH,"r") as f:
       content = f.read()

    content_list = content.split("\n")

    res = ""
    count = 0
    cend = 0
    nn = 0
    for oneline in content_list:
        if count == 1:
            cend = int(oneline)
        elif count == 0:
            res += oneline
        else:
            temp = oneline.split()
            res += " " +  ' '.join(temp[0:4])
            if int(temp[2]) > 20 and  int(temp[3]) > 20:
                nn = nn+1
        count = count+1
        if count == cend+2:
            count=0
            res += "\n"

    with open(cfg.PNET_TRAIN_FORMATTER_TXT_PATH,"w") as f:
        f.write(res)

    print(nn)

def landmark_crop_rotate_flip(img,bbx,landmark,rotate,is_flip):
    """
    从原始landmark图片和landmark产生新的sample

    :img: 图片
    :bbx: 人脸框
    :landmark: 特征点
    :is_crop: 是否剪裁
    :rotate: 旋转角度
    :is_flip: 是否翻转

    """
    bbx = np.array(bbx).reshape((-1,2))
    landmark = np.array(landmark).reshape((-1,2))
    width = img.shape[0]
    height = img.shape[1]

    w,h = bbx[:,1]-bbx[:,0]
    x,y = bbx[:,0]

    size = 0
    nx = 0
    ny = 0

    # 产生iou大于0.65的框
    ioum = 0
    while ioum < 0.65:
       #需要保证IOU的值较大
       size = np.random.randint(min(w,h)*0.8,max(w,h)*1.2)

       #确定中心点的范围，而后知左上角
       nx = np.random.randint(x+0.3*w-0.5*size,x+0.7*w-0.5*size)
       ny = np.random.randint(y+0.3*h-0.5*size,y+0.7*h-0.5*size)

       nx = max(nx,0)
       ny = max(ny,0)

       if nx+size > width or ny+size > height:
           continue

       nbox = np.array([nx,ny,size,size])
       target = np.array([x,y,w,h])

       #依据当前框计算
       iou = IOU(nbox,target.reshape(1,-1))
       ioum = np.max(iou)

    bbx = np.array([nx,nx+size,ny,ny+size]).reshape((-1,2))

    # 翻转处理
    if is_flip:
        img = cv2.flip(img,1)
        bbx[0,[0,1]] = width-bbx[0,[1,0]]
        landmark[:,0] = width-landmark[:,0]

    # 旋转处理
    # 图像旋转
    matrix = cv2.getRotationMatrix2D((width/2,height/2),rotate,1)
    dst = cv2.warpAffine(img,matrix,(width,height))

    # 特征点旋转
    ones = np.ones(5).reshape((-1,1))
    landmark = np.hstack((landmark,ones))
    landmark = np.dot(landmark,matrix.T)
    # 人脸框旋转
    tbbx = np.copy(bbx)
    tbbx[1,[0,1]] = tbbx[1,[1,0]]
    # 四个点
    bbx = np.vstack((bbx.T,tbbx.T))
    bbx = np.hstack((bbx,ones[:4]))
    bbx = np.dot(bbx,matrix.T)


    maxx = int(np.max(bbx[:,0]))
    minx = int(np.min(bbx[:,0]))
    maxy = int(np.max(bbx[:,1]))
    miny = int(np.min(bbx[:,1]))

    # print(x,y,w,h)
    # print(nx,ny,size)
    # print(bbx)
    # print(rotate)
    # print(minx,maxx,miny,maxy)

    # print(ioum)
    # 裁剪图像
    crop_img = dst[miny:maxy,minx:maxx]

    # 规一化特征点
    landmark[:,0] = (landmark[:,0]-minx)/size
    landmark[:,1] = (landmark[:,1]-miny)/size

    # landmark = landmark*size

    # for ma in landmark:
        # cv2.circle(crop_img,(int(ma[0]),int(ma[1])),3,(0,0,225),-1)
    # cv2.imwrite(filename,crop_img)
    return crop_img,landmark

def landmark_data():
    """
    产生用于训练pnet的landmark数据
    数据产生方式：
    1.iou大于0.65的剪裁(10)
    2.左右小幅度旋转(5度，10度)
    3.水平翻转
    """
    flandmark = open(cfg.PNET_TRAIN_LANDMARK_TXT_PATH,"w")

    landmark_annotations = ""
    with open(cfg.ORIGINAL_LANDMARK_TXT_PATH,"r") as f:
        landmark_annotations = f.readlines()

    flag = 0
    num_landmark = 1

    for ma in landmark_annotations:
        # if flag > 0:
            # break
        # flag = flag+1
        # ma = "net_7876\\111_0_0.jpg 38 236 94 292 91.000000 149.000000 196.000000 141.000000 174.000000 206.000000 115.000000 262.000000 198.000000 256.000000"
        ma = ma.strip().split(" ")

        img_path = cfg.ORIGINAL_LANDMARK_IMG_PATH+ma[0].replace("\\","/")
        bbx = list(map(int,ma[1:5]))
        landmark = list(map(float,ma[5:]))
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)

        print(img_path)
        for i in range(5):
            for is_crop in [True,False]:
                for rotate_degree in np.arange(-10,15,5):
                    crop_img,nlandmark = landmark_crop_rotate_flip(img,bbx,landmark,rotate_degree,is_crop)
                    # 如果人脸框旋转后超出了图片，则忽略当前
                    if crop_img.shape[0]*crop_img.shape[1] <= 0:
                        continue
                    resized_img = cv2.resize(crop_img,(12,12))
                    cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"landmark_"+str(num_landmark)+".jpg",resized_img)
                    nlandmark = nlandmark.reshape((-1))
                    nlandmark = list(map(str,nlandmark))
                    flandmark.write("landmark_%s.jpg 2 %s\n"%(num_landmark," ".join(nlandmark)))
                    num_landmark = num_landmark+1

        print("一共产生landmark图片%d张"%(num_landmark-1))

    flandmark.close()



def detection_data():
    """
    产生用于训练pnet的数据:
    产生的框为方形
    """
    fneg = open(cfg.PNET_TRAIN_NEG_TXT_PATH,"w")
    fpos = open(cfg.PNET_TRAIN_POS_TXT_PATH,"w")
    fpar = open(cfg.PNET_TRAIN_PART_TXT_PATH,"w")

    img_annotations = ""
    with open(cfg.PNET_TRAIN_FORMATTER_TXT_PATH,"r") as f:
        img_annotations = f.readlines()
    flag = 0
    #图片计数
    num_neg = 1
    num_pos = 1
    num_par = 1
    num = 1

    #一共产生图片1919240张：neg-946872张，par-689319张，pos-283049张
    #12800 images done, pos: 456503 part: 1127233 neg: 995601
    for ia in img_annotations:
        # if flag > 100:
            # break;
        # flag = flag+1;

        ia = ia.strip().split(" ")
        #图像路径
        img_path = cfg.ORIGINAL_IMG_PATH + ia[0]
        print(img_path)
        #gt框
        bbxs = list(map(float,ia[1:]))
        bbxs = np.array(bbxs,dtype="float32").reshape(-1,4)

        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img_width = img.shape[1]
        img_height = img.shape[0]

        #产生neg数据
        idx_neg = 0
        while idx_neg  < 50:
           #产生多大的框合适呢？范围：[12,?],最大值先选取短边的一半
           size = np.random.randint(12,min(img_width,img_height)/2)

           nx = np.random.randint(0,img_width-size)
           ny = np.random.randint(0,img_height-size)

           nbox = np.array([nx,ny,size,size])
           iou = IOU(nbox,bbxs)

           if np.max(iou) < 0.3:
               crop_img = img[ny:ny+size,nx:nx+size]
               resized_img = cv2.resize(crop_img,(12,12))
               cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"neg_"+str(num_neg)+".jpg",resized_img)
               fneg.write("neg_%s.jpg -1\n"%num_neg)
               num_neg = num_neg+1
               idx_neg = idx_neg+1

        for bbx in bbxs:
           x,y,w,h = bbx

           if w < 20 or h < 20 or x < 0 or y <0:
               continue

           for i in range(5):
               size = np.random.randint(12,min(img_width,img_height)/2)
               #确保相交
               nx = np.random.randint(max(0,x-size),x+w)
               ny = np.random.randint(max(0,y-size),y+h)

               if nx+size > img_width or ny+size > img_height:
                   continue

               nbox = np.array([nx,ny,size,size])
               iou = IOU(nbox,bbxs)

               if np.max(iou) < 0.3:
                   crop_img = img[ny:ny+size,nx:nx+size]
                   resized_img = cv2.resize(crop_img,(12,12))
                   cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"neg_"+str(num_neg)+".jpg",resized_img)
                   fneg.write("neg_%s.jpg -1\n"%num_neg)
                   num_neg = num_neg+1

           for i in range(20):
               #需要保证IOU的值较大
               size = np.random.randint(min(w,h)*0.8,max(w,h)*1.2)

               #确定中心点的范围，而后知左上角
               nx = np.random.randint(x+0.3*w-0.5*size,x+0.7*w-0.5*size)
               ny = np.random.randint(y+0.3*h-0.5*size,y+0.7*h-0.5*size)

               nx = max(nx,0)
               ny = max(ny,0)

               if nx+size > img_width or ny+size > img_height:
                   continue

               nbox = np.array([nx,ny,size,size])

               offset_x = (x-nx)/float(size)
               offset_y = (y-ny)/float(size)
               offset_w = (w-size)/float(size)
               offset_h = (h-size)/float(size)

               #依据当前框计算
               iou = IOU(nbox,bbx.reshape(1,-1))

               if np.max(iou) >= 0.65:
                   crop_img = img[ny:ny+size,nx:nx+size]
                   resized_img = cv2.resize(crop_img,(12,12))
                   cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"pos_"+str(num_pos)+".jpg",resized_img)
                   fpos.write("pos_%s.jpg 1 %f %f %f %f\n"%(num_pos,offset_x,offset_y,offset_h,offset_w))
                   num_pos = num_pos+1

               elif np.max(iou) >= 0.4:
                   crop_img = img[ny:ny+size,nx:nx+size]
                   resized_img = cv2.resize(crop_img,(12,12))
                   cv2.imwrite(cfg.PNET_TRAIN_IMG_PATH+"par_"+str(num_par)+".jpg",resized_img)
                   fpar.write("par_%s.jpg 0 %f %f %f %f\n"%(num_par,offset_x,offset_y,offset_h,offset_w))
                   num_par = num_par+1

        num = num_neg+num_par+num_pos
        print("一共产生图片%d张：neg-%d张，par-%d张，pos-%d张"%(num,num_neg,num_par,num_pos))
        if num_neg > 750000 and num_par > 250000 and num_pos > 250000:
            break

    fneg.close()
    fpar.close()
    fpos.close()

if __name__ == "__main__":
    # detection_data()
    landmark_data()

