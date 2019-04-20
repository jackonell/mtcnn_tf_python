import numpy as np
import cv2

def dataAugmentation(img,gtbox,landmark,bbx,rotate,is_flip):
    """
    对图片进行旋转和翻转

    :img: 图片
    :gtbox: 真实框,转换为（x1,y1,w,h）
    :landmark: 特征点
    :bbx: 建议框
    :rotate: 旋转角度
    :is_flip: 是否翻转
    """
    width = img.shape[1]
    height = img.shape[0]

    bbx = np.array(bbx).reshape((-1,2))

    #转为（x1,y1,x2,y2）
    bbx[1] = bbx[0]+bbx[1]

    if is_flip:
        img = cv2.flip(img,1)
        bbx[:,0] = width-bbx[:,0]

    matrix = cv2.getRotationMatrix2D((width/2,height/2),rotate,1)
    dst = cv2.warpAffine(img,matrix,(width,height))

    #建议框
    bbx = rotate_box(bbx,matrix)
    crop_img = dst[bbx[0,1]:bbx[1,1],bbx[0,0]:bbx[1,0]]

    if landmark is not None:
        if is_flip:
            landmark[:,0] = width-landmark[:,0]

        landmark = np.array(landmark).reshape((-1,2))
        # 特征点旋转
        ones = np.ones(5).reshape((-1,1))
        landmark = np.hstack((landmark,ones))
        landmark = np.dot(landmark,matrix.T)

        #归一化特征点
        landmark = (landmark-bbx[0])/(bbx[1]-bbx[0])

        # landmark = landmark*(bbx[1]-bbx[0])

        # for ma in landmark:
            # cv2.circle(crop_img,(int(ma[0]),int(ma[1])),3,(0,0,225),-1)
        # cv2.imwrite("1.jpg",crop_img)

    elif gtbox is not None:
        gtbox = np.array(gtbox).reshape((-1,2))

        #转为（x1,y1,x2,y2）
        gtbox[1] = gtbox[0]+gtbox[1]

        if is_flip:
            gtbox[:,0] = width-gtbox[:,0]

        #gt box 旋转
        gtbox = rotate_box(gtbox,matrix)

        #回归量
        bbx = (gtbox-bbx)/(bbx[1]-bbx[0])

    return crop_img,landmark,bbx

def rotate_box(box,matrix):
    """
    旋转人脸框
    :box: (x1,y1,x2,y2)
    """
    ones = np.ones(5).reshape((-1,1))
    tbox = np.copy(box)
    tbox[:,1] = tbox[:,1][::-1]

    box = np.vstack((box,tbox))
    box = np.hstack((box,ones[:4]))
    box = np.dot(box,matrix.T)

    #如果不加这个，翻转后的坐标会变成右上角和左下角
    maxx = int(np.max(box[:,0]))
    minx = int(np.min(box[:,0]))
    maxy = int(np.max(box[:,1]))
    miny = int(np.min(box[:,1]))

    bbx = np.array([minx,miny,maxx,maxy])

    return bbx.reshape((-1,2))


def NMS(bbxs,confidences,thresh):
    """
    对计算出的框进行极大值抑制（重叠度过高的框进行删减）
    """
    order_idx =  np.argsort(confidences)[::-1]
    remain = []

    while order_idx.size > 0:
        # 记录得分最大的框的索引
        cidx = order_idx[0]
        remain.append(cidx)
        bbx = bbxs[cidx]
        target = bbxs[order_idx[1:]]
        iou = IOU(bbx,target)

        # print(order_idx)
        # print(iou)
        idxs = np.where(iou <= thresh)
        idxs = np.array(idxs)[0]
        # print(idxs+1)
        # 因为idxs是以原数组第二个位置为基准的坐标,所以要+1
        order_idx = order_idx[idxs+1]

    return remain


def IOU(bbx,target):
    """
     计算IOU得分:
     (Area of Overlap)/(Area of Union)
     要有python矩阵计算的思维，尽量少用for循环

    :bbx: 需要判定的框
    :target: 进行IOU计算的真实bbx集合
    :returns: iou最高得分与对应gt bbox

    """
    bbx_area    = bbx[2]*bbx[3]
    target_area = target[:,2]*target[:,3]

    nx1 = np.maximum(bbx[0],target[:,0])
    ny1 = np.maximum(bbx[1],target[:,1])
    nx2 = np.minimum(bbx[0]+bbx[2],target[:,0]+target[:,2])
    ny2 = np.minimum(bbx[1]+bbx[3],target[:,1]+target[:,3])

    width  = np.maximum(0,nx2-nx1)
    height = np.maximum(0,ny2-ny1)

    inter_area = width*height

    iou = inter_area/(bbx_area+target_area-inter_area)

    return iou


if __name__ == "__main__":
    # a = np.array([3,3,2,2])
    # b = np.array([0,0,2,2])

    # for x in range(6):
        # for y in range(6):
            # t = np.array([x,y,2,2])
            # b = np.vstack((b,t))

    # s = IOU(a,b)
    # print(np.hstack((b,s.reshape(-1,1))))
    # confidences = []
    # bbxs = []

    # box = np.array([17,25,36,48])
    # box = box.reshape((-1,2))
    # gtbox = np.array([25,84,42,96])
    # gtbox = gtbox.reshape((-1,2))

    # bbx = (gtbox-box)/(box[1]-box[0])
    # print(bbx)

    box = np.array([17,25,36,48])

    box = box[[1,2]]
    box = list(box) 
    print(type(box))

