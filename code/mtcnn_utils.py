import numpy as np

def nms(bbxs,confidences,thresh):
    """
    对计算出的框进行极大值抑制（重叠度过高的框进行删减）

    :bbxs: TODO
    :scores: TODO
    :returns: TODO

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

        idxs = np.where(iou <= thresh)
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
    confidences = []
    bbxs = []
