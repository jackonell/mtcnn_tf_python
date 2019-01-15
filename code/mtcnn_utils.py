import numpy as np

def IOU(bbx,target):
    """
     计算IOU得分:
     (Area of Overlap)/(Area of Union)

    :bbx: 需要判定的框
    :target: 进行IOU计算的真实bbx集合
    :returns: iou最高得分与对应gt bbox

    """
    score = 0
    rbbx = np.zeros(4)
    bbx_area = bbx[2]*bbx[3]

    for gtb in target:
        min_left = min(bbx[0],gtb[0])
        max_right = max(bbx[0]+bbx[2],gtb[0]+gtb[2])
        inter_width = bbx[2]+gtb[2]-(max_right-min_left)

        if inter_width <= 0:
            continue

        min_top = min(bbx[1],gtb[1])
        max_bottom = max(bbx[1]+bbx[3],gtb[1]+gtb[3])
        inter_height = bbx[3]+gtb[3]-(max_bottom-min_top)

        if inter_height <= 0:
            continue

        gtarea = gtb[2]*gtb[3]
        interarea = inter_width*inter_height

        tscore = interarea/(gtarea+bbx_area-interarea)

        if tscore > score:
            score = tscore
            rbbx = gtb

    return score,rbbx

if __name__ == "__main__":
    a = np.array([3,3,2,2])
    b = np.array([2,2,4,4])
    s,_ = IOU(a,b.reshape(-1,4))
    print(s,b)
