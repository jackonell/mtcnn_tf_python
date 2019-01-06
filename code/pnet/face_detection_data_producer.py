import numpy as np

def format_data_file():
    """
    将检测人脸的txt文件格式化
    """
    content = ""
    with open("/home/brooks/deeplearning/face/mtcnn_tf_python/data/origin/wider_face_split/wider_face_train_bbx_gt.txt","r") as f:
       content = f.read()

    content_list = content.split("\n")

    res = ""
    count = 0
    cend = 0
    for oneline in content_list:
        if count == 1:
            cend = int(oneline)
        elif count == 0:
            res += oneline+" "
        else:
            temp = oneline.split()
            res += ' '.join(temp[0:4])
        count = count+1
        if count == cend+2:
            count=0
            res += "\n"

    with open("/home/brooks/deeplearning/face/mtcnn_tf_python/data/pdata/wider_face_train_bbx_gt.txt","w") as f:
        f.write(res)

if __name__ == "__main__":
    format_data_file()
