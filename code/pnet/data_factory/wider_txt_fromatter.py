import numpy as np
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg

def format_data_file():
    """
    将检测人脸的txt文件格式化
    """
    content = ""
    with open(cfg.PNET_ORIFINAL_TXT_PATH,"r") as f:
       content = f.read()

    content_list = content.split("\n")

    res = ""
    count = 0
    cend = 0
    for oneline in content_list:
        if count == 1:
            cend = int(oneline)
        elif count == 0:
            res += oneline
        else:
            temp = oneline.split()
            res += " " +  ' '.join(temp[0:4])
        count = count+1
        if count == cend+2:
            count=0
            res += "\n"

    with open(cfg.PNET_FORMATTER_TXT_PATH,"w") as f:
        f.write(res)

if __name__ == "__main__":
    format_data_file()
