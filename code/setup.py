import os
from net.cfgs import cfg

def create_if_not_exsits(path):
    """
    生成数据目录
    """
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    create_if_not_exsits("../data")
    create_if_not_exsits(cfg.TRAIN_IMG_PATH%cfg.PNET_DIR)
    create_if_not_exsits(cfg.TRAIN_IMG_PATH%cfg.RNET_DIR)
    create_if_not_exsits(cfg.TRAIN_IMG_PATH%cfg.ONET_DIR)
    create_if_not_exsits(cfg.ORIGINAL_LANDMARK_IMG_PATH)

