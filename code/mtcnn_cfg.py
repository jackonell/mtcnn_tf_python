from easydict import EasyDict as edict

cfg = edict()

#原始数据集中的索引与标签文件
cfg.PNET_ORIGINAL_TXT_PATH = "../data/origin/wider_face_split/wider_face_train_bbx_gt.txt"
#原始数据集中图片地址
cfg.PNET_ORIGINAL_IMG_PATH = "../data/origin/WIDER_train/images/"
#原始特征点数据集中的索引与标签文件
cfg.PNET_ORIGINAL_LANDMARK_TXT_PATH = "../data/origin/trainImageList.txt"
#原始特征点数据集中的图片地址
cfg.PNET_ORIGINAL_LANDMARK_IMG_PATH = "../data/origin/"

#-------------pnet-------------------
#经过规范化后的索引与标签文件
cfg.PNET_TRAIN_FORMATTER_TXT_PATH = "../data/pdata/train_original_formated.txt"
#以下四项分别为neg,pos,part与landmark类型图片文件的索引与标签文件
cfg.PNET_TRAIN_NEG_TXT_PATH = "../data/pdata/train_neg.txt"
cfg.PNET_TRAIN_POS_TXT_PATH = "../data/pdata/train_pos.txt"
cfg.PNET_TRAIN_PART_TXT_PATH = "../data/pdata/train_part.txt"
cfg.PNET_TRAIN_LANDMARK_TXT_PATH = "../data/pdata/train_landmark.txt"
cfg.PNET_TRAIN_IMG_PATH = "../data/pdata/images/"
cfg.PNET_TRAIN_TFRECORDS = "../data/pdata/train.tfrecords"

