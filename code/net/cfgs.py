from easydict import EasyDict as edict

cfg = edict()

cfg.BATCH_SIZE = 224

#原始数据集中的索引与标签文件
cfg.ORIGINAL_TXT_PATH          = "../data/origin/wider_face_split/wider_face_train_bbx_gt.txt"
#经过规范化后的索引与标签文件
cfg.ORIGINAL_FORMATTER_TXT_PATH = "../data/pdata/train_original_formated.txt"
#原始数据集中图片地址
cfg.ORIGINAL_IMG_PATH          = "../data/origin/WIDER_train/images/"

#原始特征点数据集中的索引与标签文件
cfg.ORIGINAL_LANDMARK_TXT_PATH = "../data/origin/trainImageList.txt"
#原始特征点数据集中的图片地址
cfg.ORIGINAL_LANDMARK_IMG_PATH = "../data/origin/"

#以下四项分别为neg,pos,part与landmark类型图片文件的索引与标签文件
cfg.TRAIN_NEG_TXT_PATH       = "../data/%s/train_neg.txt"
cfg.TRAIN_POS_TXT_PATH       = "../data/%s/train_pos.txt"
cfg.TRAIN_PART_TXT_PATH      = "../data/%s/train_part.txt"
cfg.TRAIN_LANDMARK_TXT_PATH  = "../data/%s/train_landmark.txt"
cfg.TRAIN_IMG_PATH           = "../data/%s/images/"
cfg.TRAIN_TFRECORDS          = "../data/%s/train.tfrecords"
cfg.TRAIN_TYPEWISE_TFRECORDS = "../data/%s/train_%s.tfrecords"
cfg.MODEL_PATH               = "../data/%s/models/"

#不同网络对应数据文件夹名称
cfg.PNET_DIR      = "pdata"
cfg.RNET_DIR      = "rdata"
cfg.ONET_DIR      = "odata"
