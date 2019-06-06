import tensorflow as tf
import numpy as np
import cv2
import random
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from net.cfgs import cfg

def _int64_feature(value):
    """
    tfrecord 整型数据
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """
    tfrecord 浮点型数据
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """
    tfrecord 二进制类型
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def produce_train_tfrecord(data,data_dir,tfrecord_path):
    """
    产生用于训练的tfrecord
    """
    #打乱list顺序
    random.shuffle(data)

    print(len(data))
    #打开一个TFRecords，用于写入
    tfrecords_writer = tf.python_io.TFRecordWriter(tfrecord_path)
    flag = 0
    num = 1

    for line in data:
        # if flag > 0:
            # break
        # flag = flag+1
        mls = line.strip().split()

        img = cv2.imread(cfg.TRAIN_IMG_PATH%data_dir+mls[0])
        #更符合情况
        img = img.astype(np.uint8)
        label = int(mls[1])
        bbx = list(np.zeros(4,dtype=np.float))
        landmark = list(np.zeros(10,dtype=np.float))

        if len(mls) == 6:
            bbx = list(map(float,mls[2:]))
        elif len(mls) == 12:
            landmark = list(map(float,mls[2:]))

        feature = {'sample/image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                'sample/label': _int64_feature(label),
                'sample/bbx': _float_feature(bbx),
                'sample/landmark': _float_feature(landmark)}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        tfrecords_writer.write(example.SerializeToString())

        if num % 100 == 0:
            print("一共需处理图片：%s, 已经处理：%s"%(len(data),num))
        num = num+1

    tfrecords_writer.close()

def produce_train_tfrecord_in_one_file(data_dir):
    """
    产生用于训练的tfrecord
    """
    data = []
    base_num = 250000
    with open(cfg.TRAIN_NEG_TXT_PATH%data_dir,"r") as f:
        data = f.readlines()[:3*base_num]

    with open(cfg.TRAIN_PART_TXT_PATH%data_dir,"r") as f:
        data.extend(f.readlines()[:base_num])

    with open(cfg.TRAIN_POS_TXT_PATH%data_dir,"r") as f:
        data.extend(f.readlines()[:base_num])

    with open(cfg.TRAIN_LANDMARK_TXT_PATH%data_dir,"r") as f:
        data.extend(f.readlines()[:2*base_num])

    produce_train_tfrecord(data,data_dir,cfg.TRAIN_IMG_PATH%data_dir)


def produce_train_tfrecord_type_specific(data_dir,img_type,base_num=None):
    """
    为指定类型图片产生用于训练的tfrecord
    """
    data = []

    img_txt_path = ""
    if img_type == "neg":
        img_txt_path = cfg.TRAIN_NEG_TXT_PATH%data_dir
    elif img_type == "pos":
        img_txt_path = cfg.TRAIN_POS_TXT_PATH%data_dir
    elif img_type == "par":
        img_txt_path = cfg.TRAIN_PART_TXT_PATH%data_dir
    elif img_type == "landmark":
        img_txt_path = cfg.TRAIN_LANDMARK_TXT_PATH%data_dir

    with open(img_txt_path,"r") as f:
        if base_num is not None:
            data = f.readlines()[:base_num]
        else:
            data = f.readlines()

    produce_train_tfrecord(data,data_dir,cfg.TRAIN_TYPEWISE_TFRECORDS%(data_dir,img_type))

def produce_train_tfrecord_in_multi_file(data_dir):
    """
    为每一类图片产生不同的tfrecord
    """
    # produce_train_tfrecord_type_specific(data_dir,"neg")
    # produce_train_tfrecord_type_specific(data_dir,"pos")
    # produce_train_tfrecord_type_specific(data_dir,"par")
    produce_train_tfrecord_type_specific(data_dir,"landmark")

def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)

    return inputs

def read_batch_data_from_tfrecord(data_path,size,batch_size_s):
    """
    从tfrecord中读取数据
    """
    batch_size_s = int(batch_size_s)
    feature = {'sample/image': tf.FixedLenFeature([],tf.string),
            'sample/label': tf.FixedLenFeature([],tf.int64),
            'sample/bbx': tf.FixedLenFeature([4],tf.float32),
            'sample/landmark': tf.FixedLenFeature([10],tf.float32)}

    filename_queue = tf.train.string_input_producer([data_path],shuffle=True)

    tfrecord_reader = tf.TFRecordReader()
    _,serialized_example = tfrecord_reader.read(filename_queue)


    features = tf.parse_single_example(serialized_example,features=feature)

    image    = tf.decode_raw(features['sample/image'],tf.uint8)
    label    = tf.cast(features['sample/label'],tf.int32)
    bbx      = tf.cast(features['sample/bbx'],tf.float32)
    landmark = tf.cast(features['sample/landmark'],tf.float32)

    image = tf.reshape(image,[size,size,3])
    image = image_color_distort(image)
    image = (tf.cast(image,tf.float32)-127.5)/128

    batch_images,batch_labels,batch_bbxs,batch_landmarks = tf.train.batch([image,label,bbx,landmark],batch_size=batch_size_s,capacity=batch_size_s,num_threads=4)

    return batch_images,batch_labels,batch_bbxs,batch_landmarks

def read_batch_data_from_multi_tfrecord(data_dir,size):
    batch_images_pos,batch_labels_pos,batch_bbxs_pos,batch_landmarks_pos = read_batch_data_from_tfrecord(cfg.TRAIN_TYPEWISE_TFRECORDS%(data_dir,"pos"),size,cfg.BATCH_SIZE/7)
    batch_images_par,batch_labels_par,batch_bbxs_par,batch_landmarks_par = read_batch_data_from_tfrecord(cfg.TRAIN_TYPEWISE_TFRECORDS%(data_dir,"par"),size,cfg.BATCH_SIZE/7)
    batch_images_neg,batch_labels_neg,batch_bbxs_neg,batch_landmarks_neg = read_batch_data_from_tfrecord(cfg.TRAIN_TYPEWISE_TFRECORDS%(data_dir,"neg"),size,cfg.BATCH_SIZE/7*3)
    batch_images_landmark,batch_labels_landmark,batch_bbxs_landmark,batch_landmarks_landmark = read_batch_data_from_tfrecord(cfg.TRAIN_TYPEWISE_TFRECORDS%(data_dir,"landmark"),size,cfg.BATCH_SIZE/7*2)

    batch_images    = tf.concat([batch_images_neg,    batch_images_par,    batch_images_pos,    batch_images_landmark], axis=0)
    batch_labels    = tf.concat([batch_labels_neg,    batch_labels_par,    batch_labels_pos,    batch_labels_landmark], axis=0)
    batch_bbxs      = tf.concat([batch_bbxs_neg,      batch_bbxs_par,      batch_bbxs_pos,      batch_bbxs_landmark],   axis=0)
    batch_landmarks = tf.concat([batch_landmarks_neg, batch_landmarks_par, batch_landmarks_pos, batch_landmarks_landmark],   axis=0)

    return batch_images,batch_labels,batch_bbxs,batch_landmarks

def read_batch_data_from_single_tfrecord(data_dir,size):
    batch_images,batch_labels,batch_bbxs,batch_landmarks = read_batch_data_from_tfrecord(cfg.TRAIN_TFRECORDS%(data_dir),size,cfg.BATCH_SIZE)

    return batch_images,batch_labels,batch_bbxs,batch_landmarks

if __name__ == "__main__":
    data_dir = cfg.PNET_DIR
    produce_train_tfrecord_in_one_file(data_dir)

