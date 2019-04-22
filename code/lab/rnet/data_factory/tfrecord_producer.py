import tensorflow as tf
import numpy as np
import cv2
import random
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg

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

def produce_train_tfrecord():
    """
    产生用于训练的tfrecord
    """
    data = []
    base_num = 250000
    with open(cfg.RNET_TRAIN_NEG_TXT_PATH,"r") as f:
        data = f.readlines()[:3*base_num]

    with open(cfg.RNET_TRAIN_PART_TXT_PATH,"r") as f:
        data.extend(f.readlines()[:base_num])

    with open(cfg.RNET_TRAIN_POS_TXT_PATH,"r") as f:
        data.extend(f.readlines()[:base_num])

    with open(cfg.RNET_TRAIN_LANDMARK_TXT_PATH,"r") as f:
        data.extend(f.readlines()[:2*base_num])

    #打乱list顺序
    random.shuffle(data)
    # print(len(data))
    # print(" ".join(data[:10]))

    print(len(data))
    #打开一个TFRecords，用于写入
    tfrecords_writer = tf.python_io.TFRecordWriter(cfg.RNET_TRAIN_TFRECORDS)
    flag = 0
    num = 1

    for line in data:
        # if flag > 0:
            # break
        # flag = flag+1
        mls = line.strip().split()

        img = cv2.imread(cfg.RNET_TRAIN_IMG_PATH+mls[0])
        # print(cfg.RNET_TRAIN_IMG_PATH+mls[0])
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

def produce_train_tfrecord_type_specific(txt_path,tfrecord_path,base_num=None):
    """
    为指定类型图片产生用于训练的tfrecord
    """
    data = []
    with open(txt_path,"r") as f:
        if base_num is not None:
            data = f.readlines()[:base_num]
        else:
            data = f.readlines()

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

        img = cv2.imread(cfg.RNET_TRAIN_IMG_PATH+mls[0])
        # print(cfg.RNET_TRAIN_IMG_PATH+mls[0])
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

def read_train_tfrecord():
    """
    测试读取用于训练的tfrecord
    """
    data_path = cfg.RNET_TRAIN_TFRECORDS
    with tf.Session() as sess:
        feature = {'sample/image': tf.FixedLenFeature([],tf.string),
                'sample/label': tf.FixedLenFeature([],tf.int64),
                'sample/bbx': tf.FixedLenFeature([4],tf.float32),
                'sample/landmark': tf.FixedLenFeature([10],tf.float32)}

        filename_queue = tf.train.string_input_producer([data_path],num_epochs=1)

        tfrecord_reader = tf.TFRecordReader()
        _,serialized_example = tfrecord_reader.read(filename_queue)


        features = tf.parse_single_example(serialized_example,features=feature)

        image    = tf.decode_raw(features['sample/image'],tf.unit8)
        label    = tf.cast(features['sample/label'],tf.int32)
        bbx      = tf.cast(features['sample/bbx'],tf.float32)
        landmark = tf.cast(features['sample/landmark'],tf.float32)

        image = tf.reshape(image,[12,12,3])

        batch_images,batch_labels,batch_bbxs,batch_landmarks = tf.train.shuffle_batch([image,label,bbx,landmark],batch_size=50,capacity=30,num_threads=4,min_after_dequeue=10)

        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for batch_idx in range(1):
            bimg,blabel,bbbx,blandmark = sess.run([batch_images,batch_labels,batch_bbxs,batch_landmarks])

            keeps = tf.where(tf.logical_or(tf.equal(blabel,1),tf.equal(blabel,0)))
            filter_res = tf.gather(bbbx,keeps)
            dd = tf.gather(blabel,keeps)

            print("%r"%filter_res.eval())
            print("%r"%dd.eval())

            # for i in range(blabel.size):
                # print("%r %d %r %r"%(keeps[i],blabel[i],bbbx[i],blandmark[i]))

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    produce_train_tfrecord_type_specific(cfg.RNET_TRAIN_NEG_TXT_PATH,cfg.RNET_TRAIN_TFRECORDS%"neg")
    produce_train_tfrecord_type_specific(cfg.RNET_TRAIN_POS_TXT_PATH,cfg.RNET_TRAIN_TFRECORDS%"pos")
    produce_train_tfrecord_type_specific(cfg.RNET_TRAIN_PART_TXT_PATH,cfg.RNET_TRAIN_TFRECORDS%"par")
    produce_train_tfrecord_type_specific(cfg.RNET_TRAIN_LANDMARK_TXT_PATH,cfg.RNET_TRAIN_TFRECORDS%"landmark")
    # produce_train_tfrecord()
    # read_train_tfrecord()
