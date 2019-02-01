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
   return tf.train.Featur(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """
    tfrecord 浮点型数据
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

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
    with open(cfg.PNET_TRAIN_PART_TXT_PATH,"r") as f:
        data = f.readlines()

    with open(cfg.PNET_TRAIN_PART_TXT_PATH,"r") as f:
        data.append(f.readlines())

    with open(cfg.PNET_TRAIN_POS_TXT_PATH,"r") as f:
        data.append(f.readlines())

    with open(cfg.PNET_TRAIN_LANDMARK_TXT_PATH,"r") as f:
        data.append(f.readlines())

    #打乱list顺序
    random.shuffle(data)

    #打开一个TFRecords，用于写入
    tfrecords_writer = tf.python_io.TFRecordWriter(cfg.PNET_TRAIN_TFRECORDS)

    for line in data:
        mls = line.strip().split()

        img = cv2.imread(cfg.PNET_TRAIN_IMG_PATH+mls[0])
        img = img.astype(np.float32)
        label = mls[1]
        bbx = np.zeros(4,dtype=np.float)
        landmark = np.zeros(10,dtype=np.float)

        if len(mls) == 6:
            bbx = mls[2:]
        elif len(mls) == 12:
            landmark = mls[2:]

        feature = {'sample/image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                'sample/label': _int64_feature(label),
                'sample/bbx': _float_feature(bbx),
                'sample/landmark': _float_feature(landmark)}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        tfrecords_writer.write(example.SerializeToString())

    writer.close()

def read_train_tfrecord():
    """
    测试读取用于训练的tfrecord
    """
    data_path = cfg.PNET_TRAIN_TFRECORDS
    with tf.Session() as sess:
        feature = {'sample/image': tf.FixedLenFeature([],tf.string),
                'sample/label': tf.FixedLenFeature([],tf.int64),
                'sample/bbx': tf.FixedLenFeature([],tf.float32),
                'sample/landmark': tf.FixedLenFeature([],tf.float32)}

        filename_queue = tf.train.string_input_producer([data_path])

        tfrecord_reader = tf.TFRecordReader()
        _,serialized_example = tfrecord_reader.read(filename_queue)


        features = tf.parse_single_example(serialized_example,features=feature)

        image    = tf.decode_raw(features['sample/image'],tf.float32)
        label    = tf.cast(features['sample/label'],tf.int32)
        bbx      = tf.cast(features['sample/bbx'],tf.float32)
        landmark = tf.cast(features['sample/landmark'],tf.float32)

        image = tf.reshape(image,[12,12,3])

        batch_images,batch_labels,batch_bbxs,batch_landmarks = tf.train.shuffle_batch([image,label,bbx,landmark],batch_size=10,capacity=30,num_threads=4,min_after_dequeue=10)

        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.start_queue_runners(coord=coord)

        for batch_idx in range(100):
            bimg,blabel,bbbx,blandmark = sess.run([batch_images,batch_labels,batch_bbxs,batch_landmarks])

            for i in range(len(blabel)):
                print("%d %s %s"%(blabel[i]," ".join(bbbx[i])," ".join(blandmark[i])))

       coord.request_stop()
       coord.join(threads)
       sess.close()


if __name__ == "__main__":
    produce_train_tfrecord()