import numpy as np
import tensorflow as tf

def cls_loss(pred,label):
    """
    计算人脸分类误差

    :pred: TODO
    :label: TODO
    :returns: TODO

    """
    pred = tf.squeeze(pred) 

def read_batch_data_from_tfrecord():
    """
    从tfrecord中读取数据
    :returns: TODO

    """
    data_path = cfg.PNET_TRAIN_TFRECORDS
    feature = {'sample/image': tf.FixedLenFeature([],tf.string),
            'sample/label': tf.FixedLenFeature([],tf.int64),
            'sample/bbx': tf.FixedLenFeature([4],tf.float32),
            'sample/landmark': tf.FixedLenFeature([10],tf.float32)}

    filename_queue = tf.train.string_input_producer([data_path],num_epochs=1)

    tfrecord_reader = tf.TFRecordReader()
    _,serialized_example = tfrecord_reader.read(filename_queue)


    features = tf.parse_single_example(serialized_example,features=feature)

    image    = tf.decode_raw(features['sample/image'],tf.float32)
    label    = tf.cast(features['sample/label'],tf.int32)
    bbx      = tf.cast(features['sample/bbx'],tf.float32)
    landmark = tf.cast(features['sample/landmark'],tf.float32)

    image = tf.reshape(image,[12,12,3])

    batch_images,batch_labels,batch_bbxs,batch_landmarks = tf.train.shuffle_batch([image,label,bbx,landmark],batch_size=50,capacity=30,num_threads=4,min_after_dequeue=10)

    return batch_images,batch_labels,batch_bbxs,batch_landmarks


def train_net_wise(current_net,ratio):
    """
    训练网络
    :current_net: TODO
    :ratio: TODO
    :returns: TODO
    """
    IMG = tf.placeholder(tf.float32,[None,size,size,3],name="IMG")
    CLS = tf.placeholder(tf.float32,[None],name="CLS")
    BBX = tf.placeholder(tf.float32,[None,size,size,4],name="BBX")
    LANDMARK = tf.placeholder(tf.float32,[None,size,size,10],name="LANDMARK")

    fcls_pred,bbr_pred,landmark_pred = current_net(IMG)

    fcls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fcls_pred,labels=CLS))
    bbx_loss = tf.reduce_mean(tf.square(bbr_pred-BBX))
    landmark_loss = tf.reduce_mean(tf.square(landmark_pred-LANDMARK))

    loss = ratio[0]*fcls_loss+ratio[1]*bbx_loss+ratio[2]*landmark_loss
    optimizer = tf.train.MomentumOptimizer().minimize(loss)

    return loss,optimizer


def train():
    """
    训练网络
    """
    batch_images,batch_labels,batch_bbxs,batch_landmarks = read_batch_data_from_tfrecord()
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # 用10000批数据训练,每一批50张图片
        for batch_idx in range(10000):
            bimg,blabel,bbbx,blandmark = sess.run([batch_images,batch_labels,batch_bbxs,batch_landmarks])

            # for i in range(blabel.size):
                # print("%d %r %r"%(blabel[i],bbbx[i],blandmark[i]))

        coord.request_stop()
        coord.join(threads)
        sess.close()
