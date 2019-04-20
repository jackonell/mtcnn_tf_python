import numpy as np
import tensorflow as tf
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg
from mtcnn import RNet

def cls_loss_and_acc(pred,label):
    """
    计算人脸分类误差
    -1 1
    """
    pred = tf.squeeze(pred)

    # keeps = tf.where(tf.logical_or(tf.equal(label,1),tf.equal(label,-1)))
    keeps = tf.where(tf.logical_or(tf.equal(label,1),tf.equal(label,0)))
    filter_pred = tf.squeeze(tf.gather(pred,keeps))
    filter_label = tf.gather(label,keeps)

    # filter_label_ori = tf.where(tf.equal(filter_label,1),filter_label,1+filter_label) #要修改
    filter_label_ori = filter_label
    filter_label_op = tf.where(tf.equal(filter_label_ori,1),1-filter_label_ori,1-filter_label_ori) #要修改

    filter_label_n = tf.concat([filter_label_ori,filter_label_op],axis=1)

    filter_pred = tf.nn.softmax(filter_pred)
    clsloss = -(filter_label_n*tf.log(filter_pred+1e-10))
    clsloss = tf.reduce_mean(clsloss)

    max_idx_l = tf.argmax(filter_pred,1)
    max_idx_p = tf.argmax(filter_label_n,1)

    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return clsloss,accuracy,filter_pred,filter_label_n

def cls_accuracy(pre,label):
    """
    计算人脸分类准确率
    """
    pred = tf.squeeze(pred)

    keeps = tf.where(tf.logical_or(tf.equal(label,1),tf.equal(label,-1)))
    filter_pred = tf.squeeze(tf.gather(pred,keeps))
    filter_label = tf.gather(label,keeps)

    filter_label_ori = tf.where(tf.equal(filter_label,1),filter_label,1+filter_label) #要修改
    filter_label_op = tf.where(tf.equal(filter_label,1),1-filter_label,1-filter_label) #要修改

    filter_label = tf.concat([filter_label,filter_label_op],axis=1)

    max_idx_l = tf.argmax(filter_pred,1)
    max_idx_p = tf.argmax(filter_label,1)

    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy

def bbx_loss(pred_bbx,bbx,label):
    """
    计算人脸框判定误差
    0 1
    """
    pred_bbx = tf.squeeze(pred_bbx)

    # keeps = tf.where(tf.logical_or(tf.equal(label,1),tf.equal(label,0)))
    keeps = tf.where(tf.logical_or(tf.equal(label,1),tf.equal(label,-1)))
    filter_pred_bbx = tf.gather(pred_bbx,keeps)
    filter_bbx = tf.gather(bbx,keeps)

    prebloss = tf.square(filter_pred_bbx-filter_bbx)
    prebloss = tf.reduce_sum(tf.squeeze(prebloss),axis=1)
    bbx_loss = tf.reduce_mean(prebloss)

    return bbx_loss

def landmark_loss(pred_landmark,landmark,label):
    """
    计算特征点预测误差
    """
    pred_landmark = tf.squeeze(pred_landmark)

    # keeps = tf.where(tf.equal(label,2))
    keeps = tf.where(tf.equal(label,-2))
    filter_pred_landmark = tf.gather(pred_landmark,keeps)
    filter_landmark = tf.gather(landmark,keeps)

    prelloss = tf.square(filter_pred_landmark-filter_landmark)
    prelloss = tf.reduce_sum(tf.squeeze(prelloss),axis=1)
    landmark_loss = tf.reduce_mean(prelloss)

    return landmark_loss

def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)

    return inputs

def read_batch_data_from_tfrecord(data_path,batch_size_s):
    """
    从tfrecord中读取数据
    """
    batch_size_s = int(batch_size_s) 
    feature = {'image/encoded': tf.FixedLenFeature([],tf.string),
            'image/label': tf.FixedLenFeature([],tf.int64),
            'image/roi': tf.FixedLenFeature([4],tf.float32),
            'image/landmark': tf.FixedLenFeature([10],tf.float32)}

    filename_queue = tf.train.string_input_producer([data_path],shuffle=True)

    tfrecord_reader = tf.TFRecordReader()
    _,serialized_example = tfrecord_reader.read(filename_queue)


    features = tf.parse_single_example(serialized_example,features=feature)

    image    = tf.decode_raw(features['image/encoded'],tf.uint8)
    label    = tf.cast(features['image/label'],tf.int32)
    bbx      = tf.cast(features['image/roi'],tf.float32)
    landmark = tf.cast(features['image/landmark'],tf.float32)

    image = tf.reshape(image,[24,24,3])
    image = image_color_distort(image)
    image = (tf.cast(image,tf.float32)-127.5)/128

    batch_images,batch_labels,batch_bbxs,batch_landmarks = tf.train.batch([image,label,bbx,landmark],batch_size=batch_size_s,capacity=batch_size_s,num_threads=4)

    return batch_images,batch_labels,batch_bbxs,batch_landmarks

def read_batch_data_from_multi_tfrecord():
    batch_images_pos,batch_labels_pos,batch_bbxs_pos,batch_landmarks_pos = read_batch_data_from_tfrecord(cfg.RNET_TRAIN_TFRECORDS%"pos",cfg.BATCH_SIZE/7)
    batch_images_par,batch_labels_par,batch_bbxs_par,batch_landmarks_par = read_batch_data_from_tfrecord(cfg.RNET_TRAIN_TFRECORDS%"par",cfg.BATCH_SIZE/7)
    batch_images_neg,batch_labels_neg,batch_bbxs_neg,batch_landmarks_neg = read_batch_data_from_tfrecord(cfg.RNET_TRAIN_TFRECORDS%"neg",cfg.BATCH_SIZE/7*3)
    batch_images_landmark,batch_labels_landmark,batch_bbxs_landmark,batch_landmarks_landmark = read_batch_data_from_tfrecord(cfg.RNET_TRAIN_TFRECORDS%"landmark",cfg.BATCH_SIZE/7*2)

    batch_images    = tf.concat([batch_images_neg,    batch_images_par,    batch_images_pos,    batch_images_landmark], axis=0)
    batch_labels    = tf.concat([batch_labels_neg,    batch_labels_par,    batch_labels_pos,    batch_labels_landmark], axis=0)
    batch_bbxs      = tf.concat([batch_bbxs_neg,      batch_bbxs_par,      batch_bbxs_pos,      batch_bbxs_landmark],   axis=0)
    batch_landmarks = tf.concat([batch_landmarks_neg, batch_landmarks_par, batch_landmarks_pos, batch_landmarks_landmark],   axis=0)

    return batch_images,batch_labels,batch_bbxs,batch_landmarks

def train_net_wise(current_net,size,batch_size):
    """
    loss and optimizer
    """
    IMG = tf.placeholder(tf.float32,[None,size,size,3],name="IMG")
    CLS = tf.placeholder(tf.float32,[None],name="CLS")
    BBX = tf.placeholder(tf.float32,[None,4],name="BBX")
    LANDMARK = tf.placeholder(tf.float32,[None,10],name="LANDMARK")

    fcls_pred,bbr_pred,landmark_pred = current_net(IMG)

    _cls_loss,accuracy,filter_pred,filter_label = cls_loss_and_acc(fcls_pred,CLS)
    _bbx_loss = bbx_loss(bbr_pred,BBX,CLS)
    _landmark_loss = landmark_loss(landmark_pred,LANDMARK,CLS)

    return _cls_loss,_bbx_loss,_landmark_loss,accuracy,filter_pred,filter_label


def train():
    """
    训练网络
    """
    # batch_images,batch_labels,batch_bbxs,batch_landmarks = read_batch_data_from_tfrecord(cfg.RNET_TRAIN_TFRECORDS)
    batch_images,batch_labels,batch_bbxs,batch_landmarks = read_batch_data_from_multi_tfrecord()

    _cls_loss,_bbx_loss,_landmark_loss,accuracy,filter_pred,filter_label = train_net_wise(RNet,24,cfg.BATCH_SIZE)

    ratio = [1,0.5,0.5]
    loss = ratio[0]*_cls_loss+ratio[1]*_bbx_loss+ratio[2]*_landmark_loss

    global_step = tf.Variable(0,trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,40000,0.1,staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # 用10000批数据训练,每一批256张图片
        for batch_idx in range(120000):
            bimg,blabel,bbbx,blandmark = sess.run([batch_images,batch_labels,batch_bbxs,batch_landmarks])

            _,vloss,vcloss,vbloss,vlloss,lr,gstep,acc,fpred,flabel = sess.run([optimizer,loss, _cls_loss,_bbx_loss,_landmark_loss,learning_rate,global_step,accuracy,filter_pred,filter_label],feed_dict={"IMG:0":bimg,"CLS:0":blabel,"BBX:0":bbbx,"LANDMARK:0":blandmark})


            if gstep % 25 == 0:
                print("训练批次：%d,准确率:%f,分类loss:%f,BBX loss:%f,landmark loss:%f,total loss：%f,lr: %f"%(gstep,acc,vcloss,vbloss,vlloss,vloss,lr))

            if gstep % 300 == 0:
                print(fpred)
                print(flabel)

            if gstep % 40000 == 0:
                saver.save(sess,cfg.RNET_MODEL_PATH+"RNet",global_step=gstep)
                print("save RNet model at iteration: %d"%gstep)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    train()

