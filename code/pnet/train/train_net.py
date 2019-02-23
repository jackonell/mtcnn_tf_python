import numpy as np
import tensorflow as tf
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg
from mtcnn import PNet

def cls_loss(pred,label):
    """
    计算人脸分类误差
    -1 1
    """
    pred = tf.squeeze(pred)[:,0]

    # print("------------------->")
    # print(label.get_shape())
    # print(pred.get_shape())
    # print(label.eval())
    # print(pred.eval())

    keeps = tf.where(tf.logical_or(tf.equal(label,1),tf.equal(label,-1)))
    filter_pred = tf.gather(pred,keeps)
    filter_label = tf.gather(label,keeps)

    print(filter_pred.eval())
    print(filter_label.eval())

    clsloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=filter_pred,labels=filter_label))

    return clsloss

def bbx_loss(pred_bbx,bbx,label):
    """
    计算人脸框判定误差
    0 1
    """
    pred_bbx = tf.squeeze(pred_bbx)

    keeps = tf.where(tf.logical_or(tf.equal(label,1),tf.equal(label,0)))
    filter_pred_bbx = tf.gather(pred_bbx,keeps)
    filter_bbx = tf.gather(bbx,keeps)

    bbx_loss = tf.reduce_sum(tf.square(filter_pred_bbx-filter_bbx))

    return bbx_loss

def landmark_loss(pred_landmark,landmark,label):
    """
    计算特征点预测误差
    """
    pred_landmark = tf.squeeze(pred_landmark)

    keeps = tf.where(tf.equal(label,2))
    filter_pred_landmark = tf.gather(pred_landmark,keeps)
    filter_landmark = tf.gather(landmark,keeps)

    landmark_loss = tf.reduce_sum(tf.square(filter_pred_landmark-filter_landmark))

    return landmark_loss

def read_batch_data_from_tfrecord():
    """
    从tfrecord中读取数据
    """
    data_path = cfg.PNET_TRAIN_TFRECORDS
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

    image = tf.reshape(image,[12,12,3])
    image = (tf.cast(image,tf.float32)-127.5)/128

    batch_images,batch_labels,batch_bbxs,batch_landmarks = tf.train.batch([image,label,bbx,landmark],batch_size=64,capacity=64,num_threads=4)

    return batch_images,batch_labels,batch_bbxs,batch_landmarks


def train():
    """
    训练网络
    """
    batch_images,batch_labels,batch_bbxs,batch_landmarks = read_batch_data_from_tfrecord()

    batch_size = 64
    size =12

    IMG = tf.placeholder(tf.float32,[batch_size,size,size,3],name="IMG")
    CLS = tf.placeholder(tf.float32,[batch_size],name="CLS")
    BBX = tf.placeholder(tf.float32,[batch_size,4],name="BBX")
    LANDMARK = tf.placeholder(tf.float32,[batch_size,10],name="LANDMARK")

    cls_pred,bbx_pred,land_pred = PNet(IMG)

    # --------------->cls loss
    fcls_pred = tf.squeeze(cls_pred)

    keeps = tf.where(tf.logical_or(tf.equal(CLS,1),tf.equal(CLS,0)))
    filter_pred = tf.squeeze(tf.gather(fcls_pred,keeps))
    # filter_label = tf.squeeze(tf.gather(CLS,keeps))
    filter_label = tf.gather(CLS,keeps)
    filter_label_op = tf.where(tf.equal(filter_label,1),1-filter_label,1-filter_label) #要修改

    filter_label = tf.concat([filter_label,filter_label_op],axis=1)

    precloss = tf.nn.softmax_cross_entropy_with_logits(logits=filter_pred,labels=filter_label)
    _cls_loss = tf.reduce_mean(precloss)

    # --------------->bbx loss
    bbr_pred = tf.squeeze(bbx_pred)

    keeps = tf.where(tf.logical_or(tf.equal(CLS,1),tf.equal(CLS,-1)))
    filter_pred_bbx = tf.gather(bbr_pred,keeps)
    filter_bbx = tf.gather(BBX,keeps)

    prebloss = tf.square(filter_pred_bbx-filter_bbx)
    print("bloss: %r"%prebloss.get_shape())
    prebloss = tf.reduce_sum(tf.squeeze(prebloss),axis=1)
    _bbx_loss = tf.reduce_mean(prebloss)

    # ---------------->landmark loss
    landmark_pred = tf.squeeze(land_pred)

    keeps = tf.where(tf.equal(CLS,-2))
    filter_pred_landmark = tf.gather(landmark_pred,keeps)
    filter_landmark = tf.gather(LANDMARK,keeps)

    prelloss = tf.square(filter_pred_landmark-filter_landmark)

    prelloss = tf.reduce_sum(tf.squeeze(prelloss),axis=1)
    _landmark_loss = tf.reduce_mean(prelloss)

    ratio = [1,0.5,0.5]
    loss = ratio[0]*_cls_loss+ratio[1]*_bbx_loss+ratio[2]*_landmark_loss

    global_step = tf.Variable(0,trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,3000,0.96,staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss)

    # saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # 用10000批数据训练,每一批50张图片
        for batch_idx in range(1):
            bimg,blabel,bbbx,blandmark = sess.run([batch_images,batch_labels,batch_bbxs,batch_landmarks])

            # fp,fl = sess.run([filter_pred,filter_label],feed_dict={IMG:bimg,CLS:blabel,BBX:bbbx,LANDMARK:blandmark})
            # print(fp)
            # print(fl)

            prec,preb,prel = sess.run([filter_pred,filter_label,precloss],feed_dict={IMG:bimg,CLS:blabel,BBX:bbbx,LANDMARK:blandmark})
            print(prec)
            print(preb)
            print(prel)

            # prec,preb,prel = sess.run([precloss,prebloss,prelloss],feed_dict={IMG:bimg,CLS:blabel,BBX:bbbx,LANDMARK:blandmark})
            # print(prec)
            # print(preb)
            # print(prel)

            # for i in range(len(blabel)):
                # print("%d, %r, %r"%(blabel[i],bbbx[i],blandmark[i]))

            _,vloss,vcloss,vbloss,vlloss = sess.run([optimizer,loss, _cls_loss,_bbx_loss,_landmark_loss],feed_dict={"IMG:0":bimg,"CLS:0":blabel,"BBX:0":bbbx,"LANDMARK:0":blandmark})

            if batch_idx % 25 == 0:
                print("训练批次：%d,分类loss:%f,BBX loss:%f,landmark loss:%f,total loss：%f"%(batch_idx,vcloss,vbloss,vlloss,vloss))

            # _,vloss,vcpred,vbpred,vlpred = sess.run([optimizer,loss, cls_pred,bbx_pred,land_pred],feed_dict={IMG:bimg,CLS:blabel,BBX:bbbx,LANDMARK:blandmark})

            # print(vcpred[:10])
            # print(vbpred[:10])
            # print(vlpred[:10])
            # print(bimg)
            # if batch_idx % 25 == 0:
            # print("训练批次：%d,分类loss:%f,BBX loss:%f,landmark loss:%f,total loss：%f"%(batch_idx,vcloss,vbloss,vlloss,vloss))

        # saver.save(sess,cfg.PNET_MODEL_PATH,global_step=step)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    train()
