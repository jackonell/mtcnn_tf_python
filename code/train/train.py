import numpy as np
import tensorflow as tf
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from net.cfgs import cfg
from net.component_nets import PNet,RNet,ONet

def cls_loss_and_acc(pred,label):
    """
    计算人脸分类误差
    -1 1
    """
    pred = tf.squeeze(pred)

    keeps = tf.where(tf.logical_or(tf.equal(label,1),tf.equal(label,-1)))
    filter_pred = tf.squeeze(tf.gather(pred,keeps))
    filter_label = tf.gather(label,keeps)

    filter_label_ori = tf.where(tf.equal(filter_label,1),filter_label,1+filter_label) #要修改
    filter_label_op = 1-filter_label_ori #要修改

    filter_label_n = tf.concat([filter_label_ori,filter_label_op],axis=1)

    filter_pred = tf.nn.softmax(filter_pred)
    clsloss = -(filter_label_n*tf.log(filter_pred+1e-10))
    #online hard example mining
    # num = clsloss.get_shape()[0]
    # # num = tf.cast(num,dtype=tf.float32)
    # keep_ratio = 0.7
    # hard_num = tf.cast(num*keep_ratio,dtype=tf.int32)
    clsloss = tf.reduce_sum(clsloss,axis=1)
    num = tf.shape(clsloss)
    num = tf.cast(num,dtype=tf.float32)
    keep_ratio = tf.constant(0.7)

    hard_num = tf.cast(num*keep_ratio,dtype=tf.int32)
    hard_num = hard_num[0]

    clsloss,_ = tf.nn.top_k(clsloss,k=hard_num)
    clsloss = tf.reduce_mean(clsloss)

    #计算准确率
    max_idx_l = tf.argmax(filter_pred,1)
    max_idx_p = tf.argmax(filter_label_n,1)

    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return clsloss,accuracy


def bbx_loss(pred_bbx,bbx,label):
    """
    计算人脸框判定误差
    0 1
    """
    pred_bbx = tf.squeeze(pred_bbx)

    keeps = tf.where(tf.logical_or(tf.equal(label,1),tf.equal(label,0)))
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

    keeps = tf.where(tf.equal(label,2))
    filter_pred_landmark = tf.gather(pred_landmark,keeps)
    filter_landmark = tf.gather(landmark,keeps)

    prelloss = tf.square(filter_pred_landmark-filter_landmark)
    prelloss = tf.reduce_sum(tf.squeeze(prelloss),axis=1)
    landmark_loss = tf.reduce_mean(prelloss)

    return landmark_loss,filter_pred_landmark,filter_landmark

def train_net_wise(current_net,size):
    """
    loss and optimizer
    """
    IMG = tf.placeholder(tf.float32,[None,size,size,3],name="IMG")
    CLS = tf.placeholder(tf.float32,[None],name="CLS")
    BBX = tf.placeholder(tf.float32,[None,4],name="BBX")
    LANDMARK = tf.placeholder(tf.float32,[None,10],name="LANDMARK")

    fcls_pred,bbr_pred,landmark_pred = current_net(IMG)

    _cls_loss,accuracy = cls_loss_and_acc(fcls_pred,CLS)
    _bbx_loss = bbx_loss(bbr_pred,BBX,CLS)
    _landmark_loss,fpl,fl = landmark_loss(landmark_pred,LANDMARK,CLS)

    return _cls_loss,_bbx_loss,_landmark_loss,accuracy,fpl,fl

def train(read_tfrecord,ratio,model_name,data_dir,size,iterations):
    """
    训练网络

    @read_tfrecord: 读取tfrecord的函数
    @ratio: 得分、bbx regression、landmark的loss比例
    @model_name: model名称,如PNet
    @data_dir: 保存模型的文件夹，如pdata
    @size: 图片大小
    @iterations: 训练的迭代次数
    """
    if "PNet" == model_name:
        xnet = PNet
    elif "RNet" == model_name:
        xnet = RNet
    elif "ONet" == model_name:
        xnet = ONet

    batch_images,batch_labels,batch_bbxs,batch_landmarks = read_tfrecord(data_dir,size)

    _cls_loss,_bbx_loss,_landmark_loss,accuracy,fpl,fl = train_net_wise(xnet,size)

    loss = ratio[0]*_cls_loss+ratio[1]*_bbx_loss+ratio[2]*_landmark_loss

    key_iterations = int(iterations/4)

    global_step = tf.Variable(0,trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,key_iterations,0.1,staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for batch_idx in range(iterations):
            bimg,blabel,bbbx,blandmark = sess.run([batch_images,batch_labels,batch_bbxs,batch_landmarks])

            _,vloss,vcloss,vbloss,vlloss,lr,gstep,acc,ffpl,ffl = sess.run([optimizer,loss, _cls_loss,_bbx_loss,_landmark_loss,learning_rate,global_step,accuracy,fpl,fl],feed_dict={"IMG:0":bimg,"CLS:0":blabel,"BBX:0":bbbx,"LANDMARK:0":blandmark})
            
            # if vlloss < 0.25:
                # print(blandmark)
                # print(ffpl)
                # print(ffl)


            if gstep % 25 == 0:
                print("训练批次：%d,准确率:%f,分类loss:%f,BBX loss:%f,landmark loss:%f,total loss：%f,lr: %f"%(gstep,acc,vcloss,vbloss,vlloss,vloss,lr))

            if gstep % key_iterations == 0:
                saver.save(sess,"%s%s"%(cfg.MODEL_PATH%data_dir,model_name),global_step=gstep)
                print("save %s model at iteration: %d"%(model_name,gstep))

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    train()

