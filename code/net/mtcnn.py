import tensorflow as tf
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from net.network import cnnbox


def PNet(input):
    cb = cnnbox(stdev=0.1)

    pnet = cb.conv2d(input,"conv1",3,10,activation_fn=cb.prelu)
    # print(pnet.get_shape)
    pnet = cb.max_pool2d(pnet,"pool1")
    # print(pnet.get_shape)
    pnet = cb.conv2d(pnet,"conv2",10,16,activation_fn=cb.prelu)
    # print(pnet.get_shape)
    pnet = cb.conv2d(pnet,"conv3",16,32,activation_fn=cb.prelu)
    # print(pnet.get_shape)

    fcls_pred     = cb.conv2d(pnet,"conv4_1",32,2,filter_size=[1,1],activation_fn=None)
    # print(fcls_pred.get_shape())
    bbr_pred      = cb.conv2d(pnet,"conv4_2",32,4,filter_size=[1,1],activation_fn=None)
    # print(bbr_pred.get_shape())
    landmark_pred = cb.conv2d(pnet,"conv4_3",32,10,filter_size=[1,1],activation_fn=None)
    # print(landmark_pred.get_shape())

    return fcls_pred,bbr_pred,landmark_pred

def RNet(input):
    cb = cnnbox()

    rnet = cb.conv2d(input,"conv1",3,28,activation_fn=cb.prelu)
    rnet = cb.max_pool2d(rnet,"pool1",filter_size=[1,3,3,1])
    rnet = cb.conv2d(rnet,"conv2",28,48,activation_fn=cb.prelu)
    rnet = cb.max_pool2d(rnet,"pool2",filter_size=[1,3,3,1],activation_fn=cb.prelu)
    rnet = cb.conv2d(rnet,"conv3",48,64,filter_size=[2,2],activation_fn=cb.prelu)
    rnet = cb.fc(rnet,"fc1",64,128,activation_fn=cb.prelu)

    fcls_pred     = cb.fc(rnet,"fc2_1",128,2,activation_fn=None)
    bbr_pred      = cb.fc(rnet,"fc2_2",128,4,activation_fn=None)
    landmark_pred = cb.fc(rnet,"fc2_3",128,10,activation_fn=None)

    return fcls_pred,bbr_pred,landmark_pred

def ONet(input):
    cb = cnnbox()

    onet = cb.conv2d(input,"conv1",3,32)
    onet = cb.max_pool2d(onet,"pool1",filter_size=[1,3,3,1])
    onet = cb.conv2d(onet,"conv2",32,64)
    onet = cb.max_pool2d(onet,"pool2",filter_size=[1,3,3,1])
    onet = cb.conv2d(onet,"conv3",64,64)
    onet = cb.max_pool2d(onet,"pool3")
    onet = cb.conv2d(onet,"conv4",64,128,filter_size=[2,2])
    onet = cb.fc(onet,"fc1",128,256)

    fcls_pred = cb.fc(onet,"fc2_1",256,2,activation_fn=tf.nn.softmax)
    bbr_pred = cb.fc(onet,"fc2_2",256,4)
    landmark_pred = cb.fc(onet,"fc2_3",256,10)

    return fcls_pred,bbr_pred,landmark_pred

