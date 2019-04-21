import tensorflow as tf
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from net.network import CnnBox


def PNet(input):
    cb = CnnBox()

    pnet = cb.conv2d(input,"conv1",10,activation_fn=cb.prelu)
    pnet = cb.max_pool2d(pnet,"pool1")
    pnet = cb.conv2d(pnet,"conv2",16,activation_fn=cb.prelu)
    pnet = cb.conv2d(pnet,"conv3",32,activation_fn=cb.prelu)

    fcls_pred     = cb.conv2d(pnet,"conv4_1",2,filter_size=[1,1],activation_fn=None)
    bbr_pred      = cb.conv2d(pnet,"conv4_2",4,filter_size=[1,1],activation_fn=None)
    landmark_pred = cb.conv2d(pnet,"conv4_3",10,filter_size=[1,1],activation_fn=None)

    return fcls_pred,bbr_pred,landmark_pred

def RNet(input):
    cb = CnnBox()

    rnet = cb.conv2d(input,"conv1",28,activation_fn=cb.prelu)
    rnet = cb.max_pool2d(rnet,"pool1")
    rnet = cb.conv2d(rnet,"conv2",48,activation_fn=cb.prelu)
    rnet = cb.max_pool2d(rnet,"pool2")
    rnet = cb.conv2d(rnet,"conv3",64,filter_size=[2,2],activation_fn=cb.prelu)
    rnet = cb.fc(rnet,"fc1",128,activation_fn=cb.prelu)

    fcls_pred     = cb.fc(rnet,"fc2_1",2,activation_fn=None)
    bbr_pred      = cb.fc(rnet,"fc2_2",4,activation_fn=None)
    landmark_pred = cb.fc(rnet,"fc2_3",10,activation_fn=None)

    return fcls_pred,bbr_pred,landmark_pred

def ONet(input):
    cb = CnnBox()

    onet = cb.conv2d(input,"conv1",32,activation_fn=cb.prelu)
    onet = cb.max_pool2d(onet,"pool1",filter_size=[1,3,3,1])
    onet = cb.conv2d(onet,"conv2",64,activation_fn=cv.prelu)
    onet = cb.max_pool2d(onet,"pool2",filter_size=[1,3,3,1])
    onet = cb.conv2d(onet,"conv3",64,activation_fn=cb.prelu)
    onet = cb.max_pool2d(onet,"pool3")
    onet = cb.conv2d(onet,"conv4",128,filter_size=[2,2],activation_fn=cb.prelu)
    onet = cb.fc(onet,"fc1",256,activation_fn=cb.prelu)

    fcls_pred = cb.fc(onet,"fc2_1",2,activation_fn=None)
    bbr_pred = cb.fc(onet,"fc2_2",4,activation_fn=None)
    landmark_pred = cb.fc(onet,"fc2_3",10,activation_fn=None)

    return fcls_pred,bbr_pred,landmark_pred

