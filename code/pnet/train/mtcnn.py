import tensorflow as tf
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from net.network import cnnbox


def PNet(input):
    cb = cnnbox(stdev=0.1)

    pnet = cb.conv2d(input,"conv1",3,10,activation_fn=cb.prelu)
    print(pnet.get_shape)
    pnet = cb.max_pool2d(pnet,"pool1")
    print(pnet.get_shape)
    pnet = cb.conv2d(pnet,"conv2",10,16,activation_fn=cb.prelu)
    print(pnet.get_shape)
    pnet = cb.conv2d(pnet,"conv3",16,32,activation_fn=cb.prelu)
    print(pnet.get_shape)

    fcls_pred     = cb.conv2d(pnet,"conv4_1",32,2,filter_size=[1,1],activation_fn=None)
    print(fcls_pred.get_shape())
    bbr_pred      = cb.conv2d(pnet,"conv4_2",32,4,filter_size=[1,1],activation_fn=None)
    print(bbr_pred.get_shape())
    landmark_pred = cb.conv2d(pnet,"conv4_3",32,10,filter_size=[1,1],activation_fn=None)
    print(landmark_pred.get_shape())

    return fcls_pred,bbr_pred,landmark_pred

