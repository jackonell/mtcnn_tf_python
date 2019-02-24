import tensorflow as tf
import sys, os
#注意到相当于将当前脚本移到code目录下执行
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mtcnn_cfg import cfg
from mtcnn_utils import IOU

class detector(object):

    """用于产生rnet与onet所需的训练数据"""

    def __init__(self):
        sess = tf.Session()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(cfg.PNET_MODEL_PATH)
        if ckpt and ckpt.model_checkpint_path:
            saver.restore(sess,ckpt.model_checkpint_path)

        width = tf.placeholder(tf.float32,name="width")
        height = tf.placeholder(tf.float32,name="height")
        img = tf.placeholder(tf.float16,shape=[None,width,height,3],name="IMG")

        self.fcls_pred,self.bbr_pred,landmark_pred = PNet(img)

    def predict(self, img):
        """
        预测结果
        """
        cls,bbr,landmark = sess.run([self.fcls_pred,self.bbr_pred,self.landmark_pred],feed_dict={"IMG:0":img})
        return cls,bbr,landmark

def calc_real_coordinate(img,ratio,bbox):
    """
    根据当前图像和缩放比例，计算出bbox中的数值对应的真实坐标值
    """
    

def produce_rnet_detection_train_dataset():
    """
    产生用于训练rnet的detection数据集
    """
    data
    with open(cfg.PNET_TRAIN_FORMATTER_TXT_PATH,"r") as f:
        data = f.readlines();

    data = data.split()
    detector = detector()

    for line in data:
        annotations = line.split()
        img = cv2.imread(annotations[0])

        width = img.shape[1]
        height = img.shape[0]

        ratio = 0.79
        #临时存放每张图片上产生的所有输出框
        boxes_cls_temp = []
        while min(width,height) > 24 :
            cls,bbr,_ = detector.predict(img)

            #计算出预测positive框的真实坐标值，并将其记录到boxes_cls_temp中
            bc = np.hstack(cls,bbr)
            keeps = np.where()

            width = width*ratio
            height = height*ratio
            img = cv2.resize(img,(width,height))

        remain = nms(boxes_cls_temp[:,:4],boxes_cls_temp[4],0.5)
        #对剩余的框，为其打标签，pos,par,neg,并产生用于训练的图片
        






