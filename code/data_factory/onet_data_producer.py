import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from net.cfgs import cfg
from net.mtcnn import Mtcnn
from rnet_data_producer import detection_data,landmark_data
from tfrecord_producer import produce_train_tfrecord_in_multi_file


if __name__ == "__main__":
    size = 48
    mtcnn = Mtcnn("RNet",[0.3,0.1,0.0])
    data_dir = cfg.ONET_DIR

    #产生训练数据
    # detection_data(mtcnn,size,data_dir)
    landmark_data(mtcnn,size,data_dir)

    #产生tfrecord
    # produce_train_tfrecord_in_multi_file(data_dir)

