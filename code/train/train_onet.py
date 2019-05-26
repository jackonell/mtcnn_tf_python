import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_factory.tfrecord_producer import read_batch_data_from_multi_tfrecord
from train import train
from net.cfgs import cfg

def train_onet():
    ratio = [1,0.5,1]
    model_name = "ONet"
    data_dir = cfg.ONET_DIR
    size = 48
    iterations = 120000 #需要根据图片数计算epoch，以确定每一批次的图片数

    train(read_batch_data_from_multi_tfrecord,ratio,model_name,data_dir,size,iterations)

if __name__ == "__main__":
    train_onet()



