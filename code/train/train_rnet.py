from data_factory.tfrecord_producer import read_batch_data_from_multi_tfrecord
from train import train
from net.cfgs import cfg

def train_rnet():
    ratio = [1,0.5,0.5]
    model_name = "RNet"
    data_dir = cfg.RNET_DIR
    size = 24
    iterations = 120000 #需要根据图片数计算epoch，以确定每一批次的图片数

    train(read_batch_data_from_multi_tfrecord,ratio,model_name,data_dir,size,iterations)

if __name__ == "__main__":
    train_rnet()



