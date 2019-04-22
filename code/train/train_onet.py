from data_factory.tfrecord_producer import read_batch_data_from_single_tfrecord
from train import train
from net.cfgs import cfg

def train_pnet():
    ratio = [1,0.5,0.5]
    model_name = "PNet"
    data_dir = cfg.PNET_DIR
    size = 12
    iterations = 120000 #需要根据图片数计算epoch，以确定每一批次的图片数

    train(read_batch_data_from_single_tfrecord,ratio,model_name,data_dir,size,iterations)

if __name__ == "__main__":
    train_pnet()



