## 简介
基于TensorFlow实现的MTCNN.

## 环境
* TensorFlow 1.2.1
* python 3.6.3
* opencv 3.1.0

所有的代码均在code目录下，首先执行如下命令：
``` Bash
cd code
python setup.py
```


## 测试
更改test.py中的图片地址，检测结果会保存在当前目录

## 数据
1. 人脸检测：[WIDER_Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
2. 特征点检测：[CNN_FacePoint](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)

将数据解压到data/origin目录

## 训练
### 1.PNet
* 产生训练数据：python data_factory/pnet_data_producer.py
* 训练PNet：python train/train_pnet.py
### 2.RNet
* 产生训练数据：python data_factory/rnet_data_producer.py
* 训练PNet：python train/train_rnet.py
### 3.ONet
* 产生训练数据：python data_factory/onet_data_producer.py
* 训练PNet：python train/train_onet.py


## 参考
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. [MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)
3. [MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
