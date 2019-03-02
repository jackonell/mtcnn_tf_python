from random import shuffle
import glob
import cv2
import numpy as np
import tensorflow as tf
import sys
#用tfrecord 可以加快速度，不用一张张图片的读取
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

shuffle_data = True  # shuffle the addresses before saving
cat_dog_train_path = 'train/*.jpg'
# read addresses and labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
print(addrs[0])
print(labels[0])
# Divide the data into 60% train, 20% validation, and 20% test
train_addrs  = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs    = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels   = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs   = addrs[int(0.8*len(addrs)):]

train_filename = 'train.tfrecords'
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    if not i%1000:
        print('Train data: {}/{}'.format(i,len(train_addrs)))
        sys.stdout.flush()

    img = load_image(train_addrs[i])
    label = train_labels[i]
    feature = {'train/label':_int64_feature(label),
             'train/image':_bytes_feature(tf.compat.as_bytes(img.tostring()))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

