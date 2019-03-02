X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH],name='X')
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN],name='Y')
keep_prob = tf.placeholder(tf.float32,name='keep_prob') # dropout

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # 第一个卷积层
    with tf.variable_scope('layer1-conv1'):
        w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]),name='weight')
        b_c1 = tf.Variable(b_alpha*tf.random_normal([32]),name='bias')
        relu1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))

    # 第一个池化层
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第二个卷积层
    with tf.variable_scope('layer3-conv2'):
        w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]),name='weight')
        b_c2 = tf.Variable(b_alpha*tf.random_normal([64]),name='bias')
        relu2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))

    # 第二个池化层
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第三个卷积层
    with tf.variable_scope('layer5-conv3'):
        w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]),name='weight')
        b_c3 = tf.Variable(b_alpha*tf.random_normal([64]),name='bias')
        relu3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))

    # 第三个池化层
    with tf.variable_scope('layer6-pool3'):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第一个全连接层
    with tf.variable_scope('layer7-fc1'):
        w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]),name='weight')
        b_d = tf.Variable(b_alpha*tf.random_normal([1024]),name='bias')
        fc1 = tf.reshape(pool3, [-1, w_d.get_shape().as_list()[0]])
        fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, w_d), b_d))
        fc1 = tf.nn.dropout(fc1, keep_prob)
    # 第二个全连接层
    with tf.variable_scope('layer8-fc2'):
        w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]),name='weight')
        b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]),name='bias')
        fc2 = tf.add(tf.matmul(fc1, w_out), b_out)

    return fc2
# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    #(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
    b = tf.constant(value=1,dtype=tf.float32)
    output_eval = tf.multiply(output,b,name='logits_eval')
    # loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(64,False)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.95:
                    saver.save(sess, model_path, global_step=step)
                    print("done")
                    break
            step += 1


