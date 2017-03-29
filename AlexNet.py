import tensorflow as tf

batch_size = 32
num_batches = 100

# AlexNet
def Alexnet(images):
    # size of images = 224 * 224
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, strides=[1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        parameters += [kernel, biases]

    lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        parameters +=[kernel, biases]

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        parameters +=[kernel, biases]

    # conv5 with pool
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        conv5 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        parameters +=[kernel, biases]

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    return pool5, parameters

def run_main():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
        pool5, parameter = Alexnet(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        # L2 regularization
        grad = tf.gradients(tf.nn.l2_loss(pool5), parameter)
        for i in range(num_batches):
            sess.run(grad)

run_main()
