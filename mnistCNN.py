
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[3]:

def w(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


# In[4]:

def b(shape):
    inital = tf.constant(0.1, tf.float32, shape)
    return tf.Variable(inital)


# In[5]:

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')


# In[6]:

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[7]:

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28,1])


# In[8]:

# 第一个卷积层 strides=1 图片大小不变，但高度方向变为32
wc1 = w([5, 5, 1, 32])
bc1 = b([32])
conv1 = tf.nn.relu(conv2d(x_image, wc1) + bc1)


# In[9]:

# 第一个maxpooling层  tf.nn.conv2d和tf.nn.max_pool input和output都是[28,28,n]的数据
pool1 = max_pool(conv1)


# In[10]:

# 第二个conv pooling
wc2 = w([5, 5, 32, 64])
bc2 = b([64])
conv2 = tf.nn.relu(conv2d(pool1, wc2) + bc2)
pool2 = max_pool(conv2)


# In[11]:

wf1 = w([3136, 1024])
bf1 = b([1024])
indata = tf.reshape(pool2,[-1, 3136])
fc1 = tf.nn.relu(tf.matmul(indata, wf1) + bf1)


# In[12]:

# drop层 降低过拟合 此处暂定为0.5 也可以采用placeholder占位
fc1d = tf.nn.dropout(fc1, 0.5)


# In[13]:

wf2 = w([1024, 10])
bf2 = b([10])
pre = tf.nn.softmax(tf.matmul(fc1d, wf2) + bf2)


# In[14]:

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pre), reduction_indices=[1]))  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)


# In[15]:

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pre, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  


# In[16]:

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y: batch_ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))
# result
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


# In[20]:

saver = tf.train.Saver()
save_path = saver.save(sess, "C:/Users\hasee\model.ckpt")
print(save_path)


# In[ ]:



