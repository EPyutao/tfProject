使用CNN完成一个验证码识别demo，详细代码见github(使用版本Tensorflow 1.0)：  
### 验证码生成
常规的使用python生成验证码方法是生成字母与数字的组合，将单个元素进行旋转，添加背景色，噪点，干扰线等。本文中使用ImageCaptcha来生成验证码。该package可以加载自己的字体图片库，并在生成的验证码上随机添加30个噪点并绘制噪声曲线。使用 pip 安装ImageCaptcha，编写一个生成随机验证码list的函数，并将其转换为str数据结构，调用ImageCaptcha.generate(str)生成验证码图片。调用PIL库中的Image.open() 和 numpy.array()将图片转换为np.ndarray数据结构。文件为gen_captcha.py

### 模型与训练
构造工具函数，包含ImageCaptcha生成的验证码转换为灰度图，该图大小为[60,160,3]；将标签文本转换为向量；将向量转换为标签文本；生成训练batch，size=64，将灰度图转换为一维数据。模型部分采用CNN模型，使用3层卷积2层全连接，对输出层的数据计算和标签的sigmod_cross_entropy作为损失函数。__当模型精度达到0.9时停止训练，此时训练步数为13400步，相当于训练集达到了857600张图片，而验证码生成的图片总共可以达到八百万张，因此只使用了其中十分之一的数据，就训练到了90%的准确率。__ 文件为train.py。保存模型后，同级目录下出现4个文件，checkpoint存储了model的位置，其余为二进制文件。而在恢复过程中，调用tf.train.latest_checkpoint函数则是直接检索该文件内内容，并返回给saver.restore的save_path。

### 测试
调用同一级目录下保存的训练模型，文件为test.py：
<pre><code>
from train import *
from gen_captcha import gen_captcha_text_and_image


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)


text, image = gen_captcha_text_and_image()
print(image.shape)


image = convert2gray(image)
print(image.shape)
image = image.flatten() / 255
print(image.shape)

predict_text = crack_captcha(image)

print("正确: {}  预测: {}".format(text, predict_text))
</code></pre>
最终测试结果如下：
![right](http://epyutao.oss-cn-shanghai.aliyuncs.com/figure_2.png)
<pre><code>
正确: jkMf  预测: jkMf
</code></pre>
反复测试中，也发现了错误。
![wrong](http://epyutao.oss-cn-shanghai.aliyuncs.com/figure_1.png)
<pre><code>
正确: e8g2  预测: e8B2
</code></pre>
