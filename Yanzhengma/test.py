from train import *
import matplotlib.pyplot as plt
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

plt.imshow(image)
plt.show()

image = convert2gray(image)
print(image.shape)

image = image.flatten() / 255
print(image.shape)

predict_text = crack_captcha(image)

print("正确: {}  预测: {}".format(text, predict_text))