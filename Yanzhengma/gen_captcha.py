from captcha.image import ImageCaptcha
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import random

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    # 后续对输入图片的尺寸做了限制，如果修改此处captcha_size，后面也要修改
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)  # 随机选择
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()  # captcha_text为list数据结构
    captcha_text = ''.join(captcha_text)  # captcha_text转换为str数据结构

    captcha = image.generate(captcha_text)  # 根据str生成验证码
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件
    # print(type(captcha))
    captcha_image = Image.open(captcha)
    # print(type(captcha_image))
    captcha_image = np.array(captcha_image)
    # print(type(captcha_image))
    return captcha_text, captcha_image


if __name__ == '__main__':
    # 测试
    '''
    text, image = gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()
    '''
