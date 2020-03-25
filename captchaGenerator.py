from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import os


size = 60

try:
    os.stat("./images/captcha_{}".format(size))
except:
    os.makedirs("./images/captcha_{}".format(size))
alphabet_lowercase = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

for i in range(26):
    for j in range(1000):   
        image_captcha = ImageCaptcha(width=size, height=size)
        image = image_captcha.generate_image(alphabet_lowercase[i])
        #image.show()
        image.save('./images/captcha_{}/Letter_{}_{}.png'.format(size,alphabet_lowercase[i],j))

