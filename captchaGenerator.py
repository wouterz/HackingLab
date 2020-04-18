from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('choice', choices=['uppercase', 'lowercase', 'digits'], help='uppercase, lowercase or digits')
parser.add_argument('--images_per_class', type=int, help="How many images should be generated per class", default=10)
parser.add_argument('--size', type=int, help="Size of the captcha image", default=60)
parser.add_argument('--font', default="")
args = parser.parse_args()
choice = args.choice
size = args.size
number_of_images = args.images_per_class

def createCaptcha(items, directory, size, number_of_images, font=""):
    fontList = []
    fstring = ""
    if(len(font)>0):
        fontList.append(font)
        fstring+=font.split(".")[0]+"_"
    savedir = "./images/{}captcha_{}_{}".format(fstring,directory, size)
    try:
        os.stat(savedir)
    except:
        os.makedirs(savedir)
    for i in range(len(items)):
        for j in range(number_of_images):   
            image_captcha = ImageCaptcha(fonts=fontList, width=size, height=size)
            image = image_captcha.generate_image(items[i])
            #image.show()
            location = '{}/{}_{}_{}.png'.format(savedir,directory,items[i],j)
            if(not os.path.exists(location)):
                image.save(location)
items = []
#choice = ""
choices = {
    "lowercase" : ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'],
    "uppercase" : list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
    "digits" : list("0123456789")
}

createCaptcha(choices[choice], choice, size, number_of_images, font=args.font)

