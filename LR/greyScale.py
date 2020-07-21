####first arg => where to
####second arg=> from where
####third arg=> which postfix


from PIL import Image
import glob
import os

path = "C:\\Users\\user1\PycharmProjects\gender-classification-1\Dataset\grey-Test\Female"
folder = "C:\\Users\\user1\PycharmProjects\gender-classification-1\Dataset\Test\Female" +  r'\*.' + "jpg"
num = 0
for filename in glob.glob(folder):
    im = Image.open(filename).convert('L')
    im = im.resize((50, 50), Image.ANTIALIAS)
    n_path = os.path.join(path, str(num)) + '.' + "jpg"
    im.save(n_path)
    num += 1