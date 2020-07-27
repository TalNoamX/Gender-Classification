####first arg => where to
####second arg=> from where
####third arg=> which postfix


from PIL import Image
import glob
import os


path2 = "C:\\Users\\user1\PycharmProjects\gender-classification-1\Dataset\Test_200\Male"
folder2 = "C:\\Users\\user1\PycharmProjects\gender-classification-1\Dataset\Test\Male" + r'\*.' + "jpg"
num = 0
print("Start converting Test Male...")
for filename in glob.glob(folder2):
    im = Image.open(filename)
    im = im.resize((200, 200), Image.ANTIALIAS)
    n_path = os.path.join(path2, str(num)) + '.' + "jpg"
    im.save(n_path)
    num += 1

print("Done.")