####first arg => where to
####second arg=> from where
####third arg=> which postfix


from PIL import Image
import glob
import os

path = "C:\\Users\\user1\PycharmProjects\gender-classification-1\Dataset\knn-train\Male"
folder = "C:\\Users\\user1\PycharmProjects\gender-classification-1\Dataset\Train\Male" + r'\*.' + "jpg"
num = 2000
print("Start converting Test Male...")
counter = 0
for filename in glob.glob(folder):
    if counter == 2000:
        break
    counter += 1
    im = Image.open(filename).convert('L')
    im = im.resize((200, 200), Image.ANTIALIAS)
    n_path = os.path.join(path, str(num)) + '.' + "jpg"
    im.save(n_path)
    num += 1

print("Done.")
