import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier as KNN_classifier
from sklearn.model_selection import train_test_split as splitSet


def knn(train_x, train_y, test_x, test_y, valid_x, valid_y):
    graph_x_value = list()
    graph_y_value = list()
    best_k = 0
    accuracy = 0
    counter = 0
    print("Start the knn algorithm!~")
    for k in range(1, 500, 2):
        model = KNN_classifier(n_neighbors=k)
        model.fit(train_x, train_y)
        graph_y_value.append(model.score(valid_x, valid_y))
        if graph_y_value[counter] > accuracy:
            accuracy = graph_y_value[counter]
            best_k = k
        print("Round: "+str(counter)+"   K is now: "+str(k)+"    best accuracy: " + str(accuracy))
        counter += 1
        graph_x_value.append(k)

    print("max acc at k=" + str(best_k) + " acc of " + str(accuracy))
    model = KNN_classifier(n_neighbors=best_k + 1)
    model.fit(train_x, train_y)
    print("Test Accuracy: " + str(model.score(test_x, test_y)))
    plt.plot(graph_x_value, graph_y_value)
    plt.show()


def data_prep():
    path_men = r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\knn-train\Male" + r'\*.' + "jpg"
    path_women = r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\knn-train\Female" + r'\*.' + "jpg"
    data_set = list()
    for filename in glob.glob(path_men):
        im = Image.open(filename)
        im = im.resize((200, 200), Image.ANTIALIAS)
        im = image.img_to_array(im)
        im /= 255
        data_set.append(im)

    for filename in glob.glob(path_women):
        im = Image.open(filename)
        im = im.resize((200, 200), Image.ANTIALIAS)
        im = image.img_to_array(im)
        im /= 255
        data_set.append(im)
    print("Loading dataset successfully!~ ")
    return np.array(data_set)


def main():
    data = data_prep()
    men_y = np.zeros(2000)
    woman_y = np.ones(2000)
    y = np.concatenate((men_y, woman_y), axis=0)
    train_x, test_x, train_y, test_y = splitSet(data, y, random_state=35, test_size=0.25)
    valid_x, test_x, valid_y, test_y = splitSet(test_x, test_y, random_state=35, test_size=0.5)
    print("Setting up arguments..")
    train_x = train_x[list(range(train_x.shape[0]))]
    train_y = train_y[list(range(train_x.shape[0]))]
    test_x = test_x[list(range(test_x.shape[0]))]
    test_y = test_y[list(range(test_x.shape[0]))]
    valid_x = valid_x[list(range(valid_x.shape[0]))]
    valid_y = valid_y[list(range(valid_x.shape[0]))]
    train_x = np.reshape(train_x, (train_x.shape[0], -1))
    test_x = np.reshape(test_x, (test_x.shape[0], -1))
    valid_x = np.reshape(valid_x, (valid_x.shape[0], -1))
    knn(train_x, train_y, test_x, test_y, valid_x, valid_y)


if __name__ == "__main__":
    sys.exit(main())
