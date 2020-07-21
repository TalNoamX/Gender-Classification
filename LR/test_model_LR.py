import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import random


###functions###
def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))


def return_class(p):
    return 'WOMEN' if p >= 0.5 else 'MEN'


def prepare_data(path_for_men_folder, path_for_women_folder):
    test_men_images = []
    test_women_images = []
    for image in glob.glob(path_for_men_folder):
        im = Image.open(image)
        test_men_images.append(np.reshape(im, (50 * 50)) / 255.)
    for image in glob.glob(path_for_women_folder):
        im = Image.open(image)
        test_women_images.append(np.reshape(im, (50 * 50)) / 255.)
    return (test_men_images, test_women_images)


def read_images(path_for_women_folder, path_for_men_folder):
    women_folder = path_for_women_folder
    men_folder = path_for_men_folder
    women_image_list = []
    men_image_list = []

    for filename in glob.glob(women_folder):
        im = Image.open(filename)
        women_image_list.append(np.reshape(im, (50 * 50)) / 255.)

    for filename in glob.glob(men_folder):
        im = Image.open(filename)
        men_image_list.append(np.reshape(im, (50 * 50)) / 255.)
    return (women_image_list, men_image_list)


def prepare_data2(path_for_women_folder, path_for_men_folder):
    tuple_of_both_images_sets = read_images(path_for_women_folder, path_for_men_folder)
    shuffled_tupels_of_all_data = shuffle_images(tuple_of_both_images_sets)
    train_data_x = []
    train_data_y = []
    for tuple_of_image_class in shuffled_tupels_of_all_data:
        train_data_x.append(tuple_of_image_class[0])

    for tuple_of_image_class in shuffled_tupels_of_all_data:
        train_data_y.append([tuple_of_image_class[1]])

    data_x = np.asarray(train_data_x)
    data_y = np.asarray(train_data_y)
    return (data_x, data_y)


def shuffle_images(tuple_of_both_images_sets):
    train_set_data_tuples = []

    for image in tuple_of_both_images_sets[0]:
        train_set_data_tuples.append((image, 1))

    for image in tuple_of_both_images_sets[1]:
        train_set_data_tuples.append((image, 0))

    random.shuffle(train_set_data_tuples)
    random.shuffle(train_set_data_tuples)
    return train_set_data_tuples


def load_model(path_for_model, name_for_model):
    sess = tf.compat.v1.Session()
    # First let's load meta graph and restore weights
    saver = tf.compat.v1.train.import_meta_graph(path_for_model + name_for_model)
    saver.restore(sess,tf.train.latest_checkpoint(path_for_model))

    graph = tf.get_default_graph()
    W = graph.get_tensor_by_name("Ws:0")
    b = graph.get_tensor_by_name("bs:0")
    x = graph.get_tensor_by_name("xs:0")
    y_ = graph.get_tensor_by_name("y_s:0")
    return (sess, W, b, x, y_)


###main###
def main():
    path_for_model = r'C:\Users\user1\PycharmProjects\gender-classification-1\results\LR-results'
    name_for_model = r'\model.ckpt-100.meta'
    path_for_men_test = r'C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\grey-Test\Male\*.jpg'
    path_for_women_test = r'C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\grey-Test\Female\*.jpg'
    tuples_of_all_data = prepare_data(path_for_men_test, path_for_women_test)

    pred_for_men = []
    pred_for_women = []

    (sess, W, b, x, y_) = load_model(path_for_model, name_for_model)

    for men_image in tuples_of_all_data[0]:
        pred_for_men.append(return_class(logistic_fun(np.matmul(men_image, sess.run(W)) + sess.run(b))))

    num_of_men_recognized = 0
    index = 0
    for _class in pred_for_men:
        if _class == 'MEN':
            num_of_men_recognized += 1

    for women_image in tuples_of_all_data[1]:
        pred_for_women.append(return_class(logistic_fun(np.matmul(women_image, sess.run(W)) + sess.run(b))))

    num_of_women_recognized = 0
    for _class in pred_for_women:
        if _class == 'WOMEN':
            num_of_women_recognized += 1
    (data_x, data_y) = prepare_data2(path_for_women_test, path_for_men_test)
    graph = tf.get_default_graph()
    loss = graph.get_tensor_by_name("loss_func:0")
    print(loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}))
    print('                         ***actual class***\n                            MEN:           WOMEN:')
    print('***Predicted class***')
    print('                      men   ', num_of_men_recognized, '             ',
          len(pred_for_women) - num_of_women_recognized)
    print('                      women   ', len(pred_for_men) - num_of_men_recognized, '             ',
          num_of_women_recognized)

    accuracy = ((num_of_women_recognized + num_of_men_recognized) / (len(pred_for_men) + len(pred_for_women)))
    precision = num_of_men_recognized / (num_of_men_recognized + (len(pred_for_men) - num_of_men_recognized))
    recall = num_of_men_recognized / (num_of_men_recognized + (len(pred_for_women) - num_of_women_recognized))

    print('model tasted from path: ', path_for_model)
    print('accuracy: %.3f' % (accuracy))
    print('precision: %.3f ' % (precision))
    print('recall: %.3f ' % (recall))
    print('f-measure: %.3f ' % (2 * ((precision * recall) / (precision + recall))))


if __name__ == '__main__':
    main()