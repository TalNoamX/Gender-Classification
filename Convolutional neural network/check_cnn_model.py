import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import tensorflow as tf


def main():
    batch_size = 64
    img_height, img_width = 200, 200
    test_image_generator = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\Test_200",
                                                             target_size=(img_height, img_width),
                                                             class_mode='binary')

    with open(
            r'C:\Users\user1\PycharmProjects\gender-classification-1\Convolutional neural network\model\CNN_model.json') as json_file:
        data = json.load(json_file)
        data = json.dumps(data)
        model = tf.keras.models.model_from_json(data)
    model.load_weights(
        r"C:\Users\user1\PycharmProjects\gender-classification-1\Convolutional neural network\model\weights_CNN_model.h5")
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    results_test = model.evaluate_generator(test_data_gen)
    print('test loss:' + str(results_test[0]) + '   test accurecy' + str(results_test[1]))
    precision = results_test[2]
    recall = results_test[3]
    f_measure = 2 * (precision * recall) / (precision + recall)
    print("F-Measure: ", str(f_measure))

if __name__ == '__main__':
    sys.exit(main())