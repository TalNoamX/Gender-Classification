from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import tensorflow as tf

batch_size = 75
IMG_HEIGHT = 50
IMG_WIDTH = 50
test_image_generator = ImageDataGenerator(rescale=1. / 255)
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\grey-Test",
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')

with open(
        r"C:\Users\user1\PycharmProjects\gender-classification-1\MLP\results\MLP_model.json") as json_file:
    data = json.load(json_file)
    data = json.dumps(data)
    model = tf.keras.models.model_from_json(data)
model.load_weights(r"C:\Users\user1\PycharmProjects\gender-classification-1\MLP\results\weights_MLP_model.h5")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
results_test = model.evaluate_generator(test_data_gen)
print('test loss, test acc:, Precision:, Recall', results_test)
acc = results_test[1]
precision = results_test[2]
recall = results_test[3]
f_measure = 2 * (precision * recall) / (precision + recall)
print("acc: ", acc)
print("F-Measure: ", f_measure)