import json
import sys
import os
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


def data(batch_size, img_height, img_width):
    train_image_generator = ImageDataGenerator(rescale=1. / 255)
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    test_image_generator = ImageDataGenerator(rescale=1. / 255)
    train_set = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                          directory=r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\grey-Train",
                                                          shuffle=True,
                                                          target_size=(img_height, img_width),
                                                          class_mode='binary')
    valid_set = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\grey-Validation",
                                                               target_size=(img_height, img_width),
                                                               class_mode='binary')
    test_set = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\grey-Test",
                                                        target_size=(img_height, img_width),
                                                        class_mode='binary')
    print("Load data successfully!~")
    return train_set, test_set, valid_set


def main():
    batch_size = 54
    epochs = 100
    img_height = 50
    img_width = 50
    train_set, test_set, valid_set = data(batch_size, img_height, img_width)

    model = Sequential([
        Flatten(),
        Dense(1, activation='sigmoid')

    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit_generator(
        train_set,
        steps_per_epoch=5003 // batch_size,
        epochs=epochs,
        validation_data=valid_set,
        validation_steps=2001 // batch_size
    )

    print('\nhistory dict:', history.history)
    json_str = model.to_json()
    with open(r'C:\Users\user1\PycharmProjects\gender-classification-1\Logistic Regression\models\LR_model.json',
              'w') as outfile:
        json.dump(json.loads(json_str), outfile, indent=4)
        model.save_weights(
            r"C:\Users\user1\PycharmProjects\gender-classification-1\Logistic Regression\models\weights_LR_model.h5",
            save_format="h5")
    print("Saved model to disk")
    print('\n# Evaluate on test data')
    results_test = model.evaluate_generator(test_set)
    print('test loss, test acc:', results_test)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
