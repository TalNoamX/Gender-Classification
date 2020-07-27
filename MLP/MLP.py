from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import matplotlib.pyplot as plt

###compilation flags to make TF more efficient
os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'



def main():
    ###predeclared parameters for the learning
    batch_size = 75
    epochs = 75
    IMG_HEIGHT = 50
    IMG_WIDTH = 50
    ###all data sets will use as train set, validation set and test set
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
    test_image_generator = ImageDataGenerator(rescale=1. / 255)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\grey-Train",
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\grey-Validation",
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=r"C:\Users\user1\PycharmProjects\gender-classification-1\Dataset\grey-Test",
                                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                             class_mode='binary')
    ###building the model
    model = Sequential([
        Flatten(),
        Dense(2500, activation=None),
        Dense(512, activation=None),
        Dense(1, activation='sigmoid')

    ])
    ###complinig the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=6500 // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=2001 // batch_size
    )

    ###summary of the model after traning
    print('\nhistory dict:', history.history)
    ###saving the model and weights as a json and h5 files
    json_str = model.to_json()
    with open(r'C:\Users\user1\PycharmProjects\gender-classification-1\MLP\results\MLP_model.json', 'w') as outfile:
        json.dump(json.loads(json_str), outfile, indent=4)  # Save the json on a file
        model.save_weights(r"C:\Users\user1\PycharmProjects\gender-classification-1\MLP\results\weights_MLP_model.h5", save_format="h5")
    print("Saved model to disk")
    ###evaluating the model on the test data
    print('\n# Evaluate on test data')
    results_test = model.evaluate_generator(test_data_gen)
    print('test loss, test acc:', results_test)
    ####printing the model as a graph
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
    main()