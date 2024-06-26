import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = 'data.json'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # Convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def plot_history(history):
    
    fig, axis = plt.subplots(2)

    # Create accuracy subplot
    axis[0].plot(history.history['accuracy'], label='train_accuracy')
    axis[0].plot(history.history['val_accuracy'], label='test_accuracy')
    axis[0].set_ylabel('accuracy')
    axis[0].legend(loc='lower right')
    axis[0].set_title('Accuracy eval')

    axis[1].plot(history.history['loss'], label='train_loss')
    axis[1].plot(history.history['val_loss'], label='test_loss')
    axis[1].set_ylabel('loss')
    axis[1].set_xlabel('epoch')    
    axis[1].legend(loc='upper right')
    axis[1].set_title('loss eval')

    plt.show()

if __name__ == '__main__':
    # load data
    inputs, targets = load_data(DATASET_PATH)


    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(inputs,
                                                        targets,
                                                        test_size=0.3)
    # build the network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        # 1st hidden layer
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 2nd hidden layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 3rd hidden layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=10**-4)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()
    # train network
    history = model.fit(X_train, y_train, 
              validation_data=(X_test, y_test),
              epochs=100,
              batch_size=32)

    # Plot accuracy and error over the epochs
    plot_history(history)