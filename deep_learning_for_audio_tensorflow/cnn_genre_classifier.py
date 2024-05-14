import numpy as np
import json
import os
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

def prepare_datasets(test_size, validation_size):

    # load data
    X, y = load_data(DATASET_PATH)
    # create train/ test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,test_size=validation_size)

    # transform to 3d array -> necessario para CNN
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def predict(model, X, y):

    X = X[..., np.newaxis]

    # prediction retorna a probabilidade para cada um dos generos
    prediction = model.predict(X)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print(f"Expected index: {y}, Predicted_index = {predicted_index}")

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

def build_model(input_shape):
    
    # create model
    # 3 layers de convolucao
    # flatten o output
    # Adicionar ele para o dense layer
    # output layer no softmax
    model = keras.Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed into tdense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
if __name__ == '__main__':
    # create train, validation and test sets
    # inputs, targets = load_data(DATASET_PATH)

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(test_size=0.25, validation_size=0.2)
    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    # train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), 
              batch_size=32, epochs=30)

    plot_history(history)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'Accuracy on test set is :{test_accuracy}')
    
    # make predictions on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)