import numpy as np
from random import random
import tensorflow as tf
from sklearn.model_selection import train_test_split


def generate_dataset(num_samples, test_size ):
    X = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in X])
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_dataset(5000, 0.3)

    # build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    # compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss="MSE")

    # train
    model.fit(X_train, y_train, epochs=100)

    # evaluate model
    print("\nModel evaluation:")
    model.evaluate(X_test, y_test, verbose=1)

    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    print(f"\nSome prediction {predictions}")
    for d, p in zip(data, predictions):
        print(f"{d[0]} + {d[1]} = {p[0]}")