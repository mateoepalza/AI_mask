import tensorflow as tf
import numpy as np
import os
import cv2

from sklearn.model_selection import train_test_split

epoch = 10 # number of iterations
width = 80 # width of the image in pixels
height = 80 # height of the image in pixels
categories = 2 # 1 (mask) or 0 (without mask)
test_size = 0.3 # percentage of the data that is going to be for testing

def main():
    # Gets the images and the labels
    images, labels = load_data()

    # split the data between training and testing
    labels =  tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=test_size
    )

    # get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs = epoch)
    
    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model
    model.save("first")

def load_data():
    # Create the data lists

    final_images = []
    labels = []

    # get the current directory and i move until our dataset
    data_folder = os.path.join(os.getcwd(), "dataset")
    # list the folders (0) => without mask / (1) => with mask
    folders = os.listdir(data_folder)

    # Iterate over both folders
    for folder in folders:
        # each folder
        dir_folder = os.path.join(data_folder,folder)
        # List the images
        images =  os.listdir(dir_folder)
        # iterate over all the images in the folder
        for image in images:
            # Read the image
            img_dir = cv2.imread(os.path.join(dir_folder, image))
            # resize the image
            img_res = cv2.resize(img_dir, (width, height))
            # Save the image in the matrix
            final_images.append(img_res)
            # Save the labels of the images
            labels.append(folder)
        
    return (final_images, labels)


def get_model():

    # initialize the model
    model = tf.keras.models.Sequential()
    
    # First convolution -> 60 filters with a kernel of 3x3 and an activation of type "relu"
    model.add(tf.keras.layers.Conv2D(60, (3,3), activation="relu", input_shape=(width, height, 3), padding="same"))

    # First max pooling with a window of 2x2
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    # Second convolution
    model.add(tf.keras.layers.Conv2D(80, (2,2), activation="relu", input_shape=(width/2, height/2,3), padding="same"))

    # Second max pooling with a window of 2x2
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    # Flatten layer
    model.add(tf.keras.layers.Flatten())

    # First Fully connected layer
    model.add(tf.keras.layers.Dense(10800, activation="relu"))

    # Second fully connected layer
    model.add(tf.keras.layers.Dense(5400, activation="relu"))

    # third fully connected layer
    model.add(tf.keras.layers.Dense(2700, activation="softmax"))

    # Dropout (prevent overfitting)
    model.add(tf.keras.layers.Dropout(0.3))

    # softmax activation
    model.add(tf.keras.layers.Dense(categories,activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


if __name__ == "__main__":
    main()