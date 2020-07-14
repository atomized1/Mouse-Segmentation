import tensorflow as tf
import nibabel as nib
import keras
import numpy as np
import keyboard
import matplotlib.pyplot as plt
import os

dirnam = 'RARE'  #os.path.dirname(__file__)
epochs = 50

def dice_metric(y_true, y_pred):
#A Fuction that calculates the dice score for the predicted and true regions
#Inputs: The predicted and true labels of what in the image is skull
#Outputs: A dice score

    threshold = 0.5

    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)
    mask = y_true > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.multiply(y_true, mask)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    hard_dice = (2. * inse) / (l + r)

    hard_dice = tf.reduce_mean(hard_dice)

    return hard_dice


def getData():
    # A fuction that loads in the data from a list of files
    # Inputs: none
    # Outputs: Arrays containing slices of each image

    alt = 0
    imageList = []
    maskList = []
    with open(os.path.normpath('files.txt')) as files:
        for arrayDataPath in files:
            if alt == 0:
                imageList.append(arrayDataPath.strip())
                alt = 1
            elif alt == 1:
                maskList.append(arrayDataPath.strip())
                alt = 0
    slices = 0
    print(os.path.join(dirnam, imageList[0]))
    for x in range(0, len(imageList)):
        image = nib.load(os.path.normpath(os.path.join(dirnam, os.path.split(imageList[x])[1])))
        data = image.get_fdata()
        slices = slices + len(data[0])
        print(slices)
    arrayData = np.empty([slices, 1, 148, 100])
    arrayTruth = np.empty([slices, 1, 148, 100])
    print(len(arrayData[0]))
    start = 0
    for file in range(0, len(imageList)):
        image = nib.load(os.path.normpath(os.path.join(dirnam, os.path.split(imageList[file])[1])))
        data = image.get_fdata()
        mask = nib.load(os.path.normpath(os.path.join(dirnam, 'Masks', os.path.split(maskList[file])[1])))
        truth = mask.get_fdata()
        data = np.rot90(data, axes=(0, 1))
        truth = np.rot90(truth, axes=(0, 1))
        for x in range(0, len(data)):
                arrayData[x + start, 0] = data[x, 0:148]
                arrayTruth[x + start, 0] = truth[x, 0:148]
        start = start + len(data)
        print("File Loaded")
    return arrayData, arrayTruth


def display(data, model):

    model.save('myModel')
    predictions = model.predict(data[50:51])
    predictions = np.round(predictions)

    plt.ion()
    plt.axis('off')
    plt.figure(0)

    print(len(predictions[0][0][0]))

    plt.imshow(data[50, :, :, 0], cmap='gray')
    plt.imshow(predictions[0, :, :, 0], alpha=0.2)
    plt.draw()
    plt.pause(0.01)

    x = 0
    View = 0

    while True:
        if View == 0:
            if x > len(data[50, 0, 0]) - 1:
                x = len(data[50, 0, 0]) - 1
            plt.clf()
            plt.imshow(data[50, :, :, x], cmap='gray')
            plt.imshow(predictions[0, :, :, x], alpha=0.2)
            plt.axis('off')
            plt.draw()
            plt.pause(0.01)
        if View == 1:
            if x > len(data[3][0]) - 1:
                x = len(predictions[0][0]) - 1
            plt.clf()
            plt.imshow(data[50, :, x, :], cmap='gray')
            plt.imshow(predictions[0, :, x, :], alpha=0.2)
            plt.axis('off')
            plt.draw()
            plt.pause(0.01)
        if View == 2:
            if x > len(data[0]) - 1:
                x = len(predictions[0]) - 1
            plt.clf()
            plt.imshow(data[50, x, :, :], cmap='gray')
            plt.imshow(predictions[0, x, :, :], alpha=0.2)
            plt.axis('off')
            plt.draw()
            plt.pause(0.01)

        if keyboard.read_key() == "t":
            x += 1
            print(x)
        elif keyboard.read_key() == "g":
            x -= 1
            print(x)
        elif keyboard.read_key() == "w":
            View = 0
        elif keyboard.read_key() == "e":
            View = 1
        elif keyboard.read_key() == "r":
            View = 2
        elif keyboard.read_key() == "y":
            break
        if x < 0:
            x = 0


def main():
    with tf.device('GPU:1'):
        input_layer = keras.layers.Input(shape=(100, 148, 1))
        conv1a = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
        conv1b = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1a)
        pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv1b)
        conv2a = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
        conv2b = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2a)
        pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2b)
        conv3a = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
        conv3b = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu', padding='same')(conv3a)

        dconv3a = keras.layers.Conv2DTranspose(filters=96, kernel_size=(3, 3), padding='same')(conv3b)
        dconv3b = keras.layers.Conv2DTranspose(filters=96, kernel_size=(3, 3), padding='same')(dconv3a)
        unpool2 = keras.layers.UpSampling2D(size=(2, 2))(dconv3b)
        cat2 = keras.layers.concatenate([conv2b, unpool2])
        dconv2a = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same')(cat2)
        dconv2b = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same')(dconv2a)
        unpool1 = keras.layers.UpSampling2D(size=(2, 2))(dconv2b)
        cat1 = keras.layers.concatenate([conv1b, unpool1])
        dconv1a = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same')(cat1)
        dconv1b = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same')(dconv1a)

        output = keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(dconv1b)

        cp_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(dirnam, "models/cp-{epoch:04d}.ckpt"), verbose=1,     save_weights_only=True, period=10)
        model = keras.models.Model(inputs=input_layer, output=output)
        opt = keras.optimizers.Adam()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[keras.metrics.binary_accuracy, dice_metric])

    arrayData, arrayTruth = getData()
    arrayData = np.rot90(arrayData, axes=(1, 3))
    arrayTruth = np.rot90(arrayTruth, axes=(1, 3))

    with tf.device('GPU:1'):
        history = model.fit(arrayData, arrayTruth, epochs=epochs, callbacks=[cp_callback], batch_size=50)

    print(history.history['loss'])

    xvals = np.arange(epochs)
    plt.figure(1)
    plt.plot(xvals, history.history['binary_accuracy'])
    plt.savefig('plotAccuracy.png')
    np.savetxt("Accuracy.csv", history.history['binary_accuracy'], delimiter=",")

    plt.figure(1)
    plt.plot(xvals, history.history['loss'])
    plt.savefig('plotLoss.png')
    np.savetxt("Loss.csv", history.history['loss'], delimiter=",")

    plt.figure(1)
    plt.plot(xvals, history.history['dice_metric'])
    plt.savefig('plotDice.png')
    np.savetxt("Dice.csv", history.history['dice_metric'], delimiter=",")

    display(arrayData, model)


if __name__ == "__main__":
    main()
