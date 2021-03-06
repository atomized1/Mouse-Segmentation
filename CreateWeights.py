import tensorflow as tf
import nibabel as nib
import keras
import numpy as np
import keyboard
import matplotlib.pyplot as plt
import os
from random import randint

dirnam = os.path.dirname(__file__)
epochs = 20

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


def dice_metric_thresh(y_true, y_pred, thresh):
#A Fuction that calculates the dice score for the predicted and true regions
#Inputs: The predicted and true labels of what in the image is skull
#Outputs: A dice score

    threshold = thresh

    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)
    mask = y_true > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.multiply(y_true, mask)
    y_true = tf.cast(y_true, dtype=tf.float32)

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

    #variables used to divide up data
    alt = 0
    countA = 0
    countB = 0

    #The empty arrays that will store filenames for each set
    imageListTrain = []
    imageListPredict = []
    maskListTrain = []
    maskListPredict = []

    #A short loop to split all the data into one of the 4 lists.
    with open(os.path.normpath(os.path.join(dirnam, '2.txt'))) as files:
        for arrayDataPath in files:
            if alt == 0:
                if countA <= 6:
                    imageListTrain.append(arrayDataPath.strip())
                    countA += 1
                else:
                    imageListPredict.append(arrayDataPath.strip())
                    countA += 1
                    if countA == 10:
                        countA = 0
                alt = 1
            elif alt == 1:
                if countB <= 6:
                    maskListTrain.append(arrayDataPath.strip())
                    countB += 1
                else:
                    maskListPredict.append(arrayDataPath.strip())
                    countB += 1
                    if countB == 10:
                        countB = 0
                alt = 0

    #Loading in all the data from the filenames using the initialize function defined below
    arrayData, arrayTruth = initialize(imageListTrain, maskListTrain)
    arrayPredictData, arrayPredictTruth = initialize(imageListPredict, maskListPredict)

    return arrayData, arrayTruth, arrayPredictData, arrayPredictTruth


def normalize(data):
    #initializing some variables so I can calculate mean intensity
    total = 0
    count = 0
    #A loop to calculate that mean
    for x in range(0, len(data)):
        for y in range(0, len(data[1])):
            for z in range(0, len(data[1][1])):
                total += data[x][y][z]
                count += 1
    mean = total / count

    #Using that mean to calculate variance, and then Standard deviation
    totalVariance = 0
    for x in range(0, len(data)):
        for y in range(0, len(data[1])):
            for z in range(0, len(data[1][1])):
                variance = pow(data[x][y][z] - mean, 2)
                totalVariance += variance
    Std = totalVariance / count

    #Adjusting all values appropriately
    for x in range(0, len(data)):
        for y in range(0, len(data[1])):
            for z in range(0, len(data[1][1])):
                data[x][y][z] -= mean
                data[x][y][z] /= Std

    return data


def initialize(imageList, maskList):
    #Counting how many slices we need to load in, then preallocating arrays of that size
    slices = 0
    print(os.path.join(dirnam, imageList[0]))
    for x in range(0, len(imageList)):
        image = nib.load(os.path.normpath(os.path.join(dirnam, imageList[x])))
        data = image.get_fdata()
        slices = slices + len(data[0])
        print(slices)

    arrayData = np.empty([slices, 1, 148, 100])
    arrayTruth = np.empty([slices, 1, 148, 100])

    #Loading in each slice.  The start value offsets by the total amount of slices loaded, so not slice is overridden
    start = 0
    for file in range(0, len(imageList)):
        image = nib.load(os.path.normpath(os.path.join(dirnam, imageList[file])))
        data = image.get_fdata()
        data = normalize(data) #Adjusting the data using mean and std
        mask = nib.load(os.path.normpath(os.path.join(dirnam, maskList[file])))
        truth = mask.get_fdata()
        data = np.rot90(data, axes=(0, 1))  #Rotating data so the thickest side is in the 0 index
        truth = np.rot90(truth, axes=(0, 1))
        for x in range(0, len(data)):
            arrayData[x + start, 0] = data[x, 0:148]
            arrayTruth[x + start, 0] = truth[x, 0:148]
        start = start + len(data)
        print("File Loaded")

    return arrayData, arrayTruth


def display(data, model):

    #Saving the model, and generating the visual to insure it worked
    model.save('myModel')
    predictions = model.predict(data[50:51])
    predictions = np.round(predictions)

    plt.ion()
    plt.axis('off')
    plt.figure(4)

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


def ROCGen(predict, truth):
    cutoffPoints = 100
    truePosRate = np.empty(cutoffPoints)
    falsePosRate = np.empty(cutoffPoints)
    for cutoff in range(0, cutoffPoints):
        mask = predict > cutoff/cutoffPoints
        mask = tf.cast(mask, dtype=tf.float32)
        predPos = tf.multiply(predict, mask)
        mask = truth > cutoff/cutoffPoints
        mask = tf.cast(mask, dtype=tf.float32)
        truthPos = tf.multiply(truth, mask)
        mask = predict < cutoff/cutoffPoints
        mask = tf.cast(mask, dtype=tf.float32)
        predNeg = tf.multiply(predict, mask)
        mask = truth < cutoff/cutoffPoints
        mask = tf.cast(mask, dtype=tf.float32)
        truthNeg = tf.multiply(truth, mask)

        truePos = tf.reduce_sum(tf.multiply(predPos, truthPos))
        falseNeg = tf.reduce_sum(tf.multiply(predNeg, truthPos))
        truePosRate[cutoff] = truePos/(truePos + falseNeg)

        falsePos = tf.reduce_sum(tf.multiply(predPos, truthNeg))
        trueNeg = tf.reduce_sum(tf.multiply(predNeg, truthNeg))
        falsePosRate[cutoff] = trueNeg/(falsePos + trueNeg)

    plt.figure(1)
    plt.plot(falsePosRate, truePosRate)
    plt.savefig('ROC.png')
    np.savetxt(os.path.join(dirnam, "ROCP.csv"), truePosRate, delimiter=",")
    np.savetxt(os.path.join(dirnam, "ROCF.csv"), falsePosRate, delimiter=",")


def main():
    #Layers to construct the model.
    #Each unit of the U-net consists of 2 convultional layers 1 pooling/unpooling layer
    #The kernel siez for these are 3x3 and 2x2 respectively
    #All layers pad to insure data dimensions stay constant before and after model
    #Skip connections connect pooling and upooling layers of the same data size using concatenation
    #All layers use rectiliar units, except the last layer which uses sigmoidal.
    #Rectilinear was chosen to effectively weed out unimportant data, and the sigmoidal was used because it looks like a percent
    #Email me any questions
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

    #This saves our model every 10 epochs, just incase it is better than the final/we crash
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(dirnam, "models/cp-{epoch:04d}.ckpt"), verbose=1, save_weights_only=True, period=10)

    #Compiling the model.  Learning rate turned down because experimentally it does better at this value
    #Cross entropy used rather than dice score because experimentally causes model to converge to local min faster
    model = keras.models.Model(inputs=input_layer, output=output)
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[keras.metrics.binary_accuracy, dice_metric])

    #Loading in all our data using massive getData function defined above.
    arrayData, arrayTruth, arrayEval, arrayTruthEval = getData()
    #Rotating data 1 more time, for thickness reasons
    arrayData = np.rot90(arrayData, axes=(1, 3))
    arrayTruth = np.rot90(arrayTruth, axes=(1, 3))
    arrayEval = np.rot90(arrayEval, axes=(1, 3))
    arrayTruthEval = np.rot90(arrayTruthEval, axes=(1, 3))

    #Saving the history of the model generation to be turned into graphs
    history = model.fit(arrayData, arrayTruth, epochs=epochs, callbacks=[cp_callback], batch_size=50)

    #
    xvals = np.arange(epochs)
    plt.figure(1)
    plt.plot(xvals, history.history['binary_accuracy'])
    plt.savefig('plotAccuracy.png')
    np.savetxt(os.path.join(dirnam, "Accuracy.csv"), history.history['binary_accuracy'], delimiter=",")

    plt.figure(2)
    plt.plot(xvals, history.history['loss'])
    plt.savefig('plotLoss.png')
    np.savetxt(os.path.join(dirnam, "Loss.csv"), history.history['loss'], delimiter=",")

    plt.figure(3)
    plt.plot(xvals, history.history['dice_metric'])
    plt.savefig('plotDice.png')
    np.savetxt(os.path.join(dirnam, "Dice.csv"), history.history['dice_metric'], delimiter=",")

    predictions = model.predict(arrayEval)

    ROCGen(predictions, arrayTruthEval)

    #display(arrayData, model)


if __name__ == "__main__":
    main()
