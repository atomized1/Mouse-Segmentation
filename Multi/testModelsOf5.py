import tensorflow as tf
import nibabel as nib
import keras
import keyboard
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

dirnam = os.path.dirname(__file__)

def dice_metric(y_true, y_pred):

    y_pred = tf.math.argmax(y_pred, axis=3)
    #print(y_pred)
    y_true = tf.math.argmax(y_true, axis=3)
    #print(y_true)

    inse = tf.equal(y_pred, y_true)
    #print(inse)
    inse = tf.reduce_sum(tf.cast(inse, tf.float32))
    #print(inse)
    l = len(y_pred) * len(y_pred[0]) * len(y_pred[0, 0])
    r = len(y_true) * len(y_true[0]) * len(y_true[0, 0])
    l = tf.cast(l, tf.float32)
    r = tf.cast(r, tf.float32)

    hard_dice = (2. * inse) / (l + r)

    #hard_dice = tf.reduce_mean(hard_dice)

    return hard_dice


def dice_metric_label(y_true, y_pred, label):
    y_predFiltered = y_pred == label
    if label < 166:
        y_trueFiltered = y_true == label
    else:
        y_trueFiltered = y_true == label + 1000 - 165
    y_trueReshaped = np.zeros([len(y_true), len(y_true[0]), len(y_true[0, 0])])
    for x in range(0, len(y_true)):
        for y in range(0, len(y_true[0])):
            for z in range(0, len(y_true[0, 0])):
                y_trueReshaped[x, y, z] = y_trueFiltered[x, y, z, 0]

    y_predFiltered = y_predFiltered.astype(int)
    y_trueReshaped = y_trueReshaped.astype(int)
    intersect = y_predFiltered & y_trueReshaped
    intersect = intersect.astype(int)
    intersect = intersect.sum()
    y_predFiltered = y_predFiltered.sum()
    y_trueReshaped = y_trueReshaped.sum()

    print(y_trueReshaped, y_predFiltered)

    hard_dice = 2 * intersect / (y_predFiltered + y_trueReshaped)

    f = open("DiceScores.txt", "a")
    f.write(str(label) + " " + str(hard_dice) + "\n")
    f.close()

    print(label, hard_dice)
    return y_predFiltered, y_trueReshaped


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

    return arrayTruth, arrayPredictData, arrayPredictTruth


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

    arrayData = np.empty([200 * len(imageList), 1, 180, 100])
    arrayTruth = np.empty([200 * len(imageList), 1, 180, 100])

    #Loading in each slice.  The start value offsets by the total amount of slices loaded, so not slice is overridden
    start = 0
    print("The length is " + str(len(imageList)))
    count = 0
    for file in range(0, len(imageList)):
        print("file " + str(count))
        image = nib.load(os.path.normpath(os.path.join(dirnam, imageList[file])))
        data = image.get_fdata()
        data = normalize(data) #Adjusting the data using mean and std
        mask = nib.load(os.path.normpath(os.path.join(dirnam, maskList[file])))
        truth = mask.get_fdata()

        data = np.rot90(data, axes=(0, 1))
        truth = np.rot90(truth, axes=(0, 1))
        if len(data) < 180:
            truth = np.append(truth, np.zeros((180 - len(data), len(truth[0]), 100)), axis=0)
            data = np.append(data, np.zeros((180 - len(data), len(data[0]), 100)), axis=0)
        data = np.rot90(data, axes=(0, 1))
        truth = np.rot90(truth, axes=(0, 1))

        print(imageList[file])
        for x in range(0, min([len(data), len(truth)])):
            arrayData[x + start, 0] = data[x, :]
            arrayTruth[x + start, 0] = truth[x, 0:len(data[0])]
        start = start + 180
        count += 1
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


def convertTruth(mask):
    #Designing a 1-hot array that can be compared to the output of the larger model
    newTruth = np.empty((len(mask), len(mask[0]), len(mask[0, 0]), int(sys.argv[2])), dtype=np.dtype('int32'))
    mask = mask - 40
    for x in range(0, len(mask)):
        for y in range(0, len(mask[0])):
            for z in range(0, len(mask[0, 0])):
                new = np.zeros(int(sys.argv[2]))
                if 10 > mask[x, y, z] >= 0:
                    new[int(mask[x, y, z])] = 1
                    newTruth[x, y, z] = new
                else:
                    new[10] = 1
                    newTruth[x, y, z] = new
    return newTruth


def deconvertTruth(labels):
    newTruth = np.empty((len(labels), len(labels[0]), len(labels[0,0])), dtype=np.dtype('int32'))
    for x in range(0, len(labels)):
        print(x)
        for y in range(0, len(labels[0])):
            for z in range(0, len(labels[0, 0])):
                biggestNum = 0
                biggestLabel = 0
                for a in range(0, int(sys.argv[2])):
                    if biggestNum < labels[x,y,z,a]:
                        biggestNum = labels[x,y,z,a]
                        biggestLabel = a
                newTruth[x,y,z] = biggestLabel
    return newTruth


def imageGen(labels):
    plt.figure(1)
    print(labels[70, 60, 60])
    labels = labels.astype('int32')
    plt.imshow(labels[70, :, :])
    plt.savefig('visuals.png')

    for x in range(0, len(labels) - 200, 200):
        img = nib.Nifti1Image(labels[x:x + 200], np.eye(4))
        nib.save(img, 'results' + str(sys.argv[1]) + str(x) + '.nii.gz')


def multichannel(data):
    arrayData = np.empty((len(data), len(data[0]), len(data[0, 0]), 3))
    for x in range(1, len(data)-1):
        for y in range(0, len(data[0])):
            for z in range(0, len(data[0, 0])):
                arrayData[x, y, z, 0] = data[x-1, y, z]
                arrayData[x, y, z, 1] = data[x, y, z]
                arrayData[x, y, z, 2] = data[x+1, y, z]

    return arrayData


def sortLabels(data):
    totals = np.empty(333)
    for x in range(0, 333):
        totals[x] = np.sum(data == x)
    sortedIndex = np.argsort(totals)
    return sortedIndex


def main():
    trainingArrayData, arrayData, layerTruth = getData()
    arrayData = np.rot90(arrayData, axes=(1, 3))
    layerTruth = np.rot90(layerTruth, axes=(1, 3))
    layerTruth = convertTruth(layerTruth)
    arrayData = multichannel(arrayData)
    totalImage = np.zeros((len(layerTruth), len(layerTruth[0]), len(layerTruth[0, 0])))
    totalImage = totalImage + 400
    sortedIndexes = sortLabels(trainingArrayData)
    predictions = np.empty((34, len(layerTruth), len(layerTruth[0]), len(layerTruth[0, 0])))
    print("Predicting Labels")
    for x in range(0, 160, 10):
        model = keras.models.load_model(sys.argv[1] + '/Model' + str(x), custom_objects={"dice_metric": dice_metric})
        opt = keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[keras.metrics.binary_accuracy, dice_metric])
        model.evaluate(arrayData, layerTruth)
        history = model.predict(arrayData)
        history = deconvertTruth(history)
        predictions[int(x/10)] = history

    sortedIndexes = np.flip(sortedIndexes)
    print("Filling in from least to greatest")
    for x in sortedIndexes:
        model = int(x/10)
        label = x % 10
        for a in range(0, len(totalImage)):
            for b in range(0, len(totalImage[0])):
                for c in range(0, len(totalImage[0, 0])):
                    if totalImage[a, b, c] == 400 and predictions[model, a, b, c] == label:
                        totalImage[a, b, c] = x

        dice_metric_label(layerTruth, predictions[model] + (model * 10), x)
        img = nib.Nifti1Image(totalImage, np.eye(4))
        nib.save(img, 'checkpoint' + '.nii.gz')

    imageGen(totalImage)


if __name__ == "__main__":
    main()
