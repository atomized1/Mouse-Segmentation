import tensorflow as tf
import nibabel as nib
import keras
import numpy as np
import matplotlib.pyplot as plt
import os

dirnam = os.path.dirname(__file__)
epochs = 20
BURST = 10

def dice_metric(y_true, y_pred):

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
    #arrayPredictData, arrayPredictTruth = initialize(imageListPredict, maskListPredict)

    return arrayData, arrayTruth, #arrayPredictData, arrayPredictTruth


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
        print(imageList[file])
        for x in range(0, len(data)):
            arrayData[x + start, 0] = data[x, 0:148]
            arrayTruth[x + start, 0] = truth[x, 0:148]
        start = start + len(data)
        print("File Loaded")

    return arrayData, arrayTruth


def detectNeighbors(mask):
    dectectingArray = np.zeros((332, 332))
    #cycling through every pixel of the image, except the last one, which will have no pixel after it
    for x in range(0, len(mask)):
        for y in range(0, len(mask[0])):
            for z in range(0, len(mask[0, 0]) - 1):
                if mask[x, y, z] < 1000:
                    if mask[x, y, z + 1] < 1000:
                        dectectingArray[int(mask[x, y, z]), int(mask[x, y, z + 1])] = 1
                    else:
                        dectectingArray[int(mask[x, y, z]), int(mask[x, y, z + 1] - 1000 + 165)] = 1
                else:
                    if mask[x, y, z + 1] < 1000:
                        dectectingArray[int(mask[x, y, z] - 1000 + 165), int(mask[x, y, z + 1])] = 1
                    else:
                        dectectingArray[int(mask[x, y, z] - 1000 + 165), int(mask[x, y, z + 1] - 1000 + 165)] = 1
    return dectectingArray


def createModels(neighbors, mask, image):
    #To loop through every label
    for x in range(0, 333):
        numOfNeigh = 0
        #Checking how many neighbors this label has
        for y in range(0, 332):
            if neighbors[x, y] == 1:
                numOfNeigh = numOfNeigh + 1

        print("Num of Neighbors")
        print(numOfNeigh)

        #Designing model to be essentially half a U-net
        input_layerN = keras.layers.Input(shape=(8, 8, 1))
        conv1aN = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layerN)
        conv1bN = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1aN)
        pool1N = keras.layers.MaxPool2D(pool_size=(2, 2))(conv1bN)
        conv2aN = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1N)
        conv2bN = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2aN)
        pool2N = keras.layers.MaxPool2D(pool_size=(4, 4))(conv2bN)
        conv3aN = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu', padding='same')(pool2N)
        conv3bN = keras.layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu', padding='same')(conv3aN)

        outputN = keras.layers.Conv2D(filters=numOfNeigh, kernel_size=(3, 3), activation='sigmoid', padding='same')(conv3bN)

        modelN = keras.models.Model(input_layerN, outputN)
        opt = keras.optimizers.Adam(learning_rate=0.0005)
        modelN.compile(optimizer=opt, loss='binary_crossentropy', metrics=[keras.metrics.binary_accuracy, dice_metric])

        numOfPixels = 0

        for a in range(0, len(mask)):
            for b in range(0, len(mask[0])):
                for c in range(0, len(mask[0, 0]) - 1):
                    if x < 166:
                        if mask[a, b, c] == x:
                            numOfPixels = numOfPixels + 1
                    else:
                        if mask[a, b, c] == x + 1000 - 165:
                            numOfPixels = numOfPixels + 1

        print("Num of Pixels")
        print(x)
        print(numOfPixels)

        data = np.empty((numOfPixels, 8, 8, 1))
        truth = np.zeros((numOfPixels, 1, 1, numOfNeigh))
        pixel = 0

        if numOfNeigh > 0:
            for a in range(0, len(image)):
                for b in range(0, len(image[0])):
                    print(b)
                    for c in range(0, len(image[0, 0]) - 1):
                        #If statement to convert from image values to consecutive values
                        if x < 166:
                            if mask[a, b, c] == x:
                                #If statements to check if we are on the edge of the image and account for that
                                if b < 4:
                                    if c < 4:
                                        data[pixel] = image[a, 0:8, 0:8]
                                    elif c > 144:
                                        data[pixel] = image[a, 0:8, 140:148]
                                    else:
                                        data[pixel] = image[a, 0:8, c-4:c+4]
                                elif b > 96:
                                    if c < 4:
                                        data[pixel] = image[a, 92:100, 0:8]
                                    elif c > 144:
                                        data[pixel] = image[a, 92:100, 140:148]
                                    else:
                                        data[pixel] = image[a, 92:100, c-4:c+4]
                                else:
                                    if c < 4:
                                        data[pixel] = image[a, b-4:b+4, 0:8]
                                    elif c > 144:
                                        data[pixel] = image[a, b-4:b+4, 140:148]
                                    else:
                                        data[pixel] = image[a, b-4:b+4, c-4:c+4]
                                neighborID = -1
                                if int(mask[a, b, c + 1]) < 1000:
                                    for d in range(0, int(mask[a, b, c + 1])):
                                        if neighbors[x, d] == 1:
                                            neighborID = neighborID + 1
                                else:
                                    for d in range(0, int(mask[a, b, c + 1]) - 1000 + 165):
                                        if neighbors[x, d] == 1:
                                            neighborID = neighborID + 1

                                truth[pixel, 0, 0, neighborID] = 1
                                pixel = pixel + 1
                        else:
                            if mask[a, b, c] == x + 1000 - 165:
                                if a < 4:
                                    if b < 4:
                                        data[pixel] = image[0:8, 0:8, c]
                                    elif b > 144:
                                        data[pixel] = image[0:8, 140:148, c]
                                    else:
                                        data[pixel] = image[0:8, b-4:b+4, c]
                                elif a > 96:
                                    if b < 4:
                                        data[pixel] = image[92:100, 0:8, c]
                                    elif b > 144:
                                        data[pixel] = image[92:100, 140:148, c]
                                    else:
                                        data[pixel] = image[92:100, b-4:b+4, c]
                                else:
                                    if b < 4:
                                        data[pixel] = image[a-4:a+4, 0:8, c]
                                    elif b > 144:
                                        data[pixel] = image[a-4:a+4, 140:148, c]
                                    else:
                                        data[pixel] = image[a-4:a+4, b-4:b+4, c]
                                neighborID = -1
                                if int(mask[a, b, c + 1]) < 1000:
                                    for d in range(0, int(mask[a, b, c + 1])):
                                        if neighbors[x, d] == 1:
                                            neighborID = neighborID + 1
                                else:
                                    for d in range(0, int(mask[a, b, c + 1]) - 1000 + 165):
                                        if neighbors[x, d] == 1:
                                            neighborID = neighborID + 1
                                truth[pixel, 0, 0, neighborID] = 1
                                pixel = pixel + 1

            history = modelN.fit(data, truth, epochs=20, batch_size=int(numOfPixels/20) + 1)
            modelN.save(os.path.join(dirnam, "models/" + str(x) + "modelN"))


def main():

    arrayDataTrain, layerTruthTrain, = getData()
    arrayDataTrain = np.rot90(arrayDataTrain, axes=(1, 3))
    layerTruthTrain = np.rot90(layerTruthTrain, axes=(1, 3))

    neighbors = detectNeighbors(layerTruthTrain)
    print(len(arrayDataTrain[0]))
    print(len(arrayDataTrain[0][0]))
    createModels(neighbors, layerTruthTrain, arrayDataTrain)


if __name__ == "__main__":
    main()

