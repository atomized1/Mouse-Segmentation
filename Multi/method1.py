import tensorflow as tf
import nibabel as nib
import keras
import numpy as np
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


def convertTruth(mask):
    #Designing a 1-hot array that can be compared to the output of the larger model
    newTruth = np.empty((len(mask), len(mask[0]), len(mask[0, 0]), 332), dtype=np.dtype('int32'))
    for x in range(0, len(mask)):
        for y in range(0, len(mask[0])):
            for z in range(0, len(mask[0, 0])):
                new = np.zeros(332)
                if mask[x, y, z] < 1000:
                    new[int(mask[x, y, z])] = 1
                    newTruth[x, y, z] = new
                else:
                    new[int(mask[x, y, z]) - 1000 + 165] = 1
                    newTruth[x, y, z] = new
    return newTruth


def main():
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

    output = keras.layers.Conv2D(filters=332, kernel_size=(3, 3), activation='sigmoid', padding='same')(dconv1b)

    model = keras.models.Model(input_layer, output)
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[keras.metrics.binary_accuracy, dice_metric])

    arrayData, layerTruth = getData()
    arrayData = np.rot90(arrayData, axes=(1, 3))
    layerTruth = np.rot90(layerTruth, axes=(1, 3))

    print(len(layerTruth))
    print(len(layerTruth[0]))
    print(len(layerTruth[0, 0]))
    print(layerTruth[0, 0, 0])
    arrayTruth = convertTruth(layerTruth)

    print(len(arrayData))

    history = model.fit(arrayData, arrayTruth, epochs=epochs, batch_size=100)

    model.save(os.path.join(dirnam, "modelsGlobal/Model"))


if __name__ == "__main__":
    main()
