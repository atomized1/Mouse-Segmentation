import tensorflow as tf
import nibabel as nib
import keras
import numpy as np
import os

dirnam = os.path.dirname(__file__)
epochs = 20
BURST = 10

def dice_metric(y_true, y_pred):

    y_pred = tf.math.argmax(y_pred, axis=4)
    #print(y_pred)
    y_true = tf.math.argmax(y_true, axis=4)
    #print(y_true)

    inse = tf.equal(y_pred, y_true)
    #print(inse)
    inse = tf.reduce_sum(tf.cast(inse, tf.float32))
    #print(inse)
    l = len(y_pred) * len(y_pred[0]) * len(y_pred[0, 0]) * len(y_pred[0, 0, 0])
    r = len(y_true) * len(y_true[0]) * len(y_true[0, 0]) * len(y_pred[0, 0, 0])
    l = tf.cast(l, tf.float32)
    r = tf.cast(r, tf.float32)

    hard_dice = (2. * inse) / (l + r)

    #hard_dice = tf.reduce_mean(hard_dice)

    return hard_dice

def sensitivity1(y_true, y_pred):
    y_true = tf.math.argmax(y_true)
    y_pred = tf.math.argmax(y_pred)

    ones = tf.ones(shape=tf.shape(y_pred), dtype=tf.int64)
    y_true = tf.cast(tf.math.equal(y_true, ones), tf.int64)
    y_pred = tf.cast(tf.math.equal(y_pred, ones), tf.int64)

    neg_y_true = ones - y_true
    neg_y_pred = ones - y_pred
    fp = tf.reduce_sum(neg_y_true * y_pred)
    tn = tf.reduce_sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp)
    return specificity


def specificity1(y_true, y_pred):
    y_pred = tf.math.argmax(y_pred, axis=4)
    y_true = tf.math.argmax(y_true, axis=4)

    ones = tf.ones(shape=tf.shape(y_pred), dtype=tf.int64)
    zeros = tf.zeros(shape=tf.shape(y_pred), dtype=tf.int64)

    y_predPos = tf.math.equal(y_pred, ones)
    y_predNegative = tf.math.equal(tf.cast(y_predPos, tf.int64), zeros)
    y_truePos = tf.math.equal(y_true, ones)
    y_trueNegative = tf.math.equal(tf.cast(y_truePos, tf.int64), zeros)

    trueNeg = tf.equal(y_predNegative, y_trueNegative)
    falsePos = tf.equal(y_predPos, y_trueNegative)

    trueNeg = tf.cast(trueNeg, tf.float32)
    falsePos = tf.cast(falsePos, tf.float32)

    specificity = trueNeg / (trueNeg + falsePos)

    return specificity


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

    arrayData = np.empty([len(imageList), 1, 200, 180, 100])
    arrayTruth = np.empty([len(imageList), 1, 200, 180, 100])

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
        print(len(data))
        print(len(data[0]))
        data = np.rot90(data, axes=(0, 1))
        truth = np.rot90(truth, axes=(0, 1))

        print(imageList[file])
        arrayData[file, 0] = data[0:200, 0:180, 0:100]
        arrayTruth[file, 0] = truth[0:200, 0:180, 0:100]
        #for x in range(0, min([len(data), len(truth)])):
        #    arrayData[x + start, 0] = data[x, :]
        #    arrayTruth[x + start, 0] = truth[x, 0:len(data[0])]
        #start = start + 180
        #count += 1
        #print("File Loaded")

    return arrayData, arrayTruth


def multichannel(data):
    arrayData = np.empty((len(data), len(data[0]), len(data[0, 0]), 3))
    for x in range(1, len(data)-1):
        for y in range(0, len(data[0])):
            for z in range(0, len(data[0, 0])):
                arrayData[x, y, z, 0] = data[x-1, y, z]
                arrayData[x, y, z, 1] = data[x, y, z]
                arrayData[x, y, z, 2] = data[x+1, y, z]

    return arrayData



def convertTruth(mask):
    #Designing a 1-hot array that can be compared to the output of the larger model
    newTruth = np.empty((len(mask), len(mask[0]), len(mask[0, 0]), len(mask[0,0,0]), 3), dtype=np.dtype('int32'))
    for x in range(0, len(mask)):
        print(x)
        for y in range(0, len(mask[0])):
            for z in range(0, len(mask[0, 0])):
                for a in range(0, len(mask[0, 0, 0])):
                    new = np.zeros(3)
                    if mask[x,y,z,a] > 1000:
                        mask[x,y,z,a] -= 1000
                    if mask[x, y, z, a] == 0:
                        new[0] = 1
                        newTruth[x, y, z, a] = new
                    #elif mask[x,y,z,a] == 51 or mask[x,y,z,a] == 126:
                        #new[3] = 1
                        #newTruth[x,y,z,a] = new
                    #elif mask[x,y,z,a] == 64 or mask[x,y,z,a] == 166:
                        #new[4] = 1
                        #newTruth[x,y,z,a] = new
                    elif 1 <= mask[x,y,z,a] <= 68 or (129 >= mask[x, y, z, a] >= 118) or mask[x, y, z, a] == 131 or mask[x, y, z, a] == 151 or (
                            157 >= mask[x, y, z, a] >= 155) or mask[x, y, z, a] == 161 or mask[x, y, z, a] == 163 or mask[x, y, z, a] == 165:
                        new[1] = 1
                        newTruth[x, y, z, a] = new
                    elif (117 >= mask[x, y, z, a] >= 69) or mask[x, y, z, a] == 130 or (150 >= mask[x, y, z, a] >= 132) or (
                            154 >= mask[x, y, z, a] >= 152) or (160 >= mask[x, y, z, a] >= 158) or mask[x, y, z, a] == 162 or mask[x, y, z, a] == 164 or mask[x, y, z, a] == 166:
                        new[2] = 1
                        newTruth[x, y, z, a] = new

    return newTruth


def main():
    input_layer = keras.layers.Input(shape=(100, 200, 180, 1))
    conv1a = keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
    conv1b = keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv1a)
    pool1 = keras.layers.MaxPool3D(pool_size=(2, 2, 2))(conv1b)
    conv2a = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool1)
    conv2b = keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv2a)
    pool2 = keras.layers.MaxPool3D(pool_size=(2, 2, 2))(conv2b)
    conv3a = keras.layers.Conv3D(filters=96, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool2)
    conv3b = keras.layers.Conv3D(filters=96, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv3a)

    dconv3a = keras.layers.Conv3DTranspose(filters=96, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv3b)
    dconv3b = keras.layers.Conv3DTranspose(filters=96, kernel_size=(3, 3, 3), activation='relu', padding='same')(dconv3a)
    unpool2 = keras.layers.UpSampling3D(size=(2, 2, 2))(dconv3b)
    cat2 = keras.layers.concatenate([conv2b, unpool2])
    dconv2a = keras.layers.Conv3DTranspose(filters=96, kernel_size=(3, 3, 3), activation='relu', padding='same')(cat2)
    dconv2b = keras.layers.Conv3DTranspose(filters=96, kernel_size=(3, 3, 3), activation='relu', padding='same')(dconv2a)
    unpool1 = keras.layers.UpSampling3D(size=(2, 2, 2))(dconv2b)
    cat1 = keras.layers.concatenate([conv1b, unpool1])
    dconv1a = keras.layers.Conv3DTranspose(filters=96, kernel_size=(3, 3, 3), activation='relu', padding='same')(cat1)
    dconv1b = keras.layers.Conv3DTranspose(filters=96, kernel_size=(3, 3, 3), activation='relu', padding='same')(dconv1a)

    output = keras.layers.Conv3D(filters=3, kernel_size=(3, 3, 3), activation='softmax', padding='same')(dconv1b)

    model = keras.models.Model(input_layer, output)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=[dice_metric, sensitivity1, specificity1])

    a = np.array([[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]])
    b = tf.constant(a)
    c = np.array([[[0, 0, 1], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]])
    d = tf.constant(c)
    print(sensitivity1(b, d))

    arrayData, layerTruth = getData()
    arrayData = np.rot90(arrayData, axes=(1, 4))
    layerTruth = np.rot90(layerTruth, axes=(1, 4))

    print(len(layerTruth))
    print(len(layerTruth[0]))
    print(len(layerTruth[0, 0]))
    print(len(layerTruth[0, 0, 0]))

    print(len(arrayData))


    arrayTruth = convertTruth(layerTruth)
    #arrayDataMult = multichannel(arrayData)
    history = model.fit(arrayData, arrayTruth, epochs=epochs, batch_size=1)

    print(history.history['loss'])
    f = open('Results.txt', 'w')
    f.write('loss\n')
    for x in range(0, len(history.history['loss'])):
        f.write(str(history.history['loss'][x]))
        f.write('\n')
    f.write('Dice Metric\n')
    for x in range(0, len(history.history['dice_metric'])):
        f.write(str(history.history['dice_metric'][x]))
        f.write('\n')
    f.write('Sensitivity\n')
    for x in range(0, len(history.history['sensitivity1'])):
        f.write(str(history.history['sensitivity1'][x]))
        f.write('\n')
    f.write('Specificity\n')
    for x in range(0, len(history.history['specificity1'])):
        f.write(str(history.history['specificity1'][x]))
        f.write('\n')

    model.save(os.path.join(dirnam, "modelsOf5/Model"))


if __name__ == "__main__":
    main()

