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

    alt = 0
    imageList = []
    maskList = []
    with open(os.path.normpath(os.path.join(dirnam, '2.txt'))) as files:
        for arrayDataPath in files:
            if alt == 0:
                imageList.append(arrayDataPath.strip())
                alt = 1
            elif alt == 1:
                maskList.append(arrayDataPath.strip())
                alt = 0
    slices = 0
    #print(os.path.join(dirnam, imageList[0]))
    for x in range(0, len(imageList)):
        image = nib.load(os.path.normpath(os.path.join(dirnam, imageList[x])))
        data = image.get_fdata()
        slices = slices + len(data[0])

    arrayData = np.empty([slices, 1, 148, 100])
    arrayTruth = np.empty([slices, 1, 148, 100])
    #print(len(arrayData[0]))
    start = 0
    for file in range(0, len(imageList)):
        image = nib.load(os.path.normpath(os.path.join(dirnam, imageList[file])))
        data = image.get_fdata()
        mask = nib.load(os.path.normpath(os.path.join(dirnam, maskList[file])))
        truth = mask.get_fdata()
        data = np.rot90(data, axes=(0, 1))
        truth = np.rot90(truth, axes=(0, 1))
        for x in range(0, len(data)):
                arrayData[x + start, 0] = data[x, 0:148]
                arrayTruth[x + start, 0] = truth[x, 0:148]
        start = start + len(data)
        #print("File Loaded")
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


def createModels(neighbors, mask):
    #To loop through every label
    for x in range(0, 332):
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
            for a in range(0, len(mask)):
                for b in range(0, len(mask[0])):
                    for c in range(0, len(mask[0, 0]) - 1):
                        #If statement to convert from image values to consecutive values
                        if x < 166:
                            if mask[a, b, c] == x:
                                #If statements to check if we are on the edge of the image and account for that
                                if a < 4:
                                    if b < 4:
                                        data[pixel] = mask[0:8, 0:8, c]
                                    elif b > 96:
                                        data[pixel] = mask[0:8, 92:100, c]
                                    else:
                                        data[pixel] = mask[0:8, b-4:b+4, c]
                                elif a > 96:
                                    if b < 4:
                                        data[pixel] = mask[92:100, 0:8, c]
                                    elif b > 96:
                                        data[pixel] = mask[92:100, 92:100, c]
                                    else:
                                        data[pixel] = mask[92:100, b-4:b+4, c]
                                else:
                                    if b < 4:
                                        data[pixel] = mask[a-4:a+4, 0:8, c]
                                    elif b > 96:
                                        data[pixel] = mask[a-4:a+4, 92:100, c]
                                    else:
                                        data[pixel] = mask[a-4:a+4, b-4:b+4, c]
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
                                        data[pixel] = mask[0:8, 0:8, c]
                                    elif b > 96:
                                        data[pixel] = mask[0:8, 92:100, c]
                                    else:
                                        data[pixel] = mask[0:8, b-4:b+4, c]
                                elif a > 96:
                                    if b < 4:
                                        data[pixel] = mask[92:100, 0:8, c]
                                    elif b > 96:
                                        data[pixel] = mask[92:100, 92:100, c]
                                    else:
                                        data[pixel] = mask[92:100, b-4:b+4, c]
                                else:
                                    if b < 4:
                                        data[pixel] = mask[a-4:a+4, 0:8, c]
                                    elif b > 96:
                                        data[pixel] = mask[a-4:a+4, 92:100, c]
                                    else:
                                        data[pixel] = mask[a-4:a+4, b-4:b+4, c]
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


            history = modelN.fit(data, truth, epochs=1, batch_size=50)
            modelN.save(os.path.join(dirnam, "models/" + str(x) + "modelN"))


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
    for x in range(10, len(layerTruth), 10):
        arrayTruth = convertTruth(layerTruth[x - 10:x])

        #neighbors = detectNeighbors(layerTruth[x:x + 30])

        #createModels(neighbors, layerTruth)

        #print(neighbors)
        print(len(arrayData))

        history = model.fit(arrayData[x - 10:x], arrayTruth, epochs=epochs, batch_size=10)

    model.save(os.path.join(dirnam, "modelsGlobal/Model"))


if __name__ == "__main__":
    main()
