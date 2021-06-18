import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

dirnam = os.path.dirname(__file__)

def getData(location):
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
    with open(os.path.normpath(os.path.join(dirnam, location))) as files:
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
    #arrayData, arrayTruth = initialize(imageListTrain, maskListTrain)
    arrayPredictData, arrayPredictTruth = initialize(imageListPredict, maskListPredict)

    return arrayPredictData, arrayPredictTruth


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


def imageGen(labels):
    plt.figure(1)
    labels = labels.astype('int32')
    plt.imshow(labels[80, :, :])
    plt.savefig('visuals2.png')

    img = nib.Nifti1Image(labels, np.eye(4))
    nib.save(img, 'true.nii.gz')


def overlap(labels, truth):
    overlapArray = np.zeros([len(labels), len(labels[0]), len(labels[0][0])])
    for x in range(0, len(labels)):
        for y in range(0, len(labels[1])):
            for z in range(0, len(labels[1][1])):
                if labels[x,y,z] == truth[x,y,z]:
                    overlapArray[x,y,z] = 0
                else:
                    overlapArray[x,y,z] = 1
    plt.figure(1)
    labels = labels.astype('int32')
    plt.imshow(labels[80, :, :])
    plt.savefig('visuals1.png')
    plt.figure(2)
    truth = truth.astype('int32')
    plt.imshow(truth[80, :, :, 0])
    plt.savefig('visuals3.png')

    imageGen(overlapArray)


def total_dice(y_true, y_pred):
    y_trueReshaped = np.zeros([len(y_true), len(y_true[0]), len(y_true[0, 0])])
    for x in range(0, len(y_true)):
        for y in range(0, len(y_true[0])):
            for z in range(0, len(y_true[0, 0])):
                if y_true[x, y, z, 0] < 1000:
                    y_trueReshaped[x, y, z] = y_true[x, y, z, 0]
                else:
                    y_trueReshaped[x, y, z] = y_true[x, y, z, 0] - 1000 + 165

    y_predFiltered = y_pred.astype(int)
    print(len(y_predFiltered))
    y_trueReshaped = y_trueReshaped.astype(int)
    print(len(y_trueReshaped))
    intersect = y_predFiltered == y_trueReshaped
    intersect = intersect.astype(int)
    intersect = intersect.sum()
    totalPixel = len(y_true) * 100 * 148
    hard_dice = 2 * intersect / (totalPixel + totalPixel)
    print(hard_dice)

def dice_metric_label(y_true, y_pred, label):
    y_trueReshaped = np.zeros([len(y_true), len(y_true[0]), len(y_true[0, 0])])
    for x in range(0, len(y_true)):
        for y in range(0, len(y_true[0])):
            for z in range(0, len(y_true[0, 0])):
                if y_true[x, y, z, 0] < 1000:
                    y_trueReshaped[x, y, z] = y_true[x, y, z, 0]
                else:
                    y_trueReshaped[x, y, z] = y_true[x, y, z, 0] - 1000 + 165
                    
    y_predFiltered = y_pred == label
    y_trueFiltered = y_true == label

    y_predFiltered = y_predFiltered.astype(int)
    y_trueReshaped = y_trueReshaped.astype(int)
    intersect = y_predFiltered & y_trueReshaped
    intersect = intersect.astype(int)
    intersect = intersect.sum()
    y_predFiltered = y_predFiltered.sum()
    y_trueReshaped = y_trueReshaped.sum()
    print(y_predFiltered, y_trueReshaped)

    hard_dice = 2 * intersect / (y_predFiltered + y_trueReshaped)

    f = open("DiceScores.txt", "a")
    f.write(str(label) + " " + str(hard_dice) + "\n")
    f.close()

    print(label, hard_dice)


def main():
    arrayData, layerTruth = getData("2.txt")
    results = nib.load(os.path.normpath(os.path.join(dirnam, 'results.nii.gz')))
    resultsData = results.get_fdata()

    arrayData = np.rot90(arrayData, axes=(1, 3))
    layerTruth = np.rot90(layerTruth, axes=(1, 3))

    overlap(resultsData, layerTruth)
    total_dice(layerTruth, resultsData)
    for x in range(165, 333):
        dice_metric_label(layerTruth, resultsData, x)


if __name__ == "__main__":
    main()
