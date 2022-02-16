
from ImageServer import ImageServer
import numpy as np

imageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/"]
boundingBoxFiles = ["../data/py3boxesLFPWTrain.pkl", "../data/py3boxesHelenTrain.pkl", ]
datasetDir = "../data/"
meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]

trainSet = ImageServer(initialization='rect')
trainSet.PrepareData(imageDirs, None, meanShape, 100, 500, True)
trainSet.LoadImages()
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25])
trainSet.NormalizeImages()
trainSet.Save(datasetDir)
validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 50, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)