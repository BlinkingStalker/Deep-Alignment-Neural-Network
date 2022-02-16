
from ImageServer import ImageServer
import numpy as np

commonSetImageDirs = ["../data/images/lfpw/testset/", "../data/images/helen/testset/"]
commonSetBoundingBoxFiles = ["../data/py3boxesLFPWTest.pkl", "../data/py3boxesHelenTest.pkl"]
datasetDir = "../data/"
meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]

commonSet = ImageServer(initialization='box')
commonSet.PrepareData(commonSetImageDirs, commonSetBoundingBoxFiles, meanShape, 0, 554, False)
commonSet.LoadImages()
commonSet.CropResizeRotateAll()
commonSet.imgs = commonSet.imgs.astype(np.float32)
commonSet.NormalizeImages()
commonSet.Save(datasetDir, "commonSet.npz")

"""challengingSet = ImageServer(initialization='box')
challengingSet.PrepareData(challengingSetImageDirs, challengingSetBoundingBoxFiles, meanShape, 0, 1000, False)
challengingSet.LoadImages()
challengingSet.CropResizeRotateAll()
challengingSet.imgs = challengingSet.imgs.astype(np.float32)
challengingSet.Save(datasetDir, "challengingSet.npz")"""

"""w300Set = ImageServer(initialization='box')
w300Set.PrepareData(w300SetImageDirs, w300SetBoundingBoxFiles, meanShape, 0, 1000, False)
w300Set.LoadImages()
w300Set.CropResizeRotateAll()
w300Set.imgs = w300Set.imgs.astype(np.float32)
w300Set.Save(datasetDir, "w300Set.npz")"""