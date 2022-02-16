
import tensorflow as tf
from ImageServer import ImageServer
from models import DAN
import numpy as np

datasetDir = "../data/"
testSet = ImageServer.Load(datasetDir + "commonSet.npz")


def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks

    return y.reshape((nSamples, nLandmarks * 2))

nSamples = testSet.gtLandmarks.shape[0]
imageHeight = testSet.imgSize[0]
imageWidth = testSet.imgSize[1]
nChannels = testSet.imgs.shape[1]

Xtest = testSet.imgs

Ytest = getLabelsForDataset(testSet)

meanImg = testSet.meanImg
stdDevImg = testSet.stdDevImg
initLandmarks = testSet.initLandmarks[0].reshape((-1))

dan = DAN(initLandmarks)


with tf.compat.v1.Session() as sess:
    Saver = tf.compat.v1.train.Saver()
    Writer = tf.compat.v1.summary.FileWriter("logs/", sess.graph)

    Saver.restore(sess,'./Model/Model')
    print('Pre-trained model has been loaded!')
       
    #Landmark68Test(MeanShape,ImageMean,ImageStd,sess)
    errs = []

    for iter in range(10):

        RandomIdx = np.random.choice(Xtest.shape[0], 1, False)

        TestErr = sess.run(dan['S2_Cost'], {dan['InputImage']:(Xtest[RandomIdx]), dan['GroundTruth']: (Ytest[RandomIdx]) ,dan['S1_isTrain']:False, dan['S2_isTrain']:False})

        """result_1=sess.run( {dan['InputImage']: (Xtest[iter])} )
        print(result_1)"""
        #TestErr = sess.run(dan['S2_Cost'],{dan['InputImage']:Xtest,dan['GroundTruth']:Ytest,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
        errs.append(TestErr)
        print('The mean error for image {:d} is: {:f}'.format(iter, TestErr))

    errs = np.array(errs)

    print('The overall mean error is: {:f}'.format(np.mean(errs)))

