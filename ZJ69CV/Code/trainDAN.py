
import tensorflow as tf
import numpy as np
if tf.__version__.startswith('1.'):  # tensorflow 1
    config = tf.compat.v1.ConfigProto()  # allow_soft_placement=True
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
else:
    # tensorflow 2
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

"""import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config = config)"""

from ImageServer import ImageServer
from models import DAN
#0.2, 0.2, 20, 0.25
datasetDir = "../data/"
trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=54220_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")

# Create a new DAN regressor.
# 两个stage训练，不宜使用estimator
# regressor = tf.estimator.Estimator(model_fn=DAN,
# 	                                params={})
def evaluateError(landmarkGt, landmarkP):
    e = np.zeros(68)
    ocular_dist = np.mean(np.linalg.norm(landmarkGt[36:42] - landmarkGt[42:48], axis=1))
    for i in range(68):
        e[i] = np.linalg.norm(landmarkGt[i] - landmarkP[i])
    e = e / ocular_dist
    return e

def evaluateBatchError(landmarkGt, landmarkP, batch_size):
    e = np.zeros([batch_size, 68])
    for i in range(batch_size):
        e[i] = evaluateError(landmarkGt[i], landmarkP[i])
    mean_err = e[:,:].mean()#axis=0)
    return mean_err


def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks

    return y.reshape((nSamples, nLandmarks * 2))

nSamples = trainSet.gtLandmarks.shape[0]
imageHeight = trainSet.imgSize[0]
imageWidth = trainSet.imgSize[1]
nChannels = trainSet.imgs.shape[3]

Xtrain = trainSet.imgs
Xvalid = validationSet.imgs
# import pdb; pdb.set_trace()

Ytrain = getLabelsForDataset(trainSet)
Yvalid = getLabelsForDataset(validationSet)
testIdxsTrainSet = range(len(Xvalid))
testIdxsValidSet = range(len(Xvalid))
meanImg = trainSet.meanImg
stdDevImg = trainSet.stdDevImg
initLandmarks = trainSet.initLandmarks[0].reshape((1, 136))

dan = DAN(initLandmarks)


STAGE = 1



with tf.compat.v1.Session() as sess:
    Saver = tf.compat.v1.train.Saver()
    Writer = tf.compat.v1.summary.FileWriter("logs/", sess.graph)
    if STAGE < 2:
        sess.run(tf.compat.v1.global_variables_initializer())

    else:
        Saver.restore(sess,'./Model/Model')
        print('Pre-trained model has been loaded!')
       
    # Landmark68Test(MeanShape,ImageMean,ImageStd,sess)
    print("Starting training......")
    for epoch in range(2):
        Count = 0
        while Count * 2 < Xtrain.shape[0]:
            RandomIdx = np.random.choice(Xtrain.shape[0],2,False)
            if STAGE == 1 or STAGE == 0:
                # sess.run(dan['S1_Optimizer'], feed_dict={dan['InputImage']:Xtrain[RandomIdx],\
                #     dan['GroundTruth']:Ytrain[RandomIdx],dan['S1_isTrain']:True,dan['S2_isTrain']:False})
                sess.run(dan['S1_Optimizer'], feed_dict={dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,dan['S1_isTrain']:True,dan['S2_isTrain']:False})
            else:
                sess.run(dan['S2_Optimizer'], feed_dict={dan['InputImage']:Xtrain[RandomIdx],dan['GroundTruth']:Ytrain[RandomIdx],dan['S1_isTrain']:False,dan['S2_isTrain']:True})

            if Count % 1000 == 0:
                TestErr = 0
                BatchErr = 0

                if STAGE == 1 or STAGE == 0:
                    TestErr = sess.run(dan['S1_Cost'], {dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    # print(evaluateBatchError(Yvalid.reshape([-1, 68, 2]), S1_Ret.reshape([-1, 68, 2]), 9))
                    BatchErr = sess.run(dan['S1_Cost'],{dan['InputImage']:Xtrain[RandomIdx],dan['GroundTruth']:Ytrain[RandomIdx],dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                else:
                    """Landmark,Img,HeatMap,FeatureUpScale =dan['S2_InputLandmark'],dan['S2_InputImage'],dan['S2_InputHeatmap'],dan['S2_FeatureUpScale']
                    #sess.run([dan['S2_InputLandmark'],dan['S2_InputImage'],dan['S2_InputHeatmap'],dan['S2_FeatureUpScale']],{Feed_dict['InputImage']:I[RandomIdx],Feed_dict['GroundTruth']:G[RandomIdx],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})
                    import cv2
                    for i in range(64):
                        TestImage = np.zeros([112,112,1])
                        for p in range(68):
                            cv2.circle(TestImage,(int(Landmark[i][p *2]),int(Landmark[i][p * 2 + 1])),1,(255),-1)

                        cv2.imshow('Landmark',TestImage)
                        cv2.imshow('Image',Img[i])
                        cv2.imshow('HeatMap',HeatMap[i])
                        cv2.imshow('FeatureUpScale',FeatureUpScale[i])
                        cv2.waitKey(-1)"""
                    TestErr = sess.run(dan['S2_Cost'],{dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    BatchErr = sess.run(dan['S2_Cost'],{dan['InputImage']:Xtrain[RandomIdx],dan['GroundTruth']:Ytrain[RandomIdx],dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                print('Epoch: ', epoch, ' Batch: ', Count, 'TestErr:', TestErr, ' BatchErr:', BatchErr)
            Count += 1
        Saver.save(sess,'./Model/Model')

