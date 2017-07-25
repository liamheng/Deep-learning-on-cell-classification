'''
This script fine-tunes pre-trained Vgg models to classify two classes of cell
Author: Heng Li
Email: lihengbit@foxmail.com
Date: 07/25/2017
You are free to modify and redistribute the script.
'''
import os
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
import keras
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Input, Dropout, Flatten, Dense
####Read Data
Data=np.load('ExampleDataset.npz')
X=Data['DataSetAugment']
Y=Data['DataSetLabel']
n_classes=np.unique(Y).shape[0]
X=X.reshape(X.shape+(1,))
DataSet=np.concatenate((X,X,X),axis=3)
SubjectNumber=Y.shape[0]
####Data for training
index=[]
for group in np.unique(Y):
	indices=np.random.choice(np.unique(np.where(Y==group)[0]), int(SubjectNumber/n_classes*0.8), replace=False)
	index.append(indices)

index=np.sort(np.array(index).flatten())
testindex=np.setdiff1d(range(SubjectNumber),index)	
TrainX=DataSet[index]
TrainY=Y[index]
TestX=DataSet[testindex]
TestY=Y[testindex]
TrainX=TrainX.reshape(TrainX.shape[0]*TrainX.shape[1],224,224,3)
TestX=TestX.reshape(TestX.shape[0]*TestX.shape[1],224,224,3)
TrainY=keras.utils.to_categorical(TrainY.reshape(TrainY.shape[0]*TrainY.shape[1]))
TestY=keras.utils.to_categorical(TestY.reshape(TestY.shape[0]*TestY.shape[1]))
#####Build model
def VggModel(modelname,fixlayer,FixCNN=True):
	if modelname=='VGG16':
		Entiremodel=VGG16(weights='imagenet')
	elif modelname=='VGG19':
		Entiremodel=VGG19(weights='imagenet')
	if FixCNN:
		for layer in Entiremodel.layers[:-3]:
			layer.trainable = False
	if fixlayer=='FC2':
		Entiremodel.layers[-3].trainable = False
	elif fixlayer=='Softmax':	
		Entiremodel.layers[-3].trainable = False
		Entiremodel.layers[-4].trainable = False
	output=Dense(n_classes, activation='softmax', name='predictions',input_shape=Entiremodel.layers[-2].output_shape[1:])(Entiremodel.layers[-2].output)
	model = Model(input= Entiremodel.input, output= output)
	return model
####Train model	
modelname='VGG16'
fixlayer='FC2'
model=VggModel(modelname,fixlayer)
model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto')
saveBestModel=keras.callbacks.ModelCheckpoint(modelname+fixlayer+'_best_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')	
history=model.fit(TrainX, TrainY,epochs=200,batch_size=100,verbose=1,callbacks=[earlyStopping,saveBestModel],validation_data=(TestX, TestY))
model.save(modelname+fixlayer+'_weights.hdf5')
loss_history = np.array(history.history["loss"])
val_loss_history = np.array(history.history["val_loss"])		
acc_history = np.array(history.history["acc"])
val_acc_history = np.array(history.history["val_acc"])
np.savetxt(modelname+fixlayer+"_history.txt", [loss_history,val_loss_history,acc_history,val_acc_history], delimiter=",")

		
