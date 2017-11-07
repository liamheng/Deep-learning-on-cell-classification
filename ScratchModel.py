'''
This script trains a model from scratch to classify two classes of cell
Author: Heng Li
Email: lihengbit@foxmail.com
Date: 07/24/2017
You are free to modify and redistribute the script.
'''
import os
import numpy as np
import scipy.io as sio
import keras
from keras import optimizers
from keras.models import Model,Sequential,model_from_json
from keras.layers import Flatten,Dense,Input,Dropout
from keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,ZeroPadding2D
from keras.utils import plot_model
from keras.models import load_model
####Read Data
Data=np.load('ExampleDataset.npz')
X=Data['DataSetAugment']
Y=Data['DataSetLabel']
n_classes=np.unique(Y).shape[0]
X=X.reshape(X.shape+(1,))
DataSet=X
SubjectNumber=Y.shape[0]
####Data for training
#np.random.seed(0)
index=[]
for group in np.unique(Y):
	indices=np.random.choice(np.unique(np.where(Y==group)[0]), int(SubjectNumber/n_classes*0.8), replace=False)
	index.append(indices)
		
index=np.sort(np.array(index).flatten())
testindex=np.setdiff1d(range(SubjectNumber),index)	
TrainX=DataSet[index]
TrainY=Y[index]
TestX=DataSet[testindex,0]
TestY=Y[testindex,0]
TrainX=TrainX.reshape(TrainX.shape[0]*TrainX.shape[1],224,224,1)
TrainY=keras.utils.to_categorical(TrainY.reshape(TrainY.shape[0]*TrainY.shape[1]))
TestY=keras.utils.to_categorical(TestY)

#####Build model
def get_model(summary=False):
	""" Return the Keras model of the network
	"""
	model = Sequential()
	input_shape=(224,224,1) # c, l, h, w
	# 1st layer group
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv1_1',
							input_shape=input_shape))
	# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
							# border_mode='same', name='conv1_2'))							
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool1'))
	# 2nd layer group
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv2_1'))
	# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
							# border_mode='same', name='conv2_2'))
	# model.add(Conv3D(128, 3, 3, 3, activation='relu',
							# border_mode='same', name='conv2_3'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool2'))
	# 3rd layer group
	model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv3_1'))
	# model.add(Conv3D(256, 3, 3, 3, activation='relu',
							# border_mode='same', name='conv3_2'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool3'))
	# 4rd layer group
	model.add(Conv2D(512, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv4_1'))
	# model.add(Conv3D(256, 3, 3, 3, activation='relu',
							# border_mode='same', name='conv3_2'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool4'))	
	# 5rd layer group
	model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv5_1'))
	# model.add(Conv3D(256, 3, 3, 3, activation='relu',
							# border_mode='same', name='conv3_2'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool5'))	
	model.add(Flatten())
	# FC layers group
	model.add(Dense(1024, activation='relu', name='fc1'))
	model.add(Dropout(.3))
	model.add(Dense(512, activation='relu', name='fc2'))
	model.add(Dropout(.3))
	model.add(Dense(2, activation='softmax', name='fc3'))
	return model
####Train model		
model = get_model()	
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto')
saveBestModel=keras.callbacks.ModelCheckpoint('./ScratchModel_best_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')	
history=model.fit(TrainX, TrainY,epochs=200,batch_size=100,verbose=1,callbacks=[saveBestModel,earlyStopping],validation_data=(TestX, TestY))
model.save('./ScratchModel_weights.hdf5')	
loss_history = np.array(history.history["loss"])
val_loss_history = np.array(history.history["val_loss"])
acc_history = np.array(history.history["acc"])
val_acc_history = np.array(history.history["val_acc"])
np.savetxt('./ScratchModel_history.txt', [loss_history,val_loss_history,acc_history,val_acc_history], delimiter=",")

