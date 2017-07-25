'''
This script extracts deep features of cell sequence with pre-trained Vgg models. Then SVM and XGboost are performed to classify two classes of cell.
Author: Heng Li
Email: lihengbit@foxmail.com
Date: 07/25/2017
You are free to modify and redistribute the script.
'''
import os
import numpy as np
import scipy.io as sio
import xgboost as xgb
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support as score
import keras
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Input, Dropout, Flatten, Dense
####Read Data
Data=np.load('ExampleDataset.npz')
X=Data['DataSetAugment'][:,0]
Y=Data['DataSetLabel'][:,0]
X=X.reshape(X.shape+(1,))
DataSet=np.concatenate((X,X,X),axis=3)
SubjectNumber=Y.shape[0]
####Index for training and testing
index=[]
for group in np.unique(Y):
	indices=np.random.choice(np.unique(np.where(Y==group)[0]), int(SubjectNumber/n_classes*0.8), replace=False)
	index.append(indices)

index=np.sort(np.array(index).flatten())
testindex=np.setdiff1d(range(SubjectNumber),index)	

#####Build model
def VggModel(modelname):
	if modelname=='VGG16':
		Entiremodel=VGG16(weights='imagenet')
	elif modelname=='VGG19':
		Entiremodel=VGG19(weights='imagenet')
	if FixCNN:
		for layer in Entiremodel.layers[:-3]:
			layer.trainable = False
	output=Dense(n_classes, activation='softmax', name='predictions',input_shape=Entiremodel.layers[-2].output_shape[1:])(Entiremodel.layers[-2].output)
	model = Model(input= Entiremodel.input, output=[Entiremodel.layers[-3].output,Entiremodel.layers[-2].output])
	return model

####Get features	
modelname='VGG16'
model=VggModel(modelname)
FC1Feature,FC2Feature = model.predict(DataSet)
np.savetxt(modelname+'FC1Feature.txt',np.concatenate((Y,FC1Feature),axis=1))
np.savetxt(modelname+'FC2Feature.txt',np.concatenate((Y,FC2Feature),axis=1))

####Classification features
TrainX=FC1Feature[index]
TrainY=Y[index]
TestX=FC1Feature[testindex]
TestY=Y[testindex]

##Evaluate result
def result(real,pred):
	precision, recall, fscore, support = score(real,pred)
	Pr=np.mean(precision)
	Re=np.mean(recall)
	F1=np.mean(fscore)
	return (Pr,Re,F1)

##SVM
SVM = svm.SVC()
SVM.fit(TrainX, TrainY) 
SVMTrainResult=result(TrainY,SVM.predict(TrainX))
SVMTestResult=result(TestY,SVM.predict(TestX))

##XGboost
xg_train = xgb.DMatrix(TrainX, label=TrainY)
xg_test = xgb.DMatrix(TestX, label=TestY)
param = {'max_depth':3,
	   'learning_rate':0.05,
	   'n_estimators':100,
	   'min_child_weight':1,
	   'subsample':1,
	   'colsample_bytree':1,
	   'objective':'binary:logistic'}

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round=500
bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50)
predtrain=bst.predict(xg_train)
predtest=bst.predict(xg_test)	
predtrain=np.int_(predtrain>0.5)
predtest=np.int_(predtest>0.5)
XGTrainResult=result(TrainY,predtrain)
XGTestResult=result(TestY,predtest)






