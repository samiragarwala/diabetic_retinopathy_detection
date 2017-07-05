import numpy as np
import sys
import math
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import decimal
from decimal import Decimal,localcontext
from decimal import getcontext
from sklearn.metrics import f1_score

def read_inputs() :

	
	feature_vec_train=np.load('train_feature.npy').astype(np.float)
	#print(feature_vec_train.shape)
	train_label=np.load('train_label.npy').astype(np.float)
	#print(train_label.shape)
	feature_vec_test=np.load('test_feature.npy').astype(np.float)
	#print(feature_vec_test.shape)
	test_label=np.load('test_label.npy').astype(np.float)

	print(feature_vec_train[0,:])
	print(train_label[0,:])
	#print(test_label.shape)
	return feature_vec_train,train_label,feature_vec_test,test_label

def accuracy(predictions,label,rows):

	predictions = np.rint(predictions)
	predictions[predictions>=1] = 1.0
	predictions[predictions<=0] = 0.0

	return accuracy_score(label,predictions)

def regression(feature_vec_train,train_label,feature_vec_test,test_label):
	reg = linear_model.LinearRegression()
	reg.fit(feature_vec_train,train_label)

	train_predictions = reg.predict(feature_vec_train)
	train_predictions.reshape(920,1)
	train_label.reshape(920,1)
	print("Train Accuracy: ")
	train_accuracy = accuracy(train_predictions,train_label,920)
	print(train_accuracy)

	test_predictions = reg.predict(feature_vec_test)
	test_predictions
	test_label

	print("Test Accuracy: ")
	test_accuracy = accuracy(test_predictions,test_label,231)
	print(test_accuracy)

def regularised_regression(feature_vec_train,train_label,feature_vec_test,test_label):
	reg = linear_model.Ridge(alpha = 1.25)
	reg.fit(feature_vec_train,train_label)

	train_predictions = reg.predict(feature_vec_train)
	train_predictions.reshape(920,1)
	train_label.reshape(920,1)
	train_accuracy = accuracy(train_predictions,train_label,920)
	print("Train Accuracy: ")
	print(train_accuracy)

	test_predictions = reg.predict(feature_vec_test)
	test_predictions
	test_label
	test_accuracy = accuracy(test_predictions,test_label,231)

	print("Test Accuracy: ")
	print(test_accuracy)

def exponential_regression(feature_vec_train,train_label,feature_vec_test,test_label,a) :
	exponent = a
	feature_vec_train = np.array(feature_vec_train, dtype=np.float128)
	feature_vec_test = np.array(feature_vec_test, dtype=np.float128)
	for x in np.nditer(feature_vec_train, op_flags=['readwrite']):
		x[...] = exponent ** x
	for y in np.nditer(feature_vec_test, op_flags=['readwrite']):
		y[...] = exponent ** y

	regression(feature_vec_train,train_label,feature_vec_test,test_label)

def logarithmic_regression(feature_vec_train,train_label,feature_vec_test,test_label) :
	
	base = 1000
	feature_vec_train =np.array(feature_vec_train, dtype=np.float64)
	feature_vec_test = np.array(feature_vec_test, dtype=np.float64)
	for x in np.nditer(feature_vec_train, op_flags=['readwrite']):
		if(x>0):
			x[...] = math.log(x,base)
	for y in np.nditer(feature_vec_test, op_flags=['readwrite']):
		if(y>0):
			y[...] = math.log(y,base)
	regression(feature_vec_train,train_label,feature_vec_test,test_label)

def polynomial_reg(feature_vec_train,train_label,feature_vec_test,test_label) :
	power=3
	getcontext().prec = 6
	# for i in range (0,920):
	# 	for j in range(0,18):
	# 		feature_vec_train[i][j] = decimal.Decimal(feature_vec_train[i][j])
	# for i in range (0,231):
	# 	for j in range(0,18):
	# 		feature_vec_test[i][j] = decimal.Decimal(feature_vec_test[i][j])
	# feature_vec_train.astype(decimal.Decimal)
	# feature_vec_test.astype(decimal.Decimal)
	for i in range (2,power+1):
		train_matrix= np.power(np.array(feature_vec_train,dtype=np.dtype(decimal.Decimal)),i)
		train_matrix=np.hstack((np.array(feature_vec_train,dtype=np.dtype(decimal.Decimal)),train_matrix))
		feature_vec_train=train_matrix
	# 	# for i in range (0,920):
	# 	# 	for j in range(0,18):
	# 	# 		feature_vec_train[i][j] = decimal.Decimal(feature_vec_train[i][j])
		test_matrix=np.power(np.array(feature_vec_test,dtype=np.dtype(decimal.Decimal)),i)
		test_matrix=np.hstack((np.array(feature_vec_test,dtype=np.dtype(decimal.Decimal)),test_matrix))
		feature_vec_test=test_matrix
	# 	# for i in range (0,231):
	# 	# 	for j in range(0,18):
	# 	# 		feature_vec_test[i][j] = decimal.Decimal(feature_vec_test[i][j])
	regression(feature_vec_train,train_label,feature_vec_test,test_label)



if __name__ == "__main__":
	# read_inputs()
	feature_vec_train,train_label,feature_vec_test,test_label = read_inputs()
	print(feature_vec_test.shape)
	print("Linear Regression")
	regression(feature_vec_train,train_label,feature_vec_test,test_label)

	print("Regularised Linear Regression")
	regularised_regression(feature_vec_train,train_label,feature_vec_test,test_label)

	print("Exponential Regression")
	exponential_regression(feature_vec_train,train_label,feature_vec_test,test_label,-0.2)

	print("Logarithmic Regression")
	logarithmic_regression(feature_vec_train,train_label,feature_vec_test,test_label)

	print("Polynomial Regression 1")
	polynomial_reg(feature_vec_train,train_label,feature_vec_test,test_label)



	print("Polynomial Regression")
	poly = PolynomialFeatures(degree=6)
	feature_vec_train = poly.fit_transform(feature_vec_train)
	feature_vec_test = poly.fit_transform(feature_vec_test)
	regression(feature_vec_train,train_label,feature_vec_test,test_label)



	



