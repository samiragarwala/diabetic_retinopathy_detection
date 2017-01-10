import numpy as np
import sys

def read_inputs() :
	feature_vec_train=np.load('train_feature.npy')
	#print(feature_vec_train.shape)
	train_label=np.load('train_label.npy')
	#print(train_label.shape)
	feature_vec_test=np.load('test_feature.npy')
	#print(feature_vec_test.shape)
	test_label=np.load('test_label.npy')
	#print(test_label.shape)
	return feature_vec_train,train_label,feature_vec_test,test_label

def regularised_regression(feature_vec_train,train_label,feature_vec_test,test_label) :
	lamda=0.2
	K=np.vstack((feature_vec_train,lamda*np.identity(19)))
	K=K.astype(float)
	zeros=np.zeros((19,1), dtype=int, order='C')
	d=(np.vstack((train_label,zeros))).astype(float)
	x_1=np.dot(np.transpose(K),K)
	x_1=np.linalg.inv(x_1)
	x_2=np.dot(np.transpose(K),d)
	x=np.dot(x_1,x_2)
	prediction_train=np.matrix.round(np.dot(feature_vec_train.astype(float),x))
	prediction_train[prediction_train>=1] = 1
	prediction_train[prediction_train<=0] = 0
	return x, prediction_train

def regression(feature_vec_train,train_label,feature_vec_test,test_label) :
	feature_vec_train=feature_vec_train.astype(float)
	train_label=train_label.astype(float)
	feature_vec_test=feature_vec_test.astype(float)
	test_label=test_label.astype(float)
	x_1=np.dot(np.transpose(feature_vec_train),feature_vec_train)
	x_1=np.linalg.inv(x_1)
	x_2=np.dot(np.transpose(feature_vec_train),train_label)
	x=np.dot(x_1,x_2)
	prediction_train=np.matrix.round(np.dot(feature_vec_train,x)) 
	train_accuracy(prediction_train,train_label)
	test(feature_vec_test,test_label,x)


def train_accuracy(train_predictions,train_label) :
	count=0.0
	train_label=train_label.astype(float)
	train_predictions=train_predictions.astype(float)
	for i in range (train_label.size):
		if train_predictions[i,0]==train_label[i,0] :
			count+=1.0

	accuracy=(count/train_label.size)
	print("Train Accuracy for Regression is: "+str(accuracy))

def test(feature_vec_test,test_label,x) :
	prediction_test=np.dot(feature_vec_test.astype(float),x.astype(float))
	count=0.0
	test_label=test_label.astype(float)
	prediction_test=np.matrix.round(prediction_test.astype(float))
	prediction_test[prediction_test>=1] = 1
	prediction_test[prediction_test<=0] = 0
	for i in range (0,test_label.size):
		if prediction_test[i,0]==test_label[i,0] :
			count+=1.0
	accuracy=(count/test_label.size)
	print("Test Accuracy for Regression is: "+str(accuracy))

def polynomial_regression(feature_vec_train,train_label,feature_vec_test,test_label) :
	power=3
	for i in range (2,power+1):
		train_matrix= np.power(np.array(feature_vec_train,dtype=np.float64),i)
		train_matrix=np.hstack((np.array(feature_vec_train,dtype=np.float64),train_matrix))
		feature_vec_train=train_matrix
		test_matrix=np.power(np.array(feature_vec_test,dtype=np.float64),i)
		test_matrix=np.hstack((np.array(feature_vec_test,dtype=np.float64),test_matrix))
		feature_vec_test=test_matrix

	x_1=np.linalg.pinv(train_matrix)
	x=np.dot(x_1,np.array(train_label,dtype=np.float64))
	prediction_train=np.matrix.round(np.dot(train_matrix,x))
	train_accuracy(prediction_train,train_label)
	test(test_matrix,test_label,x)


if __name__ == "__main__":
	feature_vec_train,train_label,feature_vec_test,test_label = read_inputs()
	print('\033[1m'+"\nRegression "+'\033[0m')
	regression(feature_vec_train,train_label,feature_vec_test,test_label)
	print('\033[1m'+"\nRegularised Regression"+'\033[0m')
	weights, trainPredictions = regularised_regression(feature_vec_train,train_label,feature_vec_test,test_label)
	train_accuracy(trainPredictions,train_label)
	test(feature_vec_test,test_label,weights)
	print('\033[1m'+"\nPolynomial Regression "+'\033[0m')
	polynomial_regression(feature_vec_train,train_label,feature_vec_test,test_label)