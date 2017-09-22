# Logistic Regression on Diabetes Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

def gradient_fun(coeff,data):
    nrow = data.shape[0]
    logit_x = 1.0/(1+np.exp(-np.dot(data, coeff)))
    all_gradient = np.dot((trainY-logit_x),data)
    return (-all_gradient / nrow) #compute mean for each column (parameter)

def normalize(data):
    data_norm = (data - data.mean()) / data.std()
    return data_norm

def coeff_update(coeff,gradient,learn_rate):
    return coeff-gradient*learn_rate


data = pd.read_csv('/Volumes/Transcend/GitHub/DeepLearningPractice/spambase/spambase.data',header=None)

train, test = train_test_split(data, test_size=1.0/3,random_state=42)
# random_state is random number generator

nparam = data.shape[1]-1
trainX = train.ix[:,:(nparam-1)]
trainY = train[57]
testX = test.ix[:,:(nparam-1)]
testY = test[57]

normalization = 1 #0: no normalization; 1: with data normalization

if normalization:
    trainX = normalize(trainX)
    testX = normalize(testX)

# insert a column with value 1s for constant/bias estimation
trainX.insert(0,'Constant',np.ones(trainX.shape[0]))
testX.insert(0,'Constant',np.ones(testX.shape[0]))

coeff = np.zeros(nparam+1) # Initialize coefficient with 0s
learning_rate = 5

for epoch in range(2000):
    print ("round", epoch)
    gradient = gradient_fun(coeff,trainX)
    coeff_new = coeff_update(coeff,gradient,learning_rate)
    coeff_diff = np.absolute(coeff_new-coeff)
    if(np.amax(coeff_diff)<0.00001): #criteria for convergence
        print ("Converged!")
        print (coeff)
        break
    coeff = coeff_new

print (coeff)

predict_train = np.dot(trainX,coeff)
predict_train = 1.0*(predict_train>0.5)
train_error = 1-metrics.accuracy_score(trainY,predict_train)

predict_test = np.dot(testX,coeff)
predict_test = 1.0*(predict_test>0.5)
test_error = 1-metrics.accuracy_score(testY,predict_test)

if normalization:
    print "Data is normalized in experiment."
else:
    print "Data is not normalized in experiment."

print "Train Error:" , train_error
print "Test Error:" ,test_error