import Network_Define
import numpy as np
import random
import csv

with open('/Volumes/Transcend/GitHub/DeepLearningPractice/spambase/spambase.data', 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)

#Separate data into 2/3 training, and 1/3 testing set.
random.seed(4321)
random.shuffle(data)
train_size=int(2.0*len(data)/3)
train_data = data[:train_size]
test_data = data[train_size:]

training_inputs = [np.reshape(x[:-1], (len(x[:-1]), 1)) for x in train_data]
training_results = [int(y[-1]) for y in train_data]
training_data = zip(training_inputs, training_results)

test_inputs = [np.reshape(x[:-1], (len(x[:-1]), 1)) for x in test_data]
test_results = [y[-1] for y in test_data]
test_data = zip(test_inputs, test_results)

net = Network_Define.Network([57, 5, 3, 1]) # three layers network, with 57 features in inputs

net.SGD(training_data, 30, 10, 0.0005, test_data=test_data) # epochs, mini_batch_size, learning rate