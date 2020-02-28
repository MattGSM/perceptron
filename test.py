import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inp):
    return 1/(1+2.718281828846**(-inp))

# Make a prediction with weights
def predict(row, weights):
    #activation is the first weight
	activation = weights[0]

    #multiply the weights by the rows and add to the activation
	for i in range(len(row)-1):
		activation += weights[i+1] * row[i]

	return sigmoid(activation)

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):

    #make a new list of weights
    weights = [0.0 for i in range(len(train[0]))]

    #start training
    error_sums = []
    for epoch in range(n_epoch):

        #set the sum error to 0
        #row_count = 0
        sum_error = 0.0
        for row in train:

            #make a prediction using the one row and the weights
            prediction = predict(row, weights)

            #error is the expected - predicted
            error = row[-1] - prediction

            #sum error += the (expected - predicted) squared
            sum_error += error**2

            #new first weight is previous first weight + learning rate * (expected - predicted value)
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] + l_rate * error * row[i]
            #weights_history[row_count] = weights[0]
            #row_count+=1
        #print('>epoch={}, lrate={}, error={}, weights={}'.format(epoch, l_rate, sum_error, weights))
        error_sums.append(sum_error)
    #print('Training Data:', dataset)
    # print(weights)
    plt.plot(error_sums)
    # plt.ylabel("ahhh")
    # plt.show()
    return weights

file = open('titanic dataset.txt', 'r')
lines = file.readlines()
dataset = []
for line in lines[1:]:
    items = line[:-1].split(',')
    items.pop(2)
    if items[2] == 'male':
        items[2] = '0'
    else:
        items[2] = '1'
    for item in range(len(items)):
        items[item] = float(items[item])
    items.append(items.pop(0))
    dataset.append(items)

l_rate = 0.0001
n_epoch = 1000
weights = train_weights(dataset, l_rate, n_epoch)

testingSet = dataset[:-10]
for epoch in range(len(testingSet)):
    actualPrediction = predict(testingSet[epoch], weights)
    prediction = 1.0 if actualPrediction >= 0.5 else 0.0

    #print('With data: {}, Weights {}, Prediction {}'.format(data, weights, prediction))
    if testingSet[epoch][-1]==prediction:
        print('Epoch: {} correct'.format(epoch))
    else:
        print('Epoch: {} incorrect. Predicted: {} expected: {}'.format(epoch, actualPrediction, testingSet[epoch][-1]))
    print('Correct: ' + ('Yes' if testingSet[epoch][-1]==prediction else 'No'))
