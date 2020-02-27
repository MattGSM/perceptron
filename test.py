def sigmoid(inp):
    return 1/(1+2.7182818**(-inp))

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	#return 1.0 if activation >= 0.0 else 0.0
	return sigmoid(activation)

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch={}, lrate={}, error={}'.format(epoch, l_rate, sum_error))
	return weights

# Calculate weights
dataset = [[200, 0.20, 1, 1],
           [210, 0.10, 1, 1],
           [230, 0.05, 0, 1]]
l_rate = 0.1
n_epoch = 100
weights = train_weights(dataset, l_rate, n_epoch)
print('Training Data:', dataset)
print(weights)
