import numpy as np


class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE


		self.data = np.add(np.matmul(X,self.weights), self.biases)
		return sigmoid(self.data)

		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation

		n = activation_prev.shape[0] # batch size
		
		###############################################
		# TASK 2 - YOUR CODE HERE
		
		#delta : n x self.out_nodes
		#activation_prev: n x self.in_nodes
		#self.data : n x self.out_nodes
		#self.weights : in_nodes x out_nodes

		Delta = np.multiply(delta, derivative_sigmoid(self.data)) # n x self.out_nodes
		weight_updates = np.matmul(np.transpose(activation_prev), Delta) # d(error)/dw_ij (in_nodes x out_nodes)
		temp = np.matmul(Delta, np.transpose(self.weights))
		self.weights = self.weights - np.multiply(weight_updates, lr)
		self.biases = self.biases - lr*np.sum(Delta,0)

		return temp
		#raise NotImplementedError
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE

		self.data = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		for i in range(0,n):
			for j in range(0, self.out_depth):
				weight = self.weights[j]
				for k in range(0, self.in_row-self.filter_row+1 ,self.stride):
					for l in range(0, self.in_col-self.filter_col+1, self.stride):
						patch = X[i,:,k:k+self.filter_row,l:l+self.filter_col]
						self.data[i][j][int(k//self.stride)][l//self.stride] = np.sum(np.multiply(weight,patch)) + self.biases[j]

		return sigmoid(self.data)

		# raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE

		# activation_prev - n x in_depth x in_row x in_col
		# delta - n x out_depth x out_row x out_col
		# weights - out_depth x in_depth x filter_row x filter_col
		# self.data = n x out_depth x out_row x out_col

		Delta = np.multiply(delta, derivative_sigmoid(self.data))
		delta_prev= np.zeros([n,self.in_depth,self.in_row,self.in_col])
		weight_update = np.zeros([self.out_depth, self.in_depth, self.filter_row, self.filter_col])
		bias_update = np.zeros([self.out_depth])
		for i in range(0,n):
			for j in range(0, self.out_depth):
				bias_update[j] += np.sum(Delta[i,j,:,:])
				for k in range(0, self.out_row):
					for l in range(0, self.out_col):
						k1 = k*self.stride
						l1 = l*self.stride
						patch = activation_prev[i,:,k1:k1+self.filter_row,l1:l1+self.filter_col]
						weight_update[j,:,:,:] += np.multiply(patch, Delta[i,j,k,l]) 
						delta_prev[i,:,k1:k1+self.filter_row,l1:l1+self.filter_col] += np.multiply(self.weights[j,:,:,:], Delta[i,j,k,l])
						
		self.weights -= lr*weight_update
		self.biases -= lr*bias_update
		return delta_prev

		# raise NotImplementedError
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for Avg_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE

		self.data = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		for i in range(0,n):
			for j in range(0, self.out_depth):
				for k in range(0, self.in_row-self.filter_row+1 ,self.stride):
					for l in range(0, self.in_col-self.filter_col+1, self.stride):
						patch = X[i,j,k:k+self.filter_row,l:l+self.filter_col]
						self.data[i][j][k//self.stride][l//self.stride] = np.average(patch)

		return self.data
		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE

		Delta = np.multiply(delta, derivative_sigmoid(self.data))
		delta_prev= np.zeros([n,self.in_depth,self.in_row,self.in_col])
		for i in range(0,n):
			for j in range(0, self.out_depth):
				for k in range(0, self.out_row):
					for l in range(0, self.out_col):
						k1 = k*self.stride
						l1 = l*self.stride
						mean_error = Delta[i,j,k,l]/(self.filter_row*self.filter_col)
						delta_prev[i,j,k1:k1+self.filter_row,l1:l1+self.filter_col].fill(mean_error)
		
		return delta_prev

		# raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
