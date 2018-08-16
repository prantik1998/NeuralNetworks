import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)#loading the data


def sigmoid(x):
	
	return 1/(1 + np.exp(-x)) 
def sigmoid_derivative(x):
	return sigmoid(x)*(1 - sigmoid(x))
	

class Function:#class NeuralNetworks containing 2 layers
	def __init__(self):
		self.weights=np.random.randn(784,100).astype(np.float64)/100 # 784, 100
		self.biases=np.random.randn(1,100).astype(np.float64)/100	# 100
		self.weights2=np.random.randn(100,10).astype(np.float64)/100	# 100, 10
		self.biases2=np.random.randn(1,10).astype(np.float64)/100	#10
		self.batch_size=1000 #batch_size
		self.lr =4e-3 #learning_rate 
		self.earlier_weights = 0
		
	def test(self,x,y):
		
		input_=sigmoid(np.matmul(x,self.weights)+self.biases)#input for the next layer
		
		output=sigmoid(np.matmul(input_,self.weights2)+self.biases2)#the output matrix
		print('\n---------------------------------------------\n')
		print('LR:', self.lr)
		print('Loss: ', np.mean(np.square(output - y)))
		print('Accuracy: ', np.mean((np.argmax(output, axis=1) == np.argmax(y, axis=1)).astype(np.float32)))

		print('\n---------------------------------------------\n')

	
	def stochastic_gradient_descent(self,x,y, i):

		x=np.array(x)
		output_=sigmoid(np.matmul(x,self.weights)+self.biases)#1000,100		
		output2=sigmoid(np.matmul(output_,self.weights2)+self.biases2)#1000,10
		output_derivative=sigmoid_derivative(np.matmul(x,self.weights)+self.biases)#1000,100
		output2_derivative=sigmoid_derivative(np.matmul(output_,self.weights2)+self.biases2)#1000,10
		s=np.multiply(output2-y,output2_derivative)/x.shape[0]#1000,100
		lwb2 = np.sum(s,axis=0)#derivative  of the loss function with respect to bias2 
		
		t=np.sum(output_,axis=0)
		lww2 = np.multiply(np.repeat(t.reshape([100, 1]), 10, axis=1), lwb2)#derivative of the loss function with respect to weights1
		s2=np.multiply(np.matmul(lwb2,self.weights2.transpose()),output_derivative)
		lwb1 = np.sum(s2,axis=0)#derivative of the loss function with respect to bias1
		t2=np.sum(x,axis=0)
		lww1=np.multiply(np.repeat(t2.reshape([784,1]),100,axis=1),lwb1)#derivative of the loss function with respect to weights1
		
		self.weights-=self.lr*lww1 #updating weights of layer1
		self.biases-=self.lr*lwb1 #updating biases oflayer1
		self.weights2-=self.lr*lww2#updating weights of layer2
		self.biases2-=self.lr*lwb2#updating biases of layer2
		

		
		
x = Function()
sum_=0
for epoch in range(100000):
		idx = np.random.choice(mnist.train.images.shape[0], 100)
		
		x.stochastic_gradient_descent(mnist.train.images[idx],mnist.train.labels[idx], epoch)	
		if epoch%1000==0 and epoch!=0:
			x.test(mnist.test.images,mnist.test.labels)








		
