import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import matplotlib.pyplot as plt

def sigmoid(x):
	# print(x)
	# return x
	return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
	return sigmoid(x)*(1 - sigmoid(x))
	# return 1

class Function:
	def __init__(self):
		self.weights=np.random.randn(784,100).astype(np.float64)/100 # 784, 100
		self.biases=np.random.randn(1,100).astype(np.float64)/100	# 100
		self.weights2=np.random.randn(100,10).astype(np.float64)/100	# 100, 10
		self.biases2=np.random.randn(1,10).astype(np.float64)/100	#10
		self.batch_size=1000
		self.lr =3e-3
		self.earlier_weights = 0
		
	def test(self,x,y):
		
		input_=sigmoid(np.matmul(x,self.weights)+self.biases)#input_
		
		output=sigmoid(np.matmul(input_,self.weights2)+self.biases2)
		print('\n---------------------------------------------\n')
		print('LR:', self.lr)
		print('Loss: ', np.mean(np.square(output - y)))
		print('Accuracy: ', np.mean((np.argmax(output, axis=1) == np.argmax(y, axis=1)).astype(np.float32)))

		print('\n---------------------------------------------\n')

	def show(self, x):
		plt.imshow(x.reshape([28, 28]))
		plt.show()
	
	def stochastic_gradient_descent(self,x,y, i):

		x=np.array(x)
		output_=sigmoid(np.matmul(x,self.weights)+self.biases)#55000,100		
		output2=sigmoid(np.matmul(output_,self.weights2)+self.biases2)# print(np.matmul(x,self.weights)+self.biases)
		output_derivative=sigmoid_derivative(np.matmul(x,self.weights)+self.biases)#1000,100
		output2_derivative=sigmoid_derivative(np.matmul(output_,self.weights2)+self.biases2)#1000,10
		s=np.multiply(output2-y,output2_derivative)/x.shape[0]#1000,100
		lwb2 = np.sum(s,axis=0)
		
		t=np.sum(output_,axis=0)
		lww2 = np.multiply(np.repeat(t.reshape([100, 1]), 10, axis=1), lwb2)
		s2=np.multiply(np.matmul(lwb2,self.weights2.transpose()),output_derivative)
		lwb1 = np.sum(s2,axis=0)
		t2=np.sum(x,axis=0)
		lww1=np.multiply(np.repeat(t2.reshape([784,1]),100,axis=1),lwb1)
		# if i % 100 == 0 and i!=0:
		# 	print(np.min(lww1[lww1!=0]), np.max(lwb1))
		# exit(-1)
		self.weights-=self.lr*lww1
		self.biases-=self.lr*lwb1
		self.weights2-=self.lr*lww2
		self.biases2-=self.lr*lwb2
		# if i%1000==0 and i!=0:
		# 	# exit(-1)
		# 	# print(lww1[lww1!=0])
		# 	#self.lr=self.lr*0.99
		# 	print('\n---------------------------------------------\n')
		# 	print('LR:', self.lr)
		# 	print('Loss: ', np.mean(np.square(output2 - y)))
		# 	print('Accuracy: ', np.mean((np.argmax(output2, axis=1) == np.argmax(y, axis=1)).astype(np.float32)))

		# 	print('\n---------------------------------------------\n')
		# 	# print('\n')

	# def stochastic_gradient_descent(self,x,y, i):
	# 	# print(y[0])
	# 	# self.show(x[0])

	# 	if i == 0:
	# 		earlier_weights = self.weights

	# 	x=np.array(x)
	# 	input_= sigmoid(np.matmul(x,self.weights)+self.biases)#55000,10		
	# 	s=(input_-y)*sigmoid_derivative(input_)
	# 	lwb1=np.sum(s,axis=0)
	# 	# lwb1 = input_ - y
	# 	t = np.sum(x, axis=0)
	# 	# print(t.shape)
	# 	lww1 = np.multiply(np.repeat(t.reshape([784, 1]), 10, axis=1), lwb1)
	# 	#1000,100
	# 	# lww1=np.zeros((784,10))
	# 	# for j in range(784):
	# 	# 	for i in range(10):
	# 	# 		sum_=0
	# 	# 		for k in range(self.batch_size):
	# 	# 			sum_=sum_+ s[k][i]*x[k][j]
	# 	# 		lww1[j][i]=sum_
	# 	self.weights-= self.lr*lww1
	# 	print(lww1[lww1!=0])
	# 	# print(self.weights - self.earlier_weights)
	# 	# self.earlier_weights = self.weights
	# 	self.biases-= self.lr*lwb1
	# 	if (i+1) % 40 == 0:
			
	# 		self.lr = self.lr*0.99
	# 		print('\n---------------------------------------------\n')
	# 		print('LR:', self.lr)
	# 		print('Loss: ', np.mean(np.square(input_ - y)))
	# 		print('Accuracy: ', np.mean((np.argmax(input_, axis=1) == np.argmax(y, axis=1)).astype(np.float32)))

	# 		print('\n---------------------------------------------\n')
			
		# print(np.mean(lww1),np.mean(lwb1))

		
		
x = Function()
sum_=0
for epoch in range(100000):
		idx = np.random.choice(mnist.train.images.shape[0], 100)
		# for i in range(1000):
		# 	rand=np.random.randint(0,mnist.train.images.shape[0])
		x.stochastic_gradient_descent(mnist.train.images[idx],mnist.train.labels[idx], epoch)	
		if epoch%1000==0 and epoch!=0:
			x.test(mnist.test.images,mnist.test.labels)








		