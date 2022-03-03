import numpy as np

def init_weight(row : int,column : int) -> np.ndarray:
	"""
	[DESC]
		Function to initialize weights with dimension = m x n
	[PARAMS]
		row : int
		column : int
	[RETURN]
		numpy.ndarray(float)
	"""
	return np.random.randn(row,column)

def init_bias(n : int) -> np.ndarray:
	"""
	[DESC]
		Function to initialize biases with length = n
	[PARAMS]
		n : int
	[RETURN]
		numpy.ndarray(float)
	"""
	return np.zeros(n)

def show_model(model):
	print("MODEL INFO")
	print("========================================================")
	layers = model.reprJSON()['layers']
	for layer in layers:
		lay = layer.reprJSON()
		print('units :',lay['units'], end="  ||  ")
		print('activation function :',lay['activation_function'], end="  ||  ")
		print('input_dim :',lay['input_dim'])
		print('weights :')
		for i in range(len(lay['biases'])):
			print([lay['biases'][i]] + lay['weights'][i])
		print("========================================================")

		
	