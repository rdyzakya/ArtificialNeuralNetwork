from .activation import linear, relu, sigmoid, softmax
from .loss import sum_squared_error, cross_entropy_error
import numpy as np
from .utils  import init_bias, init_weight

from typing import List

act_func = {
	"linear" : linear,
	"relu" : relu,
	"sigmoid" : sigmoid,
	"softmax" : softmax
}

loss_func = {
	"sum_squared_error" : sum_squared_error,
	"cross_entropy_error" : cross_entropy_error
}


class Dense:
	"""
	[DESC]
		Dense layer (fully connected neurons)
	[ATTRIB]
		units : int
		activation : np.vectorize
		input_dim : None or int
		weights : np.ndarray
		biases : np.ndarray
		error_term : np.ndarray
	"""
	def __init__(self,units:int,input_dim:int=None,activation_function:str="linear"):
		self.units = units
		self.activation_function = activation_function
		self.input_dim = input_dim
		self.error_term = None
		self.net = None

		if input_dim != None:
			self._compile_weight_and_bias(input_dim)
		else:
			self.weights = None
			self.biases = None

	def reprJSON(self) -> dict:
		"""
		[DESC]
			Method to represent Dense layer as JSON
		[RETURN]
			dict
		"""
		return dict(units=self.units, activation_function=self.activation_function, input_dim=self.input_dim,weights=self.weights.tolist(),biases=self.biases.tolist())


	def activation(self,obj : np.ndarray, derivative : bool=False) -> np.ndarray:
		"""
		[DESC]
			Method to activate the layer
		[PARAMS]
			obj : np.ndarray
		[RETURN]
			np.ndarray
		"""
		if self.activation_function == "softmax":
			return np.apply_along_axis(act_func["softmax"],1,obj)
		vfunc = np.vectorize(lambda t : act_func[self.activation_function](t,derivative))
		return vfunc(obj)

	def batch_biases(self,n : int) -> np.ndarray:
		"""
		[DESC]
			Turn bias vector into matrix
			*Deprecated
		[PARAMS]
			n : int
		[RETURN]
			np.ndarray
		"""
		return np.array([self.biases for i in range(n)])
	
	def _weights(self):
		"""
		[DESC]
			Method to return weights and biases weight
		[RETURN]
			np.ndarray
		"""
		return np.append(self.weights,[self.biases],axis=0)

	def forward_feed(self,input_matrix : np.ndarray) -> np.ndarray:
		"""
		[DESC]
			Forward propagation through 1 dense layer
		[PARAMS]
			input_matrix : np.ndarray
		[RETURN]
			np.ndarray
		"""
		appended_input_matrix = np.append(input_matrix,np.ones((input_matrix.shape[0],1)),axis=1)
		appended_weights = self._weights()
		net = np.dot(appended_input_matrix,appended_weights)
		self.net = net
		result = self.activation(net)
		return result

	def _compile_weight_and_bias(self,input_dim : int):
		"""
		[DESC]
			Method to initialize weight and bias
		[PARAMS]
			input_dim : int
		"""
		self.input_dim = input_dim
		self.weights = init_weight(input_dim,self.units)
		self.biases = init_bias(self.units)
	
	def compile_weight_and_bias(self,weights : np.ndarray,biases : np.ndarray):
		"""
		[DESC]
			Method to compile weight and bias
		[PARAMS]
			weights : np.ndarray
			biases : np.ndarray
		"""
		if weights.shape[0] != self.input_dim or weights.shape[1] != self.units:
			raise ValueError("Dimension of weights is not correct")
		if biases.shape[0] != self.units:
			raise ValueError("Dimension of biases is not correct")
		self.weights = weights
		self.biases = biases

class Sequential:
	"""
	[DESC]
		Sequential model class
	[ATTRIB]
		layers : list of layers
		loss : loss function
	"""
	def __init__(self,random_state=None):
		self.layers: List[Dense] = []
		self.loss = None
		np.random.seed(random_state)
	
	def reprJSON(self) -> dict:
		"""
		[DESC]
			Method to represent model as JSON
		[RETURN]
			dict
		"""
		return dict(layers=self.layers)

	def useJSON(self,data : dict):
		"""
		[DESC]
			Method to use JSON data to initialize model
		[PARAMS]
			data : dict
		"""
		for i in range(len(data["layers"])):
			units = data["layers"][i]["units"]
			activation_function = data["layers"][i]["activation_function"]
			input_dim = data["layers"][i]["input_dim"]
			weight = np.array(data["layers"][i]["weights"])
			bias = np.array(data["layers"][i]["biases"])
			d = Dense(units,input_dim,activation_function)
			d.compile_weight_and_bias(weight,bias)
			self.layers.append(d)

	def add(self,layer : Dense):
		"""
		[DESC]
			Method to add layer to model
		[PARAMS]
			layer : Dense
		"""
		if len(self.layers) > 0:
			input_dim = self.layers[-1].units
			layer._compile_weight_and_bias(input_dim)
		else:
			if layer.input_dim == None:
				raise Exception("First layer must contain n input dimension(s)")
		self.layers.append(layer)

	def compile(self, loss : str):
		"""
		[DESC]
			Method to compile model
		[PARAMS]
			loss : str
		"""
		if loss != "sum_squared_error" and loss != "cross_entropy_error":
			raise ValueError("Loss function must be sum_squared_error or cross_entropy_error")
		self.loss = loss

	def forward_feed(self,input_matrix : np.ndarray) -> np.ndarray:
		"""
		[DESC]
			Method to execute forward propagation with the given input
		[PARAMS]
			input_matrix : float, List[List[float]], np.ndarray
		[RETURN]
			np.ndarray
		"""
		result = input_matrix
		for layer in self.layers:
			result = layer.forward_feed(result)
		return result
	
	def predict(self,X : np.ndarray) -> np.ndarray:
		"""
		[DESC]
			Method to execute forward propagation with the given input
		[PARAMS]
			X : float, List[List[float]], np.ndarray
		[RETURN]
			np.ndarray
		"""
		return self.forward_feed(X)

	def summary(self):
		"""
		[DESC]
			Method to print summary of model
		"""
		print("MODEL INFO")
		print("========================================================")
		layers = self.reprJSON()['layers']
		for layer in layers:
			lay = layer.reprJSON()
			print('units :',lay['units'], end="  ||  ")
			print('activation function :',lay['activation_function'], end="  ||  ")
			print('input_dim :',lay['input_dim'])
			print('weights :')
			for i in range(len(lay['biases'])):
				print([lay['biases'][i]] + lay['weights'][i])
			print("========================================================")

	
	# back propagation
	def error_term(self,y_true : np.ndarray,y_pred : np.ndarray) -> np.ndarray:
		"""
		[DESC]
			Method to calculate error term
		[PARAMS]
			y_true : np.ndarray
			y_pred : np.ndarray
		[RETURN]
			np.ndarray
		"""
		if y_true.shape != y_pred.shape:
			raise ValueError("Length of y_true and y_pred must be same")
		if self.loss == None:
			raise Exception("Loss function must be set using compile method")
		for ilayer in reversed(range(len(self.layers))):
			layer = self.layers[ilayer]
			# print(layer.net)
			if ilayer == len(self.layers) - 1:
				layer.error_term =  np.sum(layer.activation(layer.net,derivative=True) * loss_func[self.loss](y_true=y_true,y_pred=y_pred,derivative=True),axis=0)
			else:
				d_ilayer = layer.activation(layer.net,derivative=True)
				wkh_dk = np.sum(self.layers[ilayer+1].weights * self.layers[ilayer+1].error_term,axis=1)
				layer.error_term = np.sum(d_ilayer * wkh_dk,axis=0)