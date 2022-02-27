from .activation import linear, relu, sigmoid
import numpy as np
from .utils  import init_bias, init_weight

from typing import List

act_func = {
	"linear" : linear,
	"relu" : relu,
	"sigmoid" : sigmoid
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
	"""
	def __init__(self,units:int,input_dim:int=None,activation:str="linear"):
		self.units = units
		self.activation = np.vectorize(lambda t : act_func[activation](t))
		self.input_dim = input_dim


		if input_dim != None:
			self._compile_weight_and_bias(input_dim)
		else:
			self.weights = None
			self.biases = None

	def batch_biases(self,n : int) -> np.ndarray:
		"""
		[DESC]
			Turn bias vector into matrix
		[PARAMS]
			n : int
		[RETURN]
			np.ndarray
		"""
		return np.array([self.biases for i in range(n)])

	def forward_feed(self,input_matrix : np.ndarray) -> np.ndarray:
		"""
		[DESC]
			Forward propagation through 1 dense layer
		[PARAMS]
			input_matrix : np.ndarray
		[RETURN]
			np.ndarray
		"""
		number_of_batch = len(input_matrix)
		net = np.dot(input_matrix,self.weights) + self.batch_biases(number_of_batch)
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
	"""
	def __init__(self,random_state=None):
		self.layers: List[Dense] = []
		np.random.seed(random_state)

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

	def compile(self):
		pass

	def forward_feed(self,input_matrix) -> np.ndarray:
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
	
	# back propagation
	def backward_feed(self,y_true,y_pred):
		pass

	def fit(self,X,y,epochs,batch_size):
		pass
