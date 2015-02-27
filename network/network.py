#!/usr/bin/env python
import numpy
import json
from layer import Layer

import theano
import theano.tensor as T

class Network(object):
	def __init__(self, new_network=True, **kwargs):
		if new_network:
			self.new_network(**kwargs)

	def new_network(self, layers, input, rng=numpy.random.RandomState(), bias=False, activation="tanh", output_activation="linear"):
		hidden_layer = []
		for layer in xrange(len(layers)-2):
			hiddenLayer = Layer(
				name=('hidden%d') % (layer),
				rng=rng,
				input=input,
				n_in=layers[layer],
				n_out=layers[layer+1],
				bias=bias,
				activation=activation
			)
			hidden_layer.append(hiddenLayer)
			input = hidden_layer[layer].output
		
		output_layer = Layer(
			name="output",
			rng=rng,
			input=hidden_layer[-1].output,
			n_in=layers[-2],
			n_out=layers[-1],
			bias=bias,
			activation=output_activation
		)
		self.create_network(hidden_layer, output_layer)


	def create_network(self, hidden_layer, output_layer):
		self.hiddenLayer = []
		self.params = []
		for layer in hidden_layer:
			self.hiddenLayer.append(layer)
			self.params += layer.params
		self.outputLayer = output_layer
		self.output = self.outputLayer.output
		self.params += output_layer.params


	def save(self, filename):
		fptr = open(filename,'w')
		data = []
		for layer in self.hiddenLayer:
			data.append(layer.get_dict())
		data.append(self.outputLayer.get_dict())
		json.dump(data, fptr)
		fptr.close()

	@staticmethod
	def load(filename, input):
		fptr = open(filename,'r')
		data = json.load(fptr)
		hidden_layer = []
		for layer_data in data[:-1]:
			layer = Layer.load(layer_data, input=input)
			hidden_layer.append(layer)
			input = layer.output
		output_layer = Layer.load(data[-1], input=hidden_layer[-1].output)
		fptr.close()
		nnet = Network(new_network=False)
		nnet.create_network(hidden_layer, output_layer)
		return nnet

if __name__ == "__main__":
	xor_data_values = numpy.array([[1.0,1.0],[1.0,0.0],[0.0,1.0],[0.0,0.0]])

	x = T.matrix('x')
	nnet = Network(input=x, layers=(2,2,1), bias=True, activation="sigmoid")

	params = theano.function(inputs=[], outputs=nnet.params)
	forward = theano.function(inputs=[x], outputs=nnet.output)

	print "Params:", params()
	print "Forward:", forward(xor_data_values)

