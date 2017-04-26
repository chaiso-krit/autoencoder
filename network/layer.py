#!/usr/bin/env python
import numpy

import theano
import theano.tensor as T

class Layer(object):
    def __init__(self, name, n_in, n_out, input=T.matrix('x'), rng=numpy.random.RandomState(), bias=False, activation="tanh", W=None, b=None):
        self.input = input
        self.name = name
        self.activation = activation

        if W == None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-(numpy.sqrt(6.)/ numpy.sqrt(n_in + n_out+1)),
                    high=(numpy.sqrt(6.)/ numpy.sqrt(n_in + n_out+1)),
                    size=(n_in, n_out)),
                    dtype=theano.config.floatX
                )
            if self.get_activation() == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        if b == None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.b = b
        
        if activation == "none":
            lin_output = input
        else:
            lin_output = T.dot(input, self.W) + self.b

        activation_fn = self.get_activation()
        self.output = (
            lin_output if activation_fn is None
            else activation_fn(lin_output)
        )
        
        if activation == "none":
            self.params = []
        elif bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

    def get_activation(self):
        if self.activation == "tanh":
            return T.tanh
        elif self.activation == "sigmoid":
            return T.nnet.sigmoid
        elif self.activation == "relu":
            return T.nnet.relu
        elif self.activation == "linear":
            return None
        elif self.activation == "none":
            return None

    def get_dict(self):
        return {'name':self.name,
                'W':self.W.get_value().tolist(),
                'b':self.b.get_value().tolist(),
                'activation':self.activation,
                'in':self.W.get_value().shape[0],
                'out':self.b.get_value().shape[0],
                'bias':True if len(self.params) > 1 else False}

    @staticmethod
    def load(layer_data, input=T.matrix('x')):
        W = theano.shared(value=numpy.asarray(layer_data['W'], dtype=theano.config.floatX), name='W', borrow=True)
        b = theano.shared(value=numpy.asarray(layer_data['b'], dtype=theano.config.floatX), name='b', borrow=True)
        return Layer(name = layer_data['name'],
                     input = input,
                     W = W,
                     b = b,
                     activation = layer_data['activation'],
                     n_in = layer_data['in'],
                     n_out = layer_data['out'],
                     bias = layer_data['bias'])

if __name__ == "__main__":
    xor_data_values = numpy.array([[1.0,1.0],[1.0,0.0],[0.0,1.0],[0.0,0.0]])
    xor_data = theano.shared(value=xor_data_values, name='xor_data', borrow=True)

    x = T.matrix('x')
    layer = Layer(name='hidden',
                  n_in=2,
                  n_out=1,
                  input=x,
                  bias=True,
                  activation="sigmoid")

    params = theano.function(inputs=[], outputs=layer.params)
    forward = theano.function(inputs=[x], outputs=layer.output)

    print "Params:", params()
    print "Forward:", forward(xor_data_values)

