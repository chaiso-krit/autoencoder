#!/usr/bin/env python
import numpy

import theano
import theano.tensor as T

class Layer(object):
    def __init__(self, name, n_in, n_out, input=T.matrix('x'), rng=numpy.random.RandomState(), bias=False, activation="tanh"):
        self.input = input
        self.name = name
        self.activation = activation
        self.bias = bias
        self.size = (n_in,n_out)

        self.theta_size = n_in*n_out
        if bias :
            self.theta_size += n_out

        theta_values = numpy.asarray(
            rng.uniform(
                low=-(numpy.sqrt(6.)/ numpy.sqrt(n_in + n_out+1)),
                high=(numpy.sqrt(6.)/ numpy.sqrt(n_in + n_out+1)),
                size=self.theta_size),
                dtype=theano.config.floatX
            )
        if self.get_activation() == theano.tensor.nnet.sigmoid:
            theta_values *= 4

        self.theta = theano.shared(value=theta_values, name='W', borrow=True)

        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))

        if activation == "none":
            self.theta = None
            lin_output = input
        elif bias:
            self.b = self.theta[n_in * n_out:n_in * n_out + n_out]
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        activation_fn = self.get_activation()

        self.output = (
            lin_output if activation_fn is None
            else activation_fn(lin_output)
        )
        
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
                'theta':self.theta.get_value().tolist(),
                'activation':self.activation,
                'size':self.size,
                'bias':self.bias}

    @staticmethod
    def load(layer_data, input=T.matrix('x')):
        n_in,n_out = layer_data['size']
        new_layer =  Layer(name = layer_data['name'],
                     input = input,
                     activation = layer_data['activation'],
                     n_in = n_in,
                     n_out = n_out,
                     bias = layer_data['bias'])

        theta_values = numpy.asarray(layer_data['theta'], dtype=theano.config.floatX)
        new_layer.theta.set_value(theta_values, borrow=True)

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

    params = theano.function(inputs=[], outputs=layer.theta)
    forward = theano.function(inputs=[x], outputs=layer.output)

    print "Params:", params()
    print "Forward:", forward(xor_data_values)

