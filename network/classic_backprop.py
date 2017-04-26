#!/usr/bin/env python
import numpy
import theano
import theano.tensor as T
from network import Network

import matplotlib.pyplot as plt

class BackProp(object):
    def __init__(self, nnet, input_x, dataset=None, learning_rate=0.01, momentum=0.0):
        if len(dataset) != 2:
            print "Error dataset must contain tuple (train_data,train_target)"

        train_data, train_target = dataset

        # share train data and target data
        shared_train_data = theano.shared(numpy.asarray(train_data,dtype=theano.config.floatX),borrow=True)
        shared_train_target = theano.shared(numpy.asarray(train_target,dtype=theano.config.floatX),borrow=True)

        w_deltas = []
        for param in nnet.params:
            w_deltas.append(theano.shared(value=param.get_value()*0, borrow=True))

        cost = T.mean(0.5*T.sum(T.pow(shared_train_target - nnet.output, 2), axis=1))
        
        gparams = [T.grad(cost, param) for param in nnet.params]

        new_params = [
            param - (learning_rate*gparam + momentum*w_delta) 
            for param, gparam, w_delta in zip(nnet.params, gparams, w_deltas)
        ]

        updates = [(param, new_param) for param, new_param in zip(nnet.params, new_params)]

        updates += [(w_delta, (learning_rate*gparam + momentum*w_delta)) for w_delta, gparam in zip(w_deltas, gparams)]

        self.train_one_epochs = theano.function(
            inputs=[],
            outputs=cost,
            updates=updates,
            givens={
                input_x: shared_train_data
            }
        )
    
    def train_epochs(self, epochs=1, verbose=False):
        errors = []
        for i in range(epochs):
            cost = self.train_one_epochs()
            if verbose: print "epochs:", i+1, "error:", cost
            errors.append(cost)
        return errors
        

if __name__ == "__main__":
    xor_data_values = [[1.0,1.0],[1.0,0.0],[0.0,1.0],[0.0,0.0]]
    xor_target_values = [[0],[1],[1],[0]]
    ds = (xor_data_values, xor_target_values)

    x = T.matrix('x')

    nnet = Network(input=x, layers=(2,2,1))
    trainer = BackProp(nnet=nnet, input_x=x, dataset=ds, learning_rate=0.5)

    errors = trainer.train_epochs(600)

    predict = theano.function(inputs=[x], outputs=nnet.output)
    print predict(numpy.array(xor_data_values))

    plt.plot(errors)
    plt.show()



