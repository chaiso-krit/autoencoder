#!/usr/bin/env python
import numpy
import random

import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import theano
import theano.tensor as T

from network.network import Network

class Autoencoder(object):
    def __init__(self, nnet, dataset=None, learning_rate=0.01, beta=0.0, sparsity=0.01, weight_decay=0.0, momentum=0.5):
        if len(dataset) < 2:
            print "Error dataset must contain tuple (train_data,train_target)"
        train_data, train_target = dataset

        target = T.matrix('y')

        square_error = T.mean(0.5*T.sum(T.pow(target - nnet.output, 2), axis=1))

        avg_activate = T.mean(nnet.hiddenLayer[0].output, axis=0)
        sparsity_penalty = beta*T.sum(T.mul(T.log(sparsity/avg_activate), sparsity) + T.mul(T.log((1-sparsity)/T.sub(1,avg_activate)), (1-sparsity)))

        regularization = 0.5*weight_decay*(T.sum(T.pow(nnet.params[0][0:64*25],2)) + T.sum(T.pow(nnet.params[1][0:25*64],2)))

        cost = square_error + sparsity_penalty + regularization
        
        gparams = [T.grad(cost, param) for param in nnet.params]
  
        index = T.lscalar()

        self.batch_grad = theano.function(
            inputs=[index],
            outputs=T.concatenate(gparams),
            givens={
                input: train_data[index * batch_size: (index + 1) * batch_size],
                target: train_target[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.batch_cost = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                input: train_data[index * batch_size: (index + 1) * batch_size],
                target: train_target[index * batch_size: (index + 1) * batch_size]
            }
        )

def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.array(data_x,dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.array(data_y,dtype=theano.config.floatX), borrow=borrow)
        return (shared_x, shared_y)

if __name__ == "__main__":
    data = sio.loadmat('dataset/patch_images.mat')
    ds = shared_dataset((data['data'], data['data']))

    batch_size = 600
    n_batch = len(data['data'])/batch_size

    input = T.matrix('input')

    nnet = Network(input=input, layers=(64,25,64), bias=True, activation="sigmoid", output_activation="sigmoid")
    trainer = Autoencoder(nnet=nnet, dataset=ds, learning_rate=1.2, beta=3.0, sparsity=0.01, weight_decay=0.0001)

    print 'N Batch:', n_batch

    def train_fn(theta_values):
        nnet.set_weight(theta_values)
        train_losses = [trainer.batch_cost(batch_index) for batch_index in xrange(n_batch)]
        return numpy.mean(train_losses)

    def train_fn_grad(theta_values):
        nnet.set_weight(theta_values)
        grad = trainer.batch_grad(0)
        for batch_index in xrange(1, n_batch):
            grad += trainer.batch_grad(batch_index)
        return grad/n_batch

    global epoch_counter
    epoch_counter = 0
    def callback(theta_values):
        global epoch_counter
        cost = train_fn(theta_values)
        epoch_counter += 1
        print "epochs", epoch_counter, cost
        
    rng = numpy.random.RandomState()
    weight_values = numpy.asarray(
        rng.uniform(
            low=-(numpy.sqrt(.5)),
            high=(numpy.sqrt(.5)),
            size=nnet.get_weight_size()),
            dtype=theano.config.floatX
        )

    import scipy.optimize
    best_w_b = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=weight_values,
        fprime=train_fn_grad,
        callback=callback,
        disp=0,
        maxiter=1000
    )

    # Visualize

    weight = nnet.hiddenLayer[0].W.eval()
    weight = numpy.transpose(weight)
    weight = weight - weight.mean()
    size = numpy.absolute(weight).max()
    weight = weight/size
    weight = weight*0.8+0.5

    image_per_row = 5
    result = []
    index = 0
    row = []
    for face in weight:
        face = face.reshape(8,8)
        face = numpy.lib.pad(face, (1,1), 'constant', constant_values=(0,0))
        if len(row) == 0:
            row = face
        else:
            row = numpy.concatenate([row,face],axis=1)

        if index % image_per_row == image_per_row-1:
            if len(result) == 0:
                result = row
            else:
                result = numpy.concatenate([result,row])
            row = []
        index += 1
    
    result *=255
    implot = plt.imshow(result, cmap=cm.Greys_r, vmin=0, vmax=255)
    implot.set_interpolation('nearest')
    plt.show()

