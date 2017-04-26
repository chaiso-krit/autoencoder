#!/usr/bin/env python
import numpy
import theano
import theano.tensor as T
import math

from network import Network
from classic_backprop import BackProp

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_values = [[2.7],[1.4],[0.5],[0.9],[3],[3],[1.3],[0.5],[0.6],[1.4],[1.2],[1.6],[1.8],[0.6],[2.2],[2.8],[1.4],[0.2],[0.3],[0.35]]
    target_values = [[29.327],[22.322],[12.97],[18.9],[29.942],[28.64],[22.278],[13.37],[15.46],[21.922],[22.28],[24.582],[25.4],[16.36],[28.248],[29.1],[23.922],[5.92],[9.6],[10.381]]

    ds = (data_values, target_values)

    x = T.matrix('x')

    nnet = Network(input=x, layers=(1, 4, 1), bias=True, activation="sigmoid", output_activation="linear")
    trainer = BackProp(nnet=nnet, input_x=x, dataset=ds, learning_rate=0.1)

    errors = trainer.train_epochs(2000)

    test_data = [[i/10.0] for i in range(1,35)]
    test_values = [math.log(i[0],10)*20+20 for i in test_data]

    predict = theano.function(inputs=[x], outputs=nnet.output)
    predict_values = predict(numpy.array(test_data))
    
    plt.plot(test_data, test_values, label='Target value')
    plt.plot(test_data, predict_values, label='Predict value')
    plt.legend()
    plt.show()



