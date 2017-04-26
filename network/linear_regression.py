#!/usr/bin/env python
import numpy
import theano
import theano.tensor as T
from network import Network
from classic_backprop import BackProp

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_values = [[2.7],[1.4],[0.5],[0.9],[3],[3],[1.3],[0.5],[0.6],[1.4],[1.2],[1.6],[1.8],[0.6],[2.2],[2.8],[1.4]]
    target_values = [[618],[297],[144],[219],[628],[608],[319],[165],[128],[302],[246],[384],[421],[198],[505],[574],[369]]

    ds = (data_values, target_values)

    x = T.matrix('x')

    nnet = Network(input=x, layers=(1, 1, 1), bias=True, activation="linear", output_activation="none")
    trainer = BackProp(nnet=nnet, input_x=x, dataset=ds, learning_rate=0.4)

    errors = trainer.train_epochs(30)

    test_data = [[i/10.0] for i in range(35)]
    test_target = [i[0]*200+50 for i in test_data]
    
    predict = theano.function(inputs=[x], outputs=nnet.output)
    predicted_values = predict(numpy.array(test_data))

    plt.plot(test_data, test_target, label='Target value')
    plt.plot(test_data, predicted_values, label='Predict value')
    plt.legend()
    plt.show()



