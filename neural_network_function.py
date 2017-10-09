import random
import numpy
from read_data import StockData
from model import Model

import theano
import theano.tensor as T

from network.network import Network

class NeuralNetworkFunction(Model):
    random_action = False
    step_size = 0.01

    def __init__(self):
        Model.__init__(self)
       
        input = T.matrix('input')
        target = T.scalar('y')

        self.nnet_input = input
        self.nnet_target = target
        self.nnet = Network(input=input, layers=(60 * 5 + 3 , 15, 10, 4), bias=True, activation="tanh", output_activation="linear")
        self.q_output = theano.function(inputs=[input], outputs=self.nnet.output)

        self.save_input_list = theano.shared(numpy.asarray([[0.0 for i in xrange(60*5+3)]],dtype=theano.config.floatX),borrow=True)
        self.train_buy = self.create_train_function('buy')
        self.train_wait_buy = self.create_train_function('wait_buy')
        self.train_wait_sell = self.create_train_function('wait_sell')
        self.train_sell = self.create_train_function('sell')

    def save(self, name):
        self.nnet.save(name)

    def load(self, name):
        self.nnet = Network.load(name, input=self.nnet_input)
        self.q_output = theano.function(inputs=[self.nnet_input], outputs=self.nnet.output)

        self.save_input_list = theano.shared(numpy.asarray([[0.0 for i in xrange(60*5+3)]],dtype=theano.config.floatX),borrow=True)
        self.train_buy = self.create_train_function('buy')
        self.train_wait_buy = self.create_train_function('wait_buy')
        self.train_wait_sell = self.create_train_function('wait_sell')
        self.train_sell = self.create_train_function('sell')

    def normalize(self, stock_data):
        max_data = StockData(0, 0, 0, 0, 0)
        for data in stock_data:
            #print data.high, data.volume
            if data.high > max_data.high:
                max_data.high = data.high
            if data.volume > max_data.volume:
                max_data.volume = data.volume

        #print max_data.high, max_data.volume
        result = []
        for i in xrange(len(stock_data)):
            new_data = StockData(0, 0, 0, 0, 0)
            new_data.open = stock_data[i].open / max_data.high
            new_data.high = stock_data[i].high / max_data.high
            new_data.low = stock_data[i].low / max_data.high
            new_data.close = stock_data[i].close / max_data.high
            new_data.volume = 1.0*stock_data[i].volume / max_data.volume
            result.append(new_data)

        return result

    def generate_input_from_game(self, game):
        input_list = self.generate_input(game.get_current_state(), game.history, game.buy_price)
        return input_list

    def generate_input(self, state, history, buy_price):
        input_list = [0.0 for i in xrange(60*5+3)]
        #print 'len',len(history)
        if buy_price == 0:
            input_list[0] = 0
        else:
            input_list[0] = (history[-1].open - buy_price) / buy_price 

        input_list[1] = 0
        input_list[2] = 0
        if state == 'hold':
            input_list[1] = 1
        elif state == 'no_hold':
            input_list[2] = 1

        history_new = self.normalize(history)
        input_index = 3
        for data in history_new:
            input_list[input_index] = data.open
            input_list[input_index+1] = data.high
            input_list[input_index+2] = data.low
            input_list[input_index+3] = data.close
            input_list[input_index+4] = data.volume
            input_index += 5

        return input_list

    def get_Q(self, state, action, history, buy_price):
        input_list = self.generate_input(state, history, buy_price)
        self.input_list = input_list

        return self.get_q_for_action(input_list, action)

    def get_q_for_action(self, input_list, action):
        q_value = self.q_output(numpy.array([input_list]))

        if action == 'buy':
            return q_value[0][0]
        elif action == 'wait_buy':
            return q_value[0][1]
        elif action == 'sell':
            return q_value[0][2]
        elif action == 'wait_sell':
            return q_value[0][3]
 

    def get_best_q(self, input_list, actions):
        max_output = self.get_q_for_action(input_list, actions[0])
        for a in actions[1:]:
            output = self.get_q_for_action(input_list, a)
            #print a,output
            if output > max_output:
                max_output = output
        return max_output

    def print_q(self):
        q_value = self.q_output(numpy.array([self.input_list]))
        print q_value
        

    def save_input(self):
        self.save_input_list = theano.shared(numpy.asarray([self.input_list],dtype=theano.config.floatX),borrow=True)

    def set_input(self, input_list):
        self.save_input_list = theano.shared(numpy.asarray([input_list],dtype=theano.config.floatX),borrow=True)

    def create_train_function(self, action):
        if action == 'buy':
            cost = T.pow(self.nnet_target - self.nnet.output[0][0], 2)
        elif action == 'wait_buy':
            cost = T.pow(self.nnet_target - self.nnet.output[0][1], 2)
        elif action == 'sell':
            cost = T.pow(self.nnet_target - self.nnet.output[0][2], 2)
        elif action == 'wait_sell':
            cost = T.pow(self.nnet_target - self.nnet.output[0][3], 2)
        
        gparams = [T.grad(cost, param) for param in self.nnet.params]
        new_params = [ param - (self.step_size*gparam) for param, gparam in zip(self.nnet.params, gparams)]
        updates = [(param, new_param) for param, new_param in zip(self.nnet.params, new_params)]

        train_function = theano.function(
            inputs=[self.nnet_target],
            outputs=cost,
            updates=updates,
            givens={
                self.nnet_input: self.save_input_list
            }
        )
        return train_function

    def update_weight(self, target, action, epochs = 1):
        for train_index in xrange(epochs):
            if action == 'buy':
                cost = self.train_buy(target)
            elif action == 'wait_buy':
                cost = self.train_wait_buy(target)
            elif action == 'sell':
                cost = self.train_sell(target)
            elif action == 'wait_sell':
                cost = self.train_wait_sell(target)
        
        return cost

    def get_action(self, game):
        state = game.get_current_state()
        history = game.history
        buy_price = game.buy_price
        actions = game.get_actions()
        return self.get_best_action(state, history, buy_price, actions)

    def get_best_action(self, state, history, buy_price, actions):
        max_output = self.get_Q(state, actions[0], history, buy_price)
        max_a = actions[0]
        for a in actions[1:]:
            output = self.get_Q(state, a, history, buy_price)
            #print a,output
            if output > max_output:
                max_output = output
                max_a = a
        return max_a, max_output
    
        

if __name__ == '__main__':
    model = NeuralNetworkFunction()
    model.load('model/neural.model')
    model.random_action = False
    epsilon = 0.05
    discount_factor = 0.9
    memory_list = []
    from game import Game
    for iteration in xrange(3000):
        print 'Iter:', iteration + 1,
        g = Game('ABICO')

        q_array = []
        sub_mem = []
        sum_r = 0
        while not g.is_stop():
            s = g.get_current_state()
            if g.is_last_day() and s == 'hold':
                a = 'sell'
            elif random.random() < epsilon:
                a = random.choice(g.get_actions())
                if a == 'sell' and g.buy_price >= g.history[-1].open:
                    a = 'wait_sell'
            else:
                a, q_value = model.get_action(g)
                #model.print_q()

            q_array.append(q_value)
            old_state = model.generate_input_from_game(g)
            actions_list = g.get_actions()
            r,s_prime = g.take_action(a)
            sum_r += r
            new_state = model.generate_input_from_game(g)

            sub_mem.append([old_state,new_state,r,a,actions_list])

        memory_list = memory_list + sub_mem

        print ' Avg. Q:', sum(q_array)/len(q_array),
        print ' Sum Reward :', sum_r,
        print ' Money :', g.money ,

        target_array = [0.0]

        # replay memory
        if len(memory_list) > 1000:
            random.shuffle(memory_list)

            if len(memory_list) > 10000:
                max_iter = 10000
            elif len(memory_list) > 5000:
                max_iter = 5000
            else:
                max_iter = 1000

            for repeat_loop in xrange(max_iter):
                old_state,new_state,r,a,actions_list = memory_list[repeat_loop]

                q_new = model.get_best_q(new_state, actions_list)

                if a == 'sell':
                    target = r
                else:
                    target = r + discount_factor*q_new

                model.set_input(old_state)
                cost = model.update_weight(target, a, epochs=5)
                target_array.append(target)

            if len(memory_list) > 30000:
                memory_list = memory_list[:30000]

        print ' Avg. Target:', sum(target_array)/len(target_array)

        # backup
        if (iteration+1) % 100 == 0:
            model.save('model/neural.model')
        
    model.save('model/neural.model')





