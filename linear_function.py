import random
from read_data import StockData
from model import Model

class LinearFunction(Model):
    step_size = 0.001

    def __init__(self):
        Model.__init__(self)
        self.w = [random.random() for i in xrange(60*5 + 8)]
        self.f = [0.0 for i in xrange(len(self.w))]

    def get_param(self):
        return self.w

    def load_param(self, data):
        self.w = data

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
            new_data = StockData(0, 0, 0, 0, 0, '19910101')
            new_data.open = stock_data[i].open / max_data.high
            new_data.high = stock_data[i].high / max_data.high
            new_data.low = stock_data[i].low / max_data.high
            new_data.close = stock_data[i].close / max_data.high
            new_data.volume = 1.0*stock_data[i].volume / max_data.volume
            result.append(new_data)

        return result

    def get_Q(self, state, action, history, buy_price):
        #print 'len',len(history)
        if buy_price == 0:
            self.f[0] = 0
        else:
            self.f[0] = (history[-1].open - buy_price) / buy_price 

        self.f[1] = 0
        self.f[2] = 0
        if state == 'hold':
            self.f[1] = 1
        elif state == 'no_hold':
            self.f[2] = 1

        self.f[3] = 0
        self.f[4] = 0
        self.f[5] = 0
        if action == 'buy':
            self.f[3] = 1
        elif action == 'sell':
            self.f[4] = 1
        elif action == 'wait_sell':
            self.f[5] = 1
        elif action == 'wait_buy':
            self.f[6] = 1

        self.f[7] = 1

        history_new = self.normalize(history)
        f_index = 8
        for data in history_new:
            self.f[f_index] = data.open
            self.f[f_index+1] = data.close
            self.f[f_index+2] = data.volume
            self.f[f_index+3] = data.high
            self.f[f_index+4] = data.low
            f_index += 5

        return self.compute_q(self.f)

    def compute_q(self, f_input):
        sum_f = 0.0
	for i in xrange(len(f_input)):
            sum_f += self.w[i] * f_input[i]
        return sum_f

    def print_q(self):
        f_temp = self.f[:]
        f_temp[3] = 0
        f_temp[4] = 0
        f_temp[5] = 0
        f_temp[6] = 0

        q_value = []
        f_temp[3] = 1
        q_value.append(self.compute_q(f_temp))
        f_temp[3] = 0
        f_temp[4] = 1
        q_value.append(self.compute_q(f_temp))
        f_temp[4] = 0
        f_temp[5] = 1
        q_value.append(self.compute_q(f_temp))
        f_temp[5] = 0
        f_temp[6] = 1
        q_value.append(self.compute_q(f_temp))
        print q_value
        
    def save_f(self):
        self.f_save = self.f[:]

    def update_weight(self, delta):
        for i in xrange(len(self.f)):
            self.w[i] = self.w[i] + (self.step_size * delta * self.f_save[i])

    def get_action(self, game):
        state = game.get_current_state()
        actions = game.get_actions()
        max_output = self.get_Q(state, actions[0], game.history, game.buy_price)
        max_a = actions[0]
        for a in actions[1:]:
            output = self.get_Q(state, a, game.history, game.buy_price)
            #print a,output
            if output > max_output:
                max_output = output
                max_a = a
        return max_a, max_output
        

if __name__ == '__main__':
    linear = LinearFunction()
    linear.load('model/linear.json')
    from game import Game
    for iteration in xrange(3000):
        print 'Iter:', iteration + 1,
        g = Game('ABICO')

        discount_factor = 0.9
        epsilon = 0.05
        delta_array = []
        while not g.is_stop():
            s = g.get_current_state()
            if g.is_last_day() and s == 'hold':
                a = 'sell'
            elif random.random() < epsilon:
                a = random.choice(g.get_actions())
                if a == 'sell' and g.buy_price >= g.history[-1].open:
                    a = 'wait_sell'
            else:
                a, q_value = linear.get_action(g)

            linear.save_f()
            r,s_prime = g.take_action(a)
            a_prime, q_new = linear.get_action(g)
            if a == 'sell':
                delta = r - q_value
            else:
                delta = r + discount_factor*q_new - q_value

            linear.update_weight(delta)
            delta_array.append(delta)

            s = s_prime
            a = a_prime

        print ' Avg. Delta:', sum(delta_array)/len(delta_array),
        print ' Money :', g.money
    linear.save('model/linear.json')





