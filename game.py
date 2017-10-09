from read_data import Data
import random

class Game:
    start_money = 5000

    def __init__(self, name):
        self.data = Data(name)
        self.graph = {'hold': ['wait_sell', 'sell'],
                      'no_hold': ['wait_buy', 'buy']}
        self.state = 'no_hold'
        self.current_price = 0.0
        self.buy_price = 0.0
        self.money = Game.start_money
        self.stock = 0.0
        self.history = self.data.get_first()
        self.current_price = self.history[-1].open

    def take_action(self, action):
        if action == 'wait_sell' or action == 'wait_buy':
            self.wait()
            reward = 0
        elif action == 'sell':
            reward = (self.current_price - self.buy_price) / self.buy_price
            self.sell()
        elif action == 'buy':
            self.buy()
            reward = 0
        return reward, self.state

    def get_actions(self):
        return self.graph[self.state]

    def get_current_state(self):
        return self.state

    def buy(self):
        self.buy_price = self.current_price
        self.stock = self.money/self.current_price
        self.state = 'hold'
        self.next_step()

    def sell(self):
        self.buy_price = 0.0
        self.money = self.stock*self.current_price
        self.stock = 0.0
        self.state = 'no_hold'
        self.next_step()

    def wait(self):
        if self.state is 'hold':
            self.money = self.stock*self.current_price
        self.next_step()

    def next_step(self):
        self.history.append(self.data.get_next())
        self.history = self.history[1:]
        self.current_price = self.history[-1].open

    def is_last_day(self):
        if not self.data.get_day_left() == 1:
            return True
        else:
            return False

    def is_stop(self):
        if not self.data.get_day_left() == 0:
            return False
        else:
            return True

if __name__ == '__main__':
    count = [0 for i in xrange(100)]
    data = []
    for i in xrange(10000):
        if (i + 1) % 1000 == 1:
            print 'iter :', i+1
        game = Game('ABICO')
        while not game.is_stop():
            action = game.get_actions()
            a = random.choice(action)
            res = game.take_action(a)
            #print a, game.state, game.money

        data.append(game.money)
        if game.money < 0:
            count[0] += 1
        elif game.money >= 200000:
            count[-1] += 1
        else:
            count[int(game.money/2000)] += 1

    print sum(data)/len(data)
    for i in xrange(len(count)):
        print i*1000, count[i]








