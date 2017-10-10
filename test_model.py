from linear_function import LinearFunction
from neural_network_function import NeuralNetworkFunction


#model = LinearFunction()
#model.load('model/linear.json')
model = NeuralNetworkFunction()
model.load('model/neural.model')

if __name__ == '__main__':
    from game import Game
    g = Game('ABICO')
    s = g.get_current_state()
    a, q = model.get_action(g)

    money_list = [g.money]
    date_list = [g.history[-1].date]
    while not g.is_stop():
        #print 'Date ', g.history[-1].date,'\t'+a,'\t'+s,  g.money
        #model.print_q()
        r,s_prime = g.take_action(a)
        money_list.append(g.money)
        date_list.append(g.history[-1].date)
        a_prime, q = model.get_action(g)
        #print r, q_old, q_new, delta, g.money,
        s = s_prime
        a = a_prime

    import matplotlib.pyplot as plt

    plt.plot(date_list, money_list)
    plt.xlabel('Years')
    plt.ylabel('Current asset')
    plt.show()
     



