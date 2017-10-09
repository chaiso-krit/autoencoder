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

    day_no = 1
    while not g.is_stop():
        print 'Day ', day_no,'\t'+a,'\t'+s,  g.money
        day_no += 1
        #model.print_q()
        r,s_prime = g.take_action(a)
        a_prime, q = model.get_action(g)
        #print r, q_old, q_new, delta, g.money,
        s = s_prime
        a = a_prime



