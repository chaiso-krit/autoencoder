import os
import json

class Model:
    def __init__(self):
        pass

    def save(self, name):
        fptr = open(name, 'w')
        data = self.get_param()
        json.dump(data, fptr)
        fptr.close()

    def load(self, name):
        if os.path.isfile(name):
            fptr = open(name, 'r')
            data = json.load(fptr)
            self.load_param(data)
            fptr.close()
        else:
            print name + ' not exist'

    def print_q(self):
        pass

    # return param dict
    def get_param(self):
        pass

    # load dict to param
    def load_param(self, data):
        pass
