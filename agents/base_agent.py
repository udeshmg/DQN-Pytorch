import abc

class BaseAgent():

    def __init__(self, network):
        pass

    def learn(self):
        raise NotImplementedError

    def select_action(self, state):
        raise NotImplementedError