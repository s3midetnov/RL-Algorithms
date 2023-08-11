import numpy as np


class Bandits:
    def __init__(self, k:int, receive_elem):
        self.k = k
        self.t = k
        
        self.receive_elem = receive_elem
        self.sums = np.array([receive_elem(i) for i in range(k)])
        self.numbers = np.ones(k)
        self.income = self.sums.sum()
        
    def choose_lever():
        raise NotImplementedError

    def means(self):
        return self.sums/self.numbers
    
    def pull_lever(self):
        index = self.choose_lever()
        response = self.receive_elem(index)
        self.numbers[index]+=1        
        self.t+=1
        self.income += response
        self.sums[index] += response
        
class UCB1(Bandits):
    def __init__(self, k:int, receive_elem):
        super().__init__(k, receive_elem)
    
    def choose_lever(self):
        return np.argmax(self.means() + np.sqrt(2*np.log(self.t)/self.numbers))
    
class EpsGreedy(Bandits):
    def __init__(self, k:int, receive_elem, eps:float):
        super().__init__(k, receive_elem)
        self.eps = eps
        
    def choose_lever(self):
        p = np.random.binomial(1, self.eps, 1).item()
        return (1-p)*np.argmax(self.means()) + p*np.random.choice(self.k)
    
    
class Thompson(Bandits):
    def __init__(self, k:int, receive_elem):
        super().__init__(k, receive_elem)
        
    def choose_lever(self):
        return np.argmax([self.receive_elem(i) for i in range(self.k)])