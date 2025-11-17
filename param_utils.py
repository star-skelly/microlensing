
import matplotlib.pyplot as plt
import numpy as np
import MulensModel as mm

# More sophisticated parameter generation
class Param:
    def __init__(self, name, pmin, pmax, init):
        self.name = name
        self.min = pmin
        self.max = pmax
        self.init = init
    
    def cutoff(self, inmin, inmax):
        x = 0
        if self.name == "u_0":
            x = np.sqrt(np.random.uniform(inmin, inmax))
        elif self.name == "alpha":
            x = np.random.normal(inmin, inmax)
        else:
            x = 10**np.random.uniform(inmin, inmax)

        if x > self.max:
            return self.max
        if x < self.min:
            return self.min
        
        return x

    def get_sample_n(self, inmin, inmax, n):
        samples = []
        labels = []
        for i in range(n):
            s = self.get_sample(inmin, inmax)
            samples.append(s[0])
            labels.append(s[1])
        return samples, labels
    
    def get_sample(self, inmin, inmax):
        x = self.cutoff(inmin, inmax)
        lab = self.get_label(x)
        return [lab, x]

    def get_label(self, x):
        return str(f"{self.name}: " + "{:.2f}".format(x))

class Params:
    def __init__(self):
        """u_0, t_E, rho, q, s, alpha"""
        self.u_0 = Param("u_0", 0, np.sqrt(1), 0.5)
        self.t_E = Param("t_E", 3, 100, 50)
        self.rho = Param("rho", 10e-8, 10e-3, 10e-4)
        self.q = Param("q", 10e-5, 1, 10e-3)
        self.s = Param("s", -0.1, 10, 1)
        self.alpha = Param("alpha", 0, 360, 180)

    def get_vals(self, n):
        all_data = [
            self.u_0.get_sample_n(0, 1, n), self.t_E.get_sample_n(0.5, 2, n),
            self.rho.get_sample_n(-10, -3, n), self.q.get_sample_n(-5, 0, n), 
            self.s.get_sample_n(-1, 1, n), self.alpha.get_sample_n(180, 40, n)
        ]
        vals = [dat[1] for dat in all_data]
        labels = [dat[0] for dat in all_data]
        return vals, labels
    
    def get_vals(self):
        all_data = [
            self.u_0.get_sample(0, 1), self.t_E.get_sample(0.5, 2),
            self.rho.get_sample(-10, -3), self.q.get_sample(-5, 0), 
            self.s.get_sample(-1, 1), self.alpha.get_sample(180, 40)
        ]
        vals = [dat[1] for dat in all_data]
        labels = [dat[0] for dat in all_data]
        return vals, labels
    
    def get_inits(self):
        return [self.u_0.init,
                self.t_E.init,
                self.rho.init,
                self.q.init,
                self.s.init,
                self.alpha.init]
    