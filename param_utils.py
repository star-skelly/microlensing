
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
            x = np.random.uniform(inmin, inmax)
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
        return np.array(samples), np.array(labels)
    
    def get_sample(self, inmin, inmax):
        x = self.cutoff(inmin, inmax)
        lab = self.get_label(x)
        return [x, lab]

    def get_label(self, x):
        return str(f"{self.name}: " + "{:.2f}".format(x))
    
    def get_label_n(self, x):
        return [self.get_label(x_i) for x_i in x]

class Params:
    def __init__(self):
        """u_0, t_E, rho, q, s, alpha"""
        self.u_0 = Param("u_0", 0, 2, 0.5)
        self.t_E = Param("t_E", 3, 50, 50)
        self.rho = Param("rho", 0.009, 0.1, 0.01) # from Earth size to Jupiter size
        self.density = Param("density", 0.33, 4.3, 1.5)
        self.q = Param("q", 0.33 * 0.009, 4.3 * 0.1, 0.01 * 1.5)
        self.s = Param("s", 0, 5, 1)
        self.alpha = Param("alpha", 0, 180, 90)

    def get_vals(self, n):
        rhos = self.rho.get_sample_n(-3, -1, n)
        q_vals = self.density.get_sample_n(-0.5, 0.65, n)[0] * rhos[0]
        q_labs = self.q.get_label_n(q_vals)

        all_data = [
            self.u_0.get_sample_n(0, 1, n), self.t_E.get_sample_n(0.5, 2, n),
            rhos, (q_vals, q_labs), 
            self.s.get_sample_n(-1, 1, n), self.alpha.get_sample_n(0, 180, n)
        ]
        vals = np.array([dat[0] for dat in all_data]).reshape(n, len(all_data))
        labels = np.array([dat[1] for dat in all_data]).reshape(n, len(all_data))
        return vals, labels
    
    def get_val(self):
        rho = self.rho.get_sample(-3, -1)
        q_val = self.density.get_sample(-0.5, 0.65)[0] * rho[0]
        q_lab = self.q.get_label(q_val)
        
        all_data = [
            self.u_0.get_sample(0, 1), self.t_E.get_sample(0.5, 2),
            rho, (q_val, q_lab), 
            self.s.get_sample(-1, 1), self.alpha.get_sample(0, 180)
        ]
        vals = [dat[0] for dat in all_data]
        labels = [dat[1] for dat in all_data]
        return vals, labels
    
    def get_inits(self):
        return [self.u_0.init,
                self.t_E.init,
                self.rho.init,
                self.density.init * self.rho.init,
                self.s.init,
                self.alpha.init]
    