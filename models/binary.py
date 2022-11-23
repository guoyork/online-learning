from models.base import Base
from math import gamma

class Binary(Base):
    def get_end(self):
        return tuple([1 for i in range(len(self.args["reports"]))])

    def calc_report(self, x):
        return tuple([self.args["reports"][i][x[i]] for i in range(len(self.args["reports"]))])
    
    def calc_prob(self, x):
        mu = self.args["mu"]
        p = 1
        p0 = 1 - mu
        p1 = mu
        for i in range(len(self.args["reports"])):
            p *= (abs(self.args["reports"][i][1 - x[i]] - mu)
                / (self.args["reports"][i][1] - self.args["reports"][i][0]))
            p0 *= (1 - self.args["reports"][i][x[i]]) / (1 - mu)
            p1 *= self.args["reports"][i][x[i]] / mu
        return p * (p0 + p1)

    def calc_benchmark(self, x):
        mu = self.args["mu"]
        b0 = 1
        b1 = (1 - mu) / mu
        for i in range(len(self.args["reports"])):
            b0 *= self.args["reports"][i][x[i]]
            b1 *= (1 - self.args["reports"][i][x[i]]) * mu / (1 - mu)
        if b0 == 0:
            return 0
        return b0 / (b0 + b1)

class BinaryOrder2(Binary):
    def calc_report(self, x):
        n = len(self.args["reports"])
        mu = self.args["mu"]
        order_1 = [self.args["reports"][i][x[i]] for i in range(len(self.args["reports"]))]
        E_0 = []
        E_1 = []
        for i in range(n):
            r0 = self.args["reports"][i][0]
            r1 = self.args["reports"][i][1]
            p = (r1 - mu) / (r1 - r0)
            E_0.append((1 - r0) * r0 * p / (1 - mu) + (1 - r1) * r1 * (1 - p) / (1 - mu))
            E_1.append(r0 * r0 * p / mu + r1 * r1 * (1 - p) / mu)
        order_2 = []
        for i in range(n):
            order_2.append(order_1[i] * (sum(E_1) - E_1[i]) / (n - 1) + (1 - order_1[i]) * (sum(E_0) - E_0[i]) / (n - 1))

        return tuple(order_1) + tuple(order_2)