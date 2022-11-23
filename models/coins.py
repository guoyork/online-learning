from models.base import Base
from math import gamma

class Coins(Base):
    def get_end(self):
        return self.args["m"]

    def calc_report(self, y):
        alpha = self.args['alpha']
        beta = self.args['beta']
        m = self.args['m']
        return tuple([(alpha + y[i]) / (alpha + beta + m[i]) for i in range(len(m))])
    
    def calc_prob(self, y):
        alpha = self.args['alpha']
        beta = self.args['beta']
        m = self.args['m']
        msum = sum(list(m))
        ysum = sum(list(y))
        res = (gamma(alpha + beta) * gamma(beta + msum - ysum) * gamma(alpha + ysum) 
            / gamma(alpha) / gamma(beta) / gamma(alpha + beta + msum))
        for i in range(len(m)):
            res *= gamma(m[i] + 1) / gamma(y[i] + 1) / gamma(m[i] - y[i] + 1)
        # print(y, integral, combinations)
        return res

    def calc_benchmark(self, y):
        alpha = self.args['alpha']
        beta = self.args['beta']
        m = self.args['m']
        msum = sum(list(m))
        ysum = sum(list(y))
        return (alpha + ysum) / (alpha + beta + msum)


class CoinsOrder2(Coins):
    def calc_report(self, y):
        alpha = self.args['alpha']
        beta = self.args['beta']
        m = self.args['m']
        msum = sum(list(m))
        order_1 = [(alpha + y[i]) / (alpha + beta + m[i]) for i in range(len(m))]
        order_2 = [(alpha + order_1[i] * (msum - m[i])) / (alpha + beta + (msum - m[i])) for i in range(len(m))]
        return tuple(order_1) + tuple(order_2)