from models.base import Base
from math import gamma, sqrt, log

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

    def myfunc(x, N=10000, theta=0.02254248593736856025573354295705, with_lower_upper=False):
        x1 = min(x[0], x[1])
        x2 = max(x[0], x[1])
        if (x1 == 0) or (x2 == 0):
            if with_lower_upper:
                return .0, .0, .0
            else:
                return .0
        elif (x1 == 1) or (x2 == 1):
            if with_lower_upper:
                return 1., 1., 1.
            else:
                return 1.
        else:
            lower = 0
            upper = 1
            for uu in range(1, N):
                u = uu / N
                p = (1 - x1) * (1 - x2) / (1 - u) + x1 * x2 / u
                b = x1 * x2 / (x1 * x2 + (1 - x1) * (1 - x2) * u / (1 - u))
                q1 = u / x1 if x1 >= u else (1 - u) / (1 - x1)
                q2 = u / x2 if x2 >= u else (1 - u) / (1 - x2)
                lower = max(lower, b - sqrt(theta / p / q1 / q2))
                upper = min(upper, b + sqrt(theta / p / q1 / q2))
                if (x1 + x2 < 1) and (u >= x1) and (u <= x2) and (u < 1 - x2):
                    p1 = x2 * (1 - x2) / (1 - u) + (1 - x2) * x2 / u
                    b1 = (1 - x2) * x2 / ((1 - x2) * x2 + x2 * (1 - x2) * u / (1 - u))
                    q1 = (1 - x2 - u) / (1 - x2 - x1)
                    reg = theta - p1 * (1 - q1) * q2 * (0.5 - b1) * (0.5 - b1)
                    lower = max(lower, b - sqrt(reg / p / q1 / q2))
                    upper = min(upper, b + sqrt(reg / p / q1 / q2))
                        
                elif (x1 + x2 > 1) and (u >= x1) and (u <= x2) and (u > 1 - x1):
                    p1 = x1 * (1 - x1) / (1 - u) + (1 - x1) * x1 / u
                    b1 = (1 - x1) * x1 / ((1 - x1) * x1 + x1 * (1 - x1) * u / (1 - u))
                    q2 = (u - (1 - x1)) / (x2 - (1 - x1))
                    reg = theta - p1 * q1 * (1 - q2) * (0.5 - b1) * (0.5 - b1)
                    lower = max(lower, b - sqrt(reg / p / q1 / q2))
                    upper = min(upper, b + sqrt(reg / p / q1 / q2))
            w1 = x1 + x2
            w2 = 2 - x1 - x2
            if with_lower_upper:
                return (w2 * lower + w1 * upper) / (w1 + w2), lower, upper
            else:
                return (w2 * lower + w1 * upper) / (w1 + w2)

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