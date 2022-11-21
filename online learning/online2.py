import numpy as np
from math import gamma


def equal(a, b):
    return 1 if (abs(a-b) < 1e-7) else 0


def weighted_sum(weight, x):
    return np.dot(weight, x)


class information_coins(object):
    def __init__(self, a, b, m1, m2):
        self.a = a
        self.b = b
        self.m1 = m1
        self.m2 = m2
        self.p = np.zeros((m1+1, m2+1))
        self.s = np.zeros((m1+1, m2+1))
        for i in range(m1):
            for j in range(m2):
                self.p[i][j] = self.calc_p(i, j)
                self.s[i][j] = self.calc_s(i, j)

    def calc_s(self, y1=None, y2=None):
        if y2 == None:
            return (self.a + y1) / (self.a + self.b + self.m1)
        elif y1 == None:
            return (self.a + y2) / (self.a + self.b + self.m2)
        return (self.a + y1 + y2) / (self.a + self.b + self.m1 + self.m2)

    def calc_p(self, y1=None, y2=None):
        a = self.a
        b = self.b
        m1 = self.m1
        m2 = self.m2
        if y2 == None:
            return gamma(a + b) * gamma(b + m1 - y1) * gamma(a + y1) / gamma(a) / gamma(b) / gamma(a + b + m1) * gamma(m1 + 1) / gamma(y1 + 1) / gamma(m1 - y1 + 1)
        elif y1 == None:
            return gamma(a + b) * gamma(b + m2 - y2) * gamma(a + y2) / gamma(a) / gamma(b) / gamma(a + b + m2) * gamma(m2 + 1) / gamma(y2 + 1) / gamma(m2 - y2 + 1)
        return gamma(a + b) * gamma(b + m1 + m2 - y1 - y2) * gamma(a + y1 + y2) / gamma(a) / gamma(b) / gamma(a + b + m1 + m2) * gamma(m1 + 1) / gamma(y1 + 1) / gamma(m1 - y1 + 1) * gamma(m2 + 1) / gamma(y2 + 1) / gamma(m2 - y2 + 1)


class online_learning(object):
    def __init__(self, interval=10, name="coins"):
        self.interval = 100
        self.information_list = []
        for alpha in range(1, interval):
            for beta in range(1, interval):
                for m1 in range(1, interval):
                    for m2 in range(1, interval):
                        self.information_list.append(information_coins(alpha/2, beta/2, m1, m2))

        self.nums = len(self.information_list)
        self.weight = np.random.random(self.nums)
        self.weight /= np.sum(self.weight)
        self.losses = np.zeros(self.nums)
        print("finish init")
        print("------------------------")

    def opt_aggregation(self, r1, r2, opt):
        if equal(r1, r2):
            return r1
        weight = (opt-r2)/(r1-r2)
        weight = min(max(weight, 0), 1)
        return weight*r1+(1-weight)*r2

    def loss_fun(self, x, y):
        return (x-y)**2

    def my_fun(self, r1, r2):
        return (r1+r2)/2

    def cal_loss(self):

        opts = np.zeros((self.interval, self.interval))
        normalizations = np.zeros((self.interval, self.interval))
        for i in range(self.nums):
            cur = self.information_list[i]
            for y1 in range(cur.m1):
                for y2 in range(cur.m2):
                    r1 = cur.calc_s(y1=y1)
                    r2 = cur.calc_s(y2=y2)
                    opts[int(r1*self.interval)][[int(r2*self.interval)]] += cur.s[y1][y2]*self.weight[i]*cur.p[y1][y2]
                    normalizations[int(r1*self.interval)][[int(r2*self.interval)]] += self.weight[i]*cur.p[y1][y2]
        normalizations = np.maximum(1e-7, normalizations)

        opt_aggregations = np.zeros((self.interval, self.interval))
        for i in range(self.interval):
            for j in range(self.interval):
                opt_aggregations[i][j] = opts[i][j]/normalizations[i][j]
                #opt_aggregations[i][j] = self.opt_aggregation(i/self.interval, j/self.interval, opts[i][j]/normalizations[i][j])
        np.savetxt("aggregation function coins.txt", opt_aggregations)

        loss = np.zeros(self.nums)
        for i in range(self.nums):
            cur = self.information_list[i]
            for y1 in range(cur.m1):
                for y2 in range(cur.m2):
                    r1 = cur.calc_s(y1=y1)
                    r2 = cur.calc_s(y2=y2)
                    loss[i] += self.loss_fun(cur.s[y1][y2], opt_aggregations[int(r1*self.interval)][[int(r2*self.interval)]])*cur.p[y1][y2]
                #loss[i] += self.loss_fun(cur.opt(r1, r2), self.my_fun(r1, r2))*cur.prob(r1, r2)
        return loss

    def update_weight(self):
        eta = 1e2
        new_weight = np.exp(eta*self.losses)
        self.weight = new_weight/np.sum(new_weight)

    def train(self, N=10000):
        for i in range(N):
            loss = self.cal_loss()
            if i % 10 == 0:
                print("epoch "+str(i)+": ", weighted_sum(self.weight, loss))
            self.losses += loss
            self.update_weight()


if __name__ == "__main__":
    model = online_learning(interval=20)
    model.train()
    # average 0.0811
