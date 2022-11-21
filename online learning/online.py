import numpy as np
from math import gamma


def equal(a, b):
    return 1 if (abs(a-b) < 1e-7) else 0


def weighted_sum(weight, x):
    return np.dot(weight, x)


class information_CI(object):
    def __init__(self, p, x1, y1, x2, y2):
        self.p = p
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.p1 = (p-y1)/(x1-y1)
        self.p2 = (p-y2)/(x2-y2)

    def prob(self, r1, r2):
        res1 = 0.0
        res2 = 0.0
        if equal(r1, self.x1):
            res1 = self.p1
        elif equal(r1, self.y1):
            res1 = 1-self.p1
        if equal(r2, self.x2):
            res2 = self.p2
        elif equal(r2, self.y2):
            res2 = 1-self.p2
        return res1*res2

    def opt(self, r1, r2):
        res = (1-self.p)*r1*r2/((1-self.p)*r1*r2+self.p*(1-r1)*(1-r2))
        return res


class information_general_coins(object):
    def __init__(self, p, m1, m2, x1, y1, x2, y2):
        self.p = p
        self.m0 = 1-m1-m2
        self.m1 = m1
        self.m2 = m2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.p1 = (p-y1)/(x1-y1)
        self.p2 = (p-y2)/(x2-y2)

    def legal(self):
        if self.x1 < self.m0*self.p/(self.m0+self.m1):
            return 0
        if self.x2 < self.m0*self.p/(self.m0+self.m2):
            return 0
        if self.y1 > (self.m0*self.p+self.m1)/(self.m0+self.m1):
            return 0
        if self.y2 > (self.m0*self.p+self.m2)/(self.m0+self.m2):
            return 0
        return 1

    def prob(self, r1, r2):
        res1 = 0.0
        res2 = 0.0
        if equal(r1, self.x1):
            res1 = self.p1
        elif equal(r1, self.y1):
            res1 = 1-self.p1
        if equal(r2, self.x2):
            res2 = self.p2
        elif equal(r2, self.y2):
            res2 = 1-self.p2
        return res1*res2

    def opt(self, r1, r2):
        res = (1-self.m2)*r1+(1-self.m1)*r2-self.m0*self.p
        return res


class online_learning(object):
    def __init__(self, interval=10, name="general_coins"):
        self.interval = interval
        self.information_list = []
        if name == "CI":
            for p in range(interval):
                for x1 in range(p):
                    for y1 in range(p+1, interval):
                        for x2 in range(p):
                            for y2 in range(p+1, interval):
                                self.information_list.append(information_CI(p/interval, x1/interval, y1/interval, x2/interval, y2/interval))
        elif name == "general_coins":
            for p in range(1, interval):
                for m1 in range(1, interval):
                    for m2 in range(1, interval-m1):
                        for x1 in range(p):
                            for y1 in range(p+1, interval):
                                for x2 in range(p):
                                    for y2 in range(p+1, interval):
                                        self.information_list.append(information_general_coins(p/interval, m1/interval, m2/interval, x1/interval, y1/interval, x2/interval, y2/interval))

        self.nums = len(self.information_list)
        self.weight = np.random.random(self.nums)
        self.weight /= np.sum(self.weight)
        self.losses = np.zeros(self.nums)

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
            for (r1, r2) in [(cur.x1, cur.x2), (cur.x1, cur.y2), (cur.y1, cur.x2), (cur.y1, cur.y2)]:
                opts[int(r1*self.interval)][[int(r2*self.interval)]] += cur.opt(r1, r2)*self.weight[i]*cur.prob(r1, r2)
                normalizations[int(r1*self.interval)][[int(r2*self.interval)]] += self.weight[i]*cur.prob(r1, r2)
        normalizations = np.maximum(1e-7, normalizations)

        opt_aggregations = np.zeros((self.interval, self.interval))
        for i in range(self.interval):
            for j in range(self.interval):
                opt_aggregations[i][j] = opts[i][j]/normalizations[i][j]
                #opt_aggregations[i][j] = self.opt_aggregation(i/self.interval, j/self.interval, opts[i][j]/normalizations[i][j])
        np.savetxt("aggregation function.txt", opt_aggregations)

        loss = np.zeros(self.nums)
        for i in range(self.nums):
            cur = self.information_list[i]
            for (r1, r2) in [(cur.x1, cur.x2), (cur.x1, cur.y2), (cur.y1, cur.x2), (cur.y1, cur.y2)]:
                loss[i] += self.loss_fun(cur.opt(r1, r2), opt_aggregations[int(r1*self.interval)][[int(r2*self.interval)]])*cur.prob(r1, r2)
                #loss[i] += self.loss_fun(cur.opt(r1, r2), self.my_fun(r1, r2))*cur.prob(r1, r2)
        return loss

    def update_weight(self):
        eta = 1
        new_weight = np.exp(eta*self.losses)
        self.weight = new_weight/np.sum(new_weight)

    def train(self, N=10000):
        for i in range(N):
            loss = self.cal_loss()
            if i % 10 == 0:
                print("epoch "+str(i)+": ", weighted_sum(self.weight, loss))
            self.losses += loss
            self.update_weight()
            np.savetxt("weight.txt",self.weight)


if __name__ == "__main__":
    model = online_learning(interval=10)
    model.train()
    # average 0.0811
    # lower bound 0.0488
