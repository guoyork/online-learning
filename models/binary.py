from models.base import Base
from math import gamma, sqrt, log
from random import uniform

from utils import Enumerator

class Binary(Base):
    def check_valid(self, err=1e-9):
        # print(self.args)
        if (self.args["mu"] <= err) or (self.args["mu"] >= 1 - err):
            return False
        for i in range(len(self.args["reports"])):
            if (abs(sum(self.args["prob"][i]) - 1) > err):
                return False
            for j in range(len(self.args["reports"][i])):
                if (self.args["reports"][i][j] < -err) or (self.args["reports"][i][j] > 1 + err):
                    return False
                if (self.args["prob"][i][j] < -err) or (self.args["prob"][i][j] > 1 + err):
                    return False
        return True
                
    def prob2mu(self, args):
        mu = .0
        for i in range(len(args["reports"][0])):
            mu += args["reports"][0][i] * args["prob"][0][i]
        return mu

    def mu2prob(self, args):
        mu = args["mu"]
        prob = []
        for i in range(len(args["reports"])):
            prob.append([abs(args["reports"][i][1] - mu) / abs(args["reports"][i][1] - args["reports"][i][0]),
                abs(mu - args["reports"][i][0]) / abs(args["reports"][i][1] - args["reports"][i][0])])
        return prob

    def __init__(self, **args):
        if "mu" in args:
            if "prob" not in args:
                args.update({"prob": self.mu2prob(args)})
        else:
            mu = self.prob2mu(args)
            if (mu <= 0) or (mu >= 1):
                self.args = None
                return
            args.update({"mu": mu})
        super().__init__(**args)

    def add_noise(self, noise):
        for i in range(len(self.args["reports"])):
            for j in range(len(self.args["reports"][i])):
                self.args["reports"][i][j] = max(min(self.args["reports"][i][j] + uniform(-noise, noise), 1), 0)
        self.args["mu"] = self.prob2mu(self.args)
        if self.args["mu"] == 0:
            self.del_noise()

    def get_end(self):
        return tuple([len(self.args["reports"][i]) - 1 for i in range(len(self.args["reports"]))])

    def calc_report(self, x):
        return tuple([self.args["reports"][i][x[i]] for i in range(len(self.args["reports"]))])
    
    def calc_prob(self, x):
        mu = self.args["mu"]
        p = 1
        p0 = 1 - mu
        p1 = mu
        for i in range(len(self.args["reports"])):
            p *= self.args["prob"][i][x[i]]
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
    def pred2E(self,args):
        E_0 = []
        E_1 = []
        n = len(args["reports"])
        for i in range(n):
            repo = args["reports"][i]
            pred = args["pred"][i]
            E_0.append((repo[1] * pred[0] - repo[0] * pred[1]) / ((1 - repo[0]) * repo[1] - (1 - repo[1]) * repo[0]))
            E_1.append(((1 - repo[1]) * pred[0] - (1 - repo[0]) * pred[1]) / ((1 - repo[1]) * repo[0] - (1 - repo[0]) * repo[1]))
        return E_0, E_1
    
    def E2muprob(self, args):
        prob = []
        E_0 = args["E_0"]
        E_1 = args["E_1"]
        n = len(args["reports"])
        for i in range(n):
            e0 = (sum(E_0) - E_0[i] * (n - 1)) / (n - 1)
            e1 = (sum(E_1) - E_1[i] * (n - 1)) / (n - 1)
            # if (e0 < min(args["reports"][i])) or (e1 > max(args["reports"][i])):
            #     return 0, 0
            if (1 - e1 + e0 == 0):
                return 0, 0
            mu = e0 / (1 - e1 + e0)
            if (mu <= 0) or (mu >= 1):
                return 0, 0
            repo = args["reports"][i]
            A = []
            B = []
            for i in range(len(repo)):
                A.append((1 - repo[i]) * repo[i] / (1 - mu))
                B.append(repo[i] * repo[i] / mu)
           
            p0 = ((e0 - A[2]) * (B[1] - B[2]) - (e1 - B[2]) * (A[1] - A[2])) / ((A[0] - A[2]) * (B[1] - B[2]) - (A[1] - A[2]) * (B[0] - B[2]))
            p1 = ((e0 - A[2]) * (B[0] - B[2]) - (e1 - B[2]) * (A[0] - A[2])) / ((A[1] - A[2]) * (B[0] - B[2]) - (A[0] - A[2]) * (B[1] - B[2]))
            # assert(abs((1 - repo[0]) * repo[0] * p0 / (1 - mu) + (1 - repo[1]) * repo[1] * p1 / (1 - mu) + (1 - repo[2]) * repo[2] * (1 - p0 - p1) / (1 - mu) - e0) < 1e-9)
            # assert(abs(repo[0] * repo[0] * p0 / mu + repo[1] * repo[1] * p1 / mu + repo[2] * repo[2] * (1 - p0 - p1) / mu - e1) < 1e-9)
            # assert(abs((A[0] - A[2]) * p0 + (A[1] - A[2]) * p1 - e0 + A[2]) < 1e-9)
            # assert(abs(B[0] * p0 + B[1] * p1 + B[2] * (1 - p0 - p1) - e1) < 1e-9)
            # assert(abs((B[0] - B[2]) * p0 + (B[1] - B[2]) * p1 - e1 + B[2]) < 1e-9)
            prob.append([p0, p1, 1 - p0 - p1])
        
        return mu, prob
            # (A[0] - A[2]) p[0] + (A[1] - A[2]) p[1] = e0 - A[2]
            # (B[0] - B[2]) p[0] + (B[1] - B[2]) p[1] = e1 - B[2]
            # (1 - repo[0]) * repo[0] * p[0] / (1 - mu) + (1 - repo[1]) * repo[1] * p[1] / (1 - mu) + (1 - repo[2]) * repo[2] * (1 - p[0] - p[1]) / (1 - mu) = e0
            # repo[0] * repo[0] * p[0] / (1 - mu) + repo[1] * repo[1] * p[1] / (1 - mu) + repo[2] * repo[2] * (1 - p[0] - p[1]) / (1 - mu) = e1
            # mu * e1 + (1 - mu) * e0 = mu
            # e0 = mu (1 - e1 + e0)

    def __init__(self, **args):
        if "pred" in args:
            E_0, E_1 = self.pred2E(args)
            args.update({
                "E_0": E_0,
                "E_1": E_1,
            })
        if "E_0" in args:
            mu, prob = self.E2muprob(args)
            if (mu <= 0) or (mu >= 1):
                self.args = None
                return
            args.pop("E_0")
            args.pop("E_1")
            args.update({"mu": mu})
            args.update({"prob": prob})
        super().__init__(**args)
        if not self.check_valid():
            # print(self.args)
            self.args = None
            return
        
    def calc_report(self, x):
        n = len(self.args["reports"])
        mu = self.args["mu"]
        order_1 = [self.args["reports"][i][x[i]] for i in range(len(self.args["reports"]))]
        E_0 = []
        E_1 = []
        for i in range(n):
            r = self.args["reports"][i]
            p = self.args["prob"][i]
            E_0.append(.0)
            E_1.append(.0)
            for j in range(len(r)):
                E_0[-1] += (1 - r[j]) * r[j] * p[j] / (1 - mu)
                E_1[-1] += r[j] * r[j] * p[j] / mu
        order_2 = []
        for i in range(n):
            order_2.append(order_1[i] * (sum(E_1) - E_1[i]) / (n - 1) + (1 - order_1[i]) * (sum(E_0) - E_0[i]) / (n - 1))

        return tuple(order_1) + tuple(order_2)

class BinaryOrder2IID(BinaryOrder2):
    def add_noise(self, noise):
        for j in range(len(self.args["reports"][0])):
            uniform_noise = uniform(-noise, noise)
            for i in range(len(self.args["reports"])):
                self.args["reports"][i][j] = max(min(self.args["reports"][i][j] + uniform_noise, 1), 0)
        self.args["mu"] = self.calc_mu(self.args)

    def calc(self, noise=0.0):
        if noise > 0:
            self.add_noise(noise)
        enu = Enumerator(end=self.get_end())
        res = []
        while True:
            flag = True
            for i in range(1, len(enu.cur)):
                if enu.cur[i] != enu.cur[i - 1]:
                    flag = False
                    break
            if (self.calc_prob(enu.cur) > 0) and flag:
                res.append({
                    "report": self.calc_report(enu.cur),
                    "p": self.calc_prob(enu.cur),
                    "benchmark": self.calc_benchmark(enu.cur),
                })
            if not enu.step():
                break
        if noise > 0:
            self.del_noise()
        # print(res)
        return res