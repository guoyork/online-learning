from utils import Enumerator
from copy import deepcopy

class Base(object):
    def __init__(self, **args):
        self.args = args
        self.fix_args = deepcopy(args)
        self.allreports = self.calc()

    def get_end(self):
        pass

    def calc_report(self):
        pass
    
    def calc_prob(self):
        pass

    def calc_benchmark(self):
        pass
    
    def add_noise(self, noise):
        pass

    def del_noise(self):
        self.args = deepcopy(self.fix_args)

    def calc(self, noise=0.0):
        if noise > 0:
            self.add_noise(noise)
        enu = Enumerator(end=self.get_end())
        res = []
        while True:
            if self.calc_prob(enu.cur) > 0:
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

    def calc_loss(self, func, output=False):
        loss = 0
        if output:
            print("Calc Loss:", self.args)
        for report in self.allreports:
            aggregate = func(report["report"])
            loss += report["p"] * (aggregate - report["benchmark"]) * (aggregate - report["benchmark"])
            if output:
                print(report, aggregate, report["p"] * (aggregate - report["benchmark"]) * (aggregate - report["benchmark"]))
        if output:
            print(loss)
        return loss
