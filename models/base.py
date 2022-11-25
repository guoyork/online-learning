from utils import Enumerator

class Base(object):
    def __init__(self, **args):
        self.args = args
        self.allreports = self.calc()

    def get_end(self):
        pass

    def calc_report(self):
        pass
    
    def calc_prob(self):
        pass

    def calc_benchmark(self):
        pass

    def calc(self):
        enu = Enumerator(end=self.get_end())
        res = []
        while True:
            res.append({
                "report": self.calc_report(enu.cur),
                "p": self.calc_prob(enu.cur),
                "benchmark": self.calc_benchmark(enu.cur),
            })
            if not enu.step():
                break
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
