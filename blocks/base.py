from utils import Enumerator
import numpy as np
class BaseBlock(object):
    def __init__(self, bounds, construct_expert, **args):
        self.bounds = bounds
        self.args = args
        self.construct_expert = construct_expert
        self.gen_experts(**args)
        self.get_experts()

    def gen_experts(self, fixed=False, **args):
        bounds = self.bounds
        experts = []
        if fixed:
            enu = Enumerator(end=[1 for i in range(len(bounds))])
            valid = {}
            while True:
                cur = enu.cur
                param = [bounds[i][cur[i]] for i in range(len(bounds))]
                # print(param)
                expert = self.construct_expert(param)
                if expert.args != None:
                    experts.append(expert)
                    valid.update({cur: len(experts) - 1})
                if not enu.step():
                    break
            self.valid = valid
        else:
            for i in range(20):
                param = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
                # param = [bounds[i][1 - np.random.randint(0, 1)] for i in range(len(bounds))]
                expert = self.construct_expert(param)
                if expert.args != None:
                    experts.append(expert)
        self.experts = experts

    def get_experts(self):
        if len(self.experts) > 1:
            self.expert = [self.experts[np.random.randint(0, len(self.experts) - 1)]]
        elif len(self.experts) == 1:
            self.expert = self.experts
        else:
            self.expert = []
        return self.expert

    def zoom_in(self, end=None, losses=None):
        bounds = self.bounds
        if end == None:
            end = [1 for i in range(len(bounds))]
        enu = Enumerator(end=end)
        res_blocks = []

        if losses != None:
            res_losses = []
            cnt = 0

        while True:
            cur = enu.cur
            new_bounds = [(bounds[i][1] * cur[i] / (end[i] + 1) + bounds[i][0] * (end[i] + 1 - cur[i]) / (end[i] + 1),
                bounds[i][1] * (cur[i] + 1) / (end[i] + 1) + bounds[i][0] * (end[i] - cur[i]) / (end[i] + 1)) for i in range(5)]
            block = BaseBlock(new_bounds, self.construct_expert)
            res_blocks.append(block)
            if losses != None:
                if cur in self.valid:
                    res_losses.append(losses[self.valid[cur]])
                else:
                    res_losses.append(0)
            if not enu.step():
                break
        if losses != None:
            return res_blocks, res_losses
        else:
            return res_blocks
