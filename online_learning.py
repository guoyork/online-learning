import numpy as np

class OnlineLearning(object):
    def build(self, noise=0.0):
        for i in range(len(self.experts)):
            if noise > 0:
                self.experts[i].calc(noise)
            for report in self.experts[i].allreports:
                if report["p"] > 0:
                    repo = np.array(report["report"])
                    idx = self.map2inputs(repo)
                    # print(repo, idx)
                    self.func2experts[idx].append((i, report["p"], report["benchmark"]))
                    self.experts2func[i].append((idx, report["p"], report["benchmark"]))

    def __init__(self, experts, map2inputs, len_inputs):
        self.eps = 1e-10
        self.experts = experts
        self.map2inputs = map2inputs
        self.losses = np.zeros((len(experts)), dtype=float)
        self.update_weight()
        self.func2experts = [[] for i in range(len_inputs)]
        self.experts2func = [[] for i in range(len(experts))]
        self.build()

    def calc_func(self):
        func = np.zeros((len(self.func2experts)), dtype=float)
        for i in range(len(self.func2experts)):
            psum = .0
            for report in self.func2experts[i]:
                func[i] += report[1] * self.weight[report[0]] * report[2]
                # print(report[1] * self.weight[report[0]])
                psum += report[1] * self.weight[report[0]]
            # print(psum)
            # assert(psum > 0)
            func[i] /= max(self.eps, psum)
        return func

    def calc_loss(self):
        loss = np.zeros((len(self.experts2func)), dtype=float)
        for i in range(len(self.experts2func)):
            for report in self.experts2func[i]:
                loss[i] += report[1] * (self.func[report[0]] - report[2]) * (self.func[report[0]] - report[2])
        return loss

    def update_weight(self, eta=20):
        weight = np.exp(eta * self.losses)
        self.weight = weight / np.sum(weight)

    def train(self, N=10000, eta=40, info_epoch=100, noise=0.0):
        minloss = [1]
        for i in range(1, N + 1):
            if noise > 0:
                self.build(noise=noise)
            self.func = self.calc_func()
            loss = self.calc_loss()
            if i % info_epoch == 0:
                print("Epoch #"+str(i)+": ", np.sum(loss * self.weight), ",", np.max(loss))
            self.loss = loss
            self.losses = np.maximum(self.losses + (i + N / 5) / N * (loss - np.sum(loss * self.weight)), 0)
            self.update_weight(eta=eta)

    
    def output_topweighted(self, k=10, output=False):
        f = [i for i in range(len(self.weight))]
        f.sort(key=lambda x: -self.weight[x])
        for i in range(k):
            print(self.experts[f[i]].args, self.weight[f[i]], self.loss[f[i]])
            self.experts[f[i]].calc_loss(lambda x: self.func[self.map2inputs(x)], output=output)