import numpy as np

class OnlineLearning(object):
    def __init__(self, experts, map2inputs, len_inputs):
        self.eps = 1e-10
        self.map2inputs = map2inputs
        self.losses = np.zeros((len(experts)), dtype=float)
        self.update_weight()
        self.func2experts = [[] for i in range(len_inputs)]
        self.experts2func = [[] for i in range(len(experts))]
        for i in range(len(experts)):
            for report in experts[i].allreports:
                repo = np.array(report["report"])
                idx = map2inputs(repo)
                # print(repo, idx)
                self.func2experts[idx].append((i, report["p"], report["benchmark"]))
                self.experts2func[i].append((idx, report["p"], report["benchmark"]))

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

    def train(self, N=10000, eta=20):
        minloss = 1
        for i in range(N):
            self.func = self.calc_func()
            loss = self.calc_loss()
            if i % 10 == 0:
                print("Epoch #"+str(i)+": ", np.sum(loss * self.weight), ",", np.max(loss))
                if np.max(loss) < minloss:
                    minloss = np.max(loss)
                    optfunc = self.func
                # print(np.argmax(loss), ", ", self.weight[np.argmax(loss)])
            # one_hot_loss = np.array([1 if i == argm else 0 for i in range(len(loss))])
            self.losses += loss

            # argm = np.argmax(loss)
            # self.losses[argm] += loss[argm]

            # self.losses += one_hot_loss
            self.update_weight(eta=eta)
            
        return optfunc