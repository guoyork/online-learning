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

    def __init__(self, experts, map2inputs, len_inputs, eta=40):
        self.eps = 1e-10
        self.eta = eta
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

    def update_weight(self):
        weight = np.exp(self.eta * self.losses)
        self.weight = weight / np.sum(weight)

    def train(self, N=10000, info_epoch=100, noise=0.0):
        self.allweights = []
        self.allfuncs = []
        minloss = [1]
        for i in range(1, N + 1):
            if noise > 0:
                self.build(noise=noise)
            self.func = self.calc_func()
            self.allfuncs.append(self.func)
            self.allweights.append(self.weight)
            loss = self.calc_loss()
            if i % info_epoch == 0:
                # print("Epoch #"+str(i)+": ", np.sum(loss * self.weight), ",", np.max(loss))
                print("Epoch #"+str(i))
                self.test()
            self.loss = loss
            # self.losses = np.maximum(self.losses + (i + N / 5) / N * (loss - np.sum(loss * self.weight)), 0)
            self.losses = self.losses + (loss - np.sum(loss * self.weight))
            self.update_weight()

    def test(self, weight=None, func=None, k=10, output=False):
        if weight != None:
            self.weight = weight
        else:
            self.weight = self.average(self.allweights)
        if func != None:
            self.func = func
        else:
            self.func = self.average(self.allfuncs)
        loss = self.calc_loss()
        print("Result: ", np.sum(loss * self.weight), ",", np.max(loss))
        self.output_topweighted(k=k, output=output)

    def average(self, alldata):
        res = np.zeros((len(alldata[0])), dtype=float)
        for i in range(len(alldata)):
            res += alldata[i]
        res /= len(alldata)
        return res
    
    def output_topweighted(self, k=10, output=False):
        f = [i for i in range(len(self.weight))]
        f.sort(key=lambda x: -self.weight[x])
        for i in range(k):
            print(self.experts[f[i]].args, self.weight[f[i]], self.loss[f[i]])
            self.experts[f[i]].calc_loss(lambda x: self.func[self.map2inputs(x)], output=output)


class OnlineLearningForBlocks(object):

    def build(self, get_expert=False):
        blocks = self.blocks
        losses = []
        experts = []
        from_blocks = []
        for i in range(len(blocks)):
            if get_expert:
                blocks[i]["block"].get_experts()
            for j in range(len(blocks[i]["block"].expert)):
                losses.append(blocks[i]["losses"])
                from_blocks.append(i)
                experts.append(blocks[i]["block"].expert[j])
        self.losses = losses
        self.experts = experts
        weight = np.exp(self.eta * np.array(self.losses))
        # print(np.sum(weight))
        self.weight = weight / np.sum(weight)
        self.from_blocks = from_blocks

    def update_losses(self):
        blocks = self.blocks
        cnt = 0
        for i in range(len(blocks)):
            for j in range(len(blocks[i]["block"].expert)):
                blocks[i]["losses"] = self.losses[cnt]
                cnt += 1

    def __init__(self, blocks, map2inputs, len_inputs, eta=40):
        self.eps = 1e-10
        self.eta = eta
        self.blocks = []
        for block in blocks:
            self.blocks.append({
                "block": block,
                "losses": .0
            })
        self.map2inputs = map2inputs
        self.len_inputs = len_inputs
        self.build()

    def calc_func(self):
        bench_sum = np.zeros((self.len_inputs), dtype=float)
        prob_sum = np.zeros((self.len_inputs), dtype=float)
        for i in range(len(self.experts)):
            expert = self.experts[i]
            for report in expert.allreports:
                if report["p"] > 0:
                    idx = self.map2inputs(report["report"])
                    bench_sum[idx] += report["benchmark"] * report["p"] * self.weight[i]
                    prob_sum[idx] += report["p"] * self.weight[i]
        func = np.zeros((self.len_inputs), dtype=float)
        for i in range(len(prob_sum)):
            if prob_sum[i] > 0:
                func[i] = bench_sum[i] / prob_sum[i]
        return func

    def calc_loss(self):
        loss = np.zeros((len(self.experts)), dtype=float)
        for i in range(len(self.experts)):
            expert = self.experts[i]
            for report in expert.allreports:
                if report["p"] > 0:
                    idx = self.map2inputs(report["report"])
                    loss[i] += report["p"] * (self.func[idx] - report["benchmark"]) * (self.func[idx] - report["benchmark"])
        self.loss = loss
        return loss

        

    def train(self, N=10000, info_epoch=100, thre=10):
        for i in range(1, N + 1):
            self.func = self.calc_func()
            loss = self.calc_loss()
            self.losses = np.maximum(self.losses + (loss - np.sum(loss * self.weight)), 0)
            # self.losses = self.losses + (loss - np.sum(loss * self.weight))
            self.update_losses()
            if i % info_epoch == 0:
                print("Epoch #"+str(i)+": ", np.sum(loss * self.weight), ",", np.max(loss), ",", np.max(self.weight), ",", len(self.weight))
            get_expert = (i % 1 == 0)
            self.build(get_expert)
            while (np.max(self.weight) > thre * (N / 2) / (N / 2 + i)) and (i < N / 2):
                t = np.argmax(self.weight)
                self.zoom_in(self.from_blocks[t])
                self.build()
                # print(self.weight)
                # assert(False)

    def zoom_in(self, id):
        block = self.blocks[id]
        blocks = block["block"].zoom_in()
        self.blocks = self.blocks[:id] + self.blocks[id + 1:]
        for i in range(len(blocks)):
            # print(np.exp(self.eta * losses[i]), np.exp(self.eta * (losses[i] - np.log(len(blocks[i].expert)) / self.eta)), len(blocks[i].expert))
            self.blocks.append({
                "block": blocks[i],
                "losses": block["losses"] - np.log(len(blocks)) / self.eta
            })
            # print(np.sum(np.exp(self.eta * np.array(self.blocks[-1]["losses"]))))
    
    def output_topweighted(self, k=10, output=False):
        f = [i for i in range(len(self.weight))]
        f.sort(key=lambda x: -self.weight[x])
        for i in range(k):
            print(self.experts[f[i]].args, self.weight[f[i]], self.loss[f[i]])
            self.experts[f[i]].calc_loss(lambda x: self.func[self.map2inputs(x)], output=output)

