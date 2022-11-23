from models.coins import Coins, CoinsOrder2
from models.binary import Binary, BinaryOrder2
from online_learning import OnlineLearning
from utils import Enumerator
import numpy as np
import heatmap

def coins_order_1():
    experts = []
    enu = Enumerator(end=(6, 6, 6, 6))
    while True:
        cur = enu.cur
        experts.append(Coins(alpha=(cur[0] + 1) / 2, beta=(cur[1] + 1) / 2, m=(cur[2], cur[3])))
        if not enu.step():
            break

    M = 100
    def map2inputs(repo):
        return int(repo[0] * M + 0.5) * (M + 1) + int(repo[1] * M + 0.5)
    learner = OnlineLearning(experts, map2inputs, (M + 1) * (M + 1))
    learner.train()

def coins_order_2():
    experts = []
    enu = Enumerator(end=(6, 6, 6, 6))
    while True:
        cur = enu.cur
        experts.append(CoinsOrder2(alpha=(cur[0] + 1) / 2, beta=(cur[1] + 1) / 2, m=(cur[2], cur[3])))
        if not enu.step():
            break

    M = 20
    def map2inputs(repo):
        return (int(repo[0] * M + 0.5) * (M + 1) * (M + 1) * (M + 1) + int(repo[1] * M + 0.5) * (M + 1) * (M + 1)
            + int(repo[2] * M + 0.5) * (M + 1) + int(repo[3] * M + 0.5))
    
    learner = OnlineLearning(experts, map2inputs, (M + 1) * (M + 1) * (M + 1) * (M + 1))
    learner.train()

def binary_order_1():
    M = 20
    experts = []
    enu = Enumerator(end=(M, M, M, M, M))
    while True:
        cur = enu.cur
        if ((cur[1] < cur[0]) and ((cur[2] > cur[0]) or ((cur[2] == cur[0]) and (cur[1] == 0)))
            and (cur[3] < cur[0]) and ((cur[4] > cur[0]) or ((cur[4] == cur[0]) and (cur[3] == 0))) and (cur[0] > 0) and (cur[0] < M)):
            experts.append(Binary(mu=cur[0] / M, reports=[[cur[1] / M, cur[2] / M], [cur[3] / M, cur[4] / M]]))
        if not enu.step():
            break

    def map2inputs(repo):
        return int(repo[0] * M + 0.5) * (M + 1) + int(repo[1] * M + 0.5)
    learner = OnlineLearning(experts, map2inputs, (M + 1) * (M + 1))
    learner.train(N=2000)
    minloss, minweight, func = learner.loss, learner.weight, learner.func
    for i in range(len(experts)):
        if minloss[i] > 0.02249:
            print(minweight[i], 1 / minweight[i], minloss[i], i)
            print(experts[i].args["mu"], experts[i].args["reports"])
    myfunc = np.zeros((M + 1, M + 1), dtype=int)
    pnasfunc = np.zeros((M + 1, M + 1), dtype=int)
    myprior = np.zeros((M + 1, M + 1), dtype=int)
    for i in range(M + 1):
        for j in range(M + 1):
            myfunc[i][j] = int(func[map2inputs((i / M, j / M))] * 100 + 0.5)
            
            x1 = i / M
            x2 = j / M
            u = (x1 + x2) / 2
            if (x1 * x2 * (1 - u) + (1 - x1) * (1 - x2) * u) == 0:
                pnasfunc[i][j] = int(min(x1, x2) * 100 + 0.5)
            else:
                pnasfunc[i][j] = int(x1 * x2 * (1 - u) / (x1 * x2 * (1 - u) + (1 - x1) * (1 - x2) * u) * 100 + 0.5)

            u = func[map2inputs((i / M, j / M))]
            if (x1 * x2 * (1 - u) + (1 - x1) * (1 - x2) * u) == 0:
                myprior[i][j] = int(min(x1, x2) * 100 + 0.5)
            else:
                myprior[i][j] = int(x1 * x2 * (1 - u) / (x1 * x2 * (1 - u) + (1 - x1) * (1 - x2) * u) * 100 + 0.5)
    heatmap.print_heatmap("MyFunction", myfunc, labels=[i * 100 / M for i in range(M + 1)])
    heatmap.print_heatmap("PNASFunction", pnasfunc, labels=[i * 100 / M for i in range(M + 1)])
    # heatmap.print_heatmap("PNASFunction49", pnasfunc49, labels=[i * 100 / M for i in range(M + 1)])
    heatmap.print_heatmap("MyPrior", myprior, labels=[i * 100 / M for i in range(M + 1)])
    print(myfunc)
    print(pnasfunc)

def binary_order_2():
    M = 10
    experts = []
    enu = Enumerator(end=(M, M, M, M, M))
    while True:
        cur = enu.cur
        if (cur[1] < cur[0]) and (cur[2] >= cur[0]) and (cur[3] < cur[0]) and (cur[4] >= cur[0]) and (cur[0] > 0) and (cur[0] < M):
            experts.append(BinaryOrder2(mu=cur[0] / M, reports=[[cur[1] / M, cur[2] / M], [cur[3] / M, cur[4] / M]]))
        if not enu.step():
            break

    def map2inputs(repo):
        return (int(repo[0] * M + 0.5) * (M + 1) * (M + 1) * (M + 1) + int(repo[1] * M + 0.5) * (M + 1) * (M + 1)
            + int(repo[2] * M + 0.5) * (M + 1) + int(repo[3] * M + 0.5))

    learner = OnlineLearning(experts, map2inputs, (M + 1) * (M + 1) * (M + 1) * (M + 1))
    func = learner.train(eta=4, N=10000)

if __name__ == "__main__":
    # coins_order_1()
    # coins_order_2()
    binary_order_1()
    # binary_order_2()
    # binary_order_1_fix_one_side()