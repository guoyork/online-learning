from models.coins import Coins, CoinsOrder2
from models.binary import Binary
from online_learning import OnlineLearning
from utils import Enumerator
import numpy as np

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
    M = 10
    experts = []
    enu = Enumerator(end=(M, M, M, M, M))
    while True:
        cur = enu.cur
        if (cur[1] < cur[0]) and (cur[2] >= cur[0]) and (cur[3] < cur[0]) and (cur[4] >= cur[0]) and (cur[0] > 0) and (cur[0] < M):
            experts.append(Binary(mu=cur[0] / M, reports=[[cur[1] / M, cur[2] / M], [cur[3] / M, cur[4] / M]]))
        if not enu.step():
            break

    def map2inputs(repo):
        return int(repo[0] * M + 0.5) * (M + 1) + int(repo[1] * M + 0.5)
    learner = OnlineLearning(experts, map2inputs, (M + 1) * (M + 1))
    func = learner.train(eta=4, N=500)

    myfunc = np.zeros((M + 1, M + 1), dtype=int)
    mydiff = np.zeros((M + 1, M + 1), dtype=int)
    for i in range(M + 1):
        for j in range(M + 1):
            myfunc[i][j] = int(func[map2inputs((i / M, j / M))] * 100 + 0.5)
            mydiff[i][j] = int(func[map2inputs((i / M, j / M))] * 100 + 0.5) - (i * 10 + j * 10) / 2
    print(myfunc)
    print(mydiff)

if __name__ == "__main__":
    # coins_order_1()
    # coins_order_2()
    binary_order_1()