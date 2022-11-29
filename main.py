from models.coins import Coins, CoinsOrder2
from models.binary import Binary, BinaryOrder2, BinaryOrder2IID
from online_learning import OnlineLearning
from utils import Enumerator
import numpy as np
import heatmap
import charts

def coins_order_1():
    experts = []
    enu = Enumerator(end=(10, 10, 10, 10))
    while True:
        cur = enu.cur
        experts.append(Coins(alpha=(cur[0] + 1) / 4, beta=(cur[1] + 1) / 4, m=(cur[2], cur[3])))
        if not enu.step():
            break

    M = 100
    def map2inputs(repo):
        return int(repo[0] * M + 0.5) * (M + 1) + int(repo[1] * M + 0.5)
    learner = OnlineLearning(experts, map2inputs, (M + 1) * (M + 1))
    learner.train(N=1000)

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

def calc_aggregator_for_binary_order_1(M, N, map2inputs):
    
    upperfunc = np.array([1.0 for i in range((M + 1) * (M + 1))])
    lowerfunc = np.array([0.0 for i in range((M + 1) * (M + 1))])
    func = np.array([0.0 for i in range((M + 1) * (M + 1))])
    theta = 0.02254248593736856025573354295705
    for i in range(M + 1):
        for j in range(M + 1):
            x1 = min(i / M, j / M)
            x2 = max(i / M, j / M)
            if (x1 == 0) or (x2 == 0):
                func[map2inputs((i / M, j / M))] = 0
                lowerfunc[map2inputs((i / M, j / M))] = upperfunc[map2inputs((i / M, j / M))] = 0
            elif (x1 == 1) or (x2 == 1):
                func[map2inputs((i / M, j / M))] = 1
                lowerfunc[map2inputs((i / M, j / M))] = upperfunc[map2inputs((i / M, j / M))] = 1
            else:
                lower = 0
                upper = 1
                bsum = 0
                weightsum = 0
                for uu in range(1, N):
                    u = uu / N
                    p = (1 - x1) * (1 - x2) / (1 - u) + x1 * x2 / u
                    b = x1 * x2 / (x1 * x2 + (1 - x1) * (1 - x2) * u / (1 - u))
                    q1 = u / x1 if x1 >= u else (1 - u) / (1 - x1)
                    q2 = u / x2 if x2 >= u else (1 - u) / (1 - x2)
                    lower = max(lower, b - np.sqrt(theta / p / q1 / q2))
                    upper = min(upper, b + np.sqrt(theta / p / q1 / q2))
                    if (x1 + x2 < 1) and (u >= x1) and (u <= x2) and (u < 1 - x2):
                        p1 = x2 * (1 - x2) / (1 - u) + (1 - x2) * x2 / u
                        b1 = (1 - x2) * x2 / ((1 - x2) * x2 + x2 * (1 - x2) * u / (1 - u))
                        q1 = (1 - x2 - u) / (1 - x2 - x1)
                        reg = theta - p1 * (1 - q1) * q2 * (0.5 - b1) * (0.5 - b1)
                        # print(x1, x2, u, reg)
                        lower = max(lower, b - np.sqrt(reg / p / q1 / q2))
                        upper = min(upper, b + np.sqrt(reg / p / q1 / q2))
                            
                    elif (x1 + x2 > 1) and (u >= x1) and (u <= x2) and (u > 1 - x1):
                        p1 = x1 * (1 - x1) / (1 - u) + (1 - x1) * x1 / u
                        b1 = (1 - x1) * x1 / ((1 - x1) * x1 + x1 * (1 - x1) * u / (1 - u))
                        q2 = (u - (1 - x1)) / (x2 - (1 - x1))
                        reg = theta - p1 * q1 * (1 - q2) * (0.5 - b1) * (0.5 - b1)
                        lower = max(lower, b - np.sqrt(reg / p / q1 / q2))
                        upper = min(upper, b + np.sqrt(reg / p / q1 / q2))
                w1 = x1 + x2
                w2 = 2 - x1 - x2
                # w1 = np.sqrt(x1 * x2)
                # w2 = np.sqrt((1 - x1) * (1 - x2))
                # w1 = 1 / (lower + 1e-10) / (1 - lower + 1e-10)
                # w2 = 1 / (upper + 1e-10) / (1 - upper + 1e-10)
                # func[map2inputs((i / M, j / M))] = bsum / (N - 1)
                lowerfunc[map2inputs((i / M, j / M))] = lower
                upperfunc[map2inputs((i / M, j / M))] = upper
                func[map2inputs((i / M, j / M))] = (w1 * upper + w2 * lower) / (w1 + w2)
            # if (x1 == 0) or (x2 == 0):
            #     func[map2inputs((i / M, j / M))] = 0
            #     lowerfunc[map2inputs((i / M, j / M))] = upperfunc[map2inputs((i / M, j / M))] = 0
            # elif (x1 == 1) or (x2 == 1):
            #     func[map2inputs((i / M, j / M))] = 1
            #     lowerfunc[map2inputs((i / M, j / M))] = upperfunc[map2inputs((i / M, j / M))] = 1
            # else:
            #     lower = 0
            #     upper = 1
            #     for uu in range(1, N):
            #         u = uu / N
            #         p = (1 - x1) * (1 - x2) / (1 - u) + x1 * x2 / u
            #         b = x1 * x2 / (x1 * x2 + (1 - x1) * (1 - x2) * u / (1 - u))
            #         q1 = u / x1 if x1 >= u else (1 - u) / (1 - x1)
            #         q2 = u / x2 if x2 >= u else (1 - u) / (1 - x2)
            #         lower = max(lower, b - np.sqrt(theta / p / q1 / q2))
            #         upper = min(upper, b + np.sqrt(theta / p / q1 / q2))
            #     lowerfunc[map2inputs((i / M, j / M))] = lower
            #     upperfunc[map2inputs((i / M, j / M))] = upper
            #     func[map2inputs((i / M, j / M))] = (lower * (2 - x1 - x2) / 2 + upper * (x1 + x2) / 2)

    return func, upperfunc, lowerfunc

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
    learner.train(N=200)
    minloss, minweight, func = learner.loss, learner.weight, learner.func
    learner.output_topweighted()


def binary_order_1_3agents():
    M = 10
    experts = []
    enu = Enumerator(end=(M, M, M, M, M, M, M))
    while True:
        cur = enu.cur
        if (cur[1] < cur[0]) and (cur[2] >= cur[0]) and (cur[3] < cur[0]) and (cur[4] >= cur[0]) and (cur[5] < cur[0]) and (cur[6] >= cur[0]) and (cur[0] > 0) and (cur[0] < M):
            experts.append(Binary(mu=cur[0] / M, reports=[[cur[1] / M, cur[2] / M], [cur[3] / M, cur[4] / M], [cur[5] / M, cur[6] / M]]))
        if not enu.step():
            break

    def map2inputs(repo):
        return int(repo[0] * M + 0.5) * (M + 1) * (M + 1) + int(repo[1] * M + 0.5) * (M + 1) + int(repo[2] * M + 0.5)

    learner = OnlineLearning(experts, map2inputs, (M + 1) * (M + 1) * (M + 1))
    learner.train(N=1000, info_epoch=10, eta=10)
    minloss, minweight, func = learner.loss, learner.weight, learner.func
    learner.output_topweighted()

def save_binary_function(M=100, N=10000):
    def map2inputs(repo):
        return int(repo[0] * M + 0.5) * (M + 1) + int(repo[1] * M + 0.5)
    func, upperfunc, lowerfunc = calc_aggregator_for_binary_order_1(M, N, map2inputs)
    lower = np.zeros((M + 1, M + 1), dtype=int)
    upper = np.zeros((M + 1, M + 1), dtype=int)
    myfunc = np.zeros((M + 1, M + 1), dtype=int)
    pnasfunc = np.zeros((M + 1, M + 1), dtype=int)
    realfunc = np.zeros((M + 1, M + 1), dtype=float)
    for i in range(M + 1):
        for j in range(M + 1):
            realfunc[i][j] = func[map2inputs((i / M, j / M))]
            lower[i][j] = int(lowerfunc[map2inputs((i / M, j / M))] * 100 + 0.5)
            upper[i][j] = int(upperfunc[map2inputs((i / M, j / M))] * 100 + 0.5)
            myfunc[i][j] = int(func[map2inputs((i / M, j / M))] * 100 + 0.5)
            if (i > 0) and (j > 0) and (i < M) and (j < M):
                pnasfunc[i][j] = int(((i / M) * (j / M)) / ((i / M) * (j / M) + (i + j) / M / 2 * (1 - i / M) * (1 - j / M) / (1 - (i + j) / M / 2)) * 100 + 0.5)
    heatmap.print_heatmap("LowerFunction", lower, labels=[i * 100 / M for i in range(M + 1)], figsize=(14, 13))
    heatmap.print_heatmap("UpperFunction", upper, labels=[i * 100 / M for i in range(M + 1)], figsize=(14, 13))
    heatmap.print_heatmap("MyFunction", myfunc, labels=[i * 100 / M for i in range(M + 1)], figsize=(14, 13))
    heatmap.print_heatmap("PNASFunction", pnasfunc, labels=[i * 100 / M for i in range(M + 1)], figsize=(14, 13))
    np.savetxt("A.txt", realfunc, fmt='%.10lf', delimiter=',')
    np.savetxt("B.txt", realfunc, fmt='%.10lf', delimiter='\t')
    np.savetxt("C.txt", realfunc, fmt='%.10lf', delimiter=' ')
    np.savetxt("D.txt", realfunc, fmt='%.0lf', delimiter=',')

def binary_order_1_test(y = 0.5):
    M = 100
    Y = [[0, y], [y, 1]]
    experts = []
    enu = Enumerator(end=(M, M, M, 1))

    while True:
        cur = enu.cur
        if ((cur[1] < cur[0]) and ((cur[2] > cur[0]) or ((cur[2] == cur[0]) and (cur[1] == 0))) and (cur[0] > 0) and (cur[0] < M)
            and ((cur[1] <= 1) or (cur[2] >= M - 1)) and (cur[0] / M > Y[cur[3]][0]) and (cur[0] / M <= Y[cur[3]][1])):
            experts.append(Binary(mu=cur[0] / M, reports=[Y[cur[3]], [cur[1] / M, cur[2] / M]]))
            # print(experts[-1].args)
        if not enu.step():
            break

    def map2inputs(repo):
        return (int(repo[0] * M + 0.5) * (M + 1) + int(repo[1] * M + 0.5))

    learner = OnlineLearning(experts, map2inputs, (M + 1) * (M + 1))
    learner.train(N=2000)
    # dic = {}
    # for expert in experts:
    #     for report in expert.allreports:
    #         if map2inputs(report["report"]) not in dic:
    #             learner.func[map2inputs(report["report"])] = Binary.myfunc(report["report"])
    #             dic.update({map2inputs(report["report"]): 0})
    # learner.func, upperfunc, lowerfunc = calc_aggregator_for_binary_order_1(M, 2000, map2inputs)
    minloss = learner.calc_loss()
    print(np.max(minloss))
    # minloss, minweight, func = learner.loss, learner.weight, learner.func

    myfunc = np.zeros((M + 1, M + 1), dtype=float)
    
    for j in range(M + 1):
        func, lower, upper = Binary.myfunc((y, j / M), with_lower_upper=True)
        myfunc[0][j] = lower
        myfunc[1][j] = func
        myfunc[int(y * M + 0.5)][j] = learner.func[map2inputs((y, j / M))]
        #myfunc[M - 1][j] = R
        myfunc[M][j] = upper
    
    charts.print_chart("Only y", myfunc, Xlis=[0, 1, int(y * M + 0.5), M])
    # for i in range(len(experts)):
    #     if minweight[i] > 0.01:
    #         print(minweight[i], 1 / minweight[i], minloss[i], i)
    #         print(experts[i].args["mu"], experts[i].args["reports"])
    #         print(experts[i].calc_report((0, 0)))
    #         print(experts[i].calc_report((0, 1)))
    #         print(experts[i].calc_report((1, 0)))
    #         print(experts[i].calc_report((1, 1)))

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
    learner.train(N=1000)
    minloss, minweight, func = learner.loss, learner.weight, learner.func
    learner.output_topweighted(output=True)

def binary_order_2_iid(noise=0.0):
    M = 30
    N = 900
    experts = []
    enu = Enumerator(end=(M, M, M, N / M, N / M))
    while True:
        cur = enu.cur
        if (cur[0] < cur[1]) and (cur[1] < cur[2]) and (cur[3] < cur[4]):
            model = BinaryOrder2IID(reports=[[cur[0] / M, cur[1] / M, cur[2] / M] for i in range(2)], 
                E_0=[cur[3] / M for i in range(2)], E_1=[cur[4] / M for i in range(2)])
            if model.args != None:
                experts.append(model)
        if not enu.step():
            break
    # print(dic)
    # print("Rinidaba")
    def map2inputs(repo):
        # print(repo)
        return int(repo[0] * M + 0.5) * N + int(repo[2] * N + 0.5)

    learner = OnlineLearning(experts, map2inputs, (M + 1) * (N + 1))
    learner.train(N=1000, info_epoch=10)
    minloss, minweight, func = learner.loss, learner.weight, learner.func
    learner.output_topweighted(output=True)
if __name__ == "__main__":
    # model = BinaryOrder2IID(reports=[[0.2, 0.8, 1.0], [0.2, 0.8, 1.0]], pred=[[0.3, 0.5], [0.3, 0.5]])
    # print(model.args)
    # print(model.allreports)
    # save_binary_function(M=20,N=10000)
    #save_binary_function(M=200, N=20000)
    # binary_order_1_3agents()
    # coins_order_1()
    # coins_order_2()
    binary_order_2_iid()
    # binary_order_1()
    # binary_order_1_test(y=0.7)
    # binary_order_2()
    # binary_order_1_fix_one_side()
    # func = np.loadtxt("A.txt", delimiter=',')
    # print(func)
    # charts.print_chart("MyFunctionCharts", func)
    # loss = 0
    # expert = Binary(mu=0.2666666666666666, reports=[[0.0, 0.266666666666666666], [0.03333333333333333333333, 0.86666666666666666666666]])
    # expert.calc_loss(Binary.myfunc, output=True)
    # expert = Binary(mu=0.1666666666666666666666, reports=[[0, 0.2], [0.0, 0.7666666666666667]])
    # expert.calc_loss(Binary.myfunc, output=True)
    # expert = Binary(mu=0.8045, reports=[[0.5954, 0.9997], [0.3341, 0.9985]])
    # expert.calc_loss(Binary.myfunc, output=True)
    # expert = Binary(mu=0.7853, reports=[[0.6394, 0.9994], [0.3211, 0.9983]])
    # expert.calc_loss(Binary.myfunc, output=True)
    # expert = Binary(mu=0.7853, reports=[[0.6394, 0.9994], [0.3606, 1]])
    # expert.calc_loss(Binary.myfunc, output=True)
    # expert = Binary(mu=0.8170, reports=[[0.2654, 0.9997], [0.7346, 0.9997]])
    # expert.calc_loss(Binary.myfunc, output=True)
    # expert = Binary(mu=0.19098300562505257589770658281718, reports=[[0, .7], [0.1, .3]])
    # expert.calc_loss(Binary.myfunc, output=True)
    # expert = Binary(mu=0.19098300562505257589770658281718, reports=[[.1, .7], [.15, .3]])
    # expert.calc_loss(Binary.myfunc, output=True)
    # print(loss)