from aggregators.binary import BinaryAggregator
from models.binary import Binary
from math import sqrt
from utils import Enumerator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import heatmap

def print_chart(path, data, labels=[]):
    fig, ax = plt.subplots()
    # X = np.linspace(0, 1, len(data[0]))
    for i in range(len(data)):
        # print(data[i], i)
        plt.plot(data[i][0], data[i][1], label=labels[i])
    #plt.plot(X, [data[i][len(X) - i - 1] for i in range(len(X))])
    #plt.plot(X, [data[i][i] for i in range(len(X))])
    fig.legend()
    fig.tight_layout()
    fig.savefig(path + ".png", bbox_inches='tight', dpi = 400)
    fig.savefig(path + ".pdf", bbox_inches='tight')
    
def search(L, R, func, eps=1e-9):
    # M = 40
    # if R - L < eps:
    #     return (L + R) / 2
    # mx = -1e9
    # mxu = 1
    # for u in range(1, M):
    #     v = func((u * L + (M - u) * R) / M)
    #     if v > mx:
    #         mx = v
    #         mxu = u
    # LL = (L * (mxu + 1) + (M - 1 - mxu) * R) / M
    # RR = (L * (mxu - 1) + (M + 1 - mxu) * R) / M
    # return search(LL, RR, func, eps)
    while (R - L) > eps:
        mu1 = (L * 2 + R) / 3
        mu2 = (L + R * 2) / 3
        v1 = func(mu1)
        v2 = func(mu2)
        if v1 < v2:
            L = mu1
        else:
            R = mu2
    return (L + R) / 2

def binarysearch(L, R, func, eps=1e-9):
    while (R - L) > eps:
        theta = (L + R) / 2
        v = func(theta)
        if v < 0:
            L = theta
        else:
            R = theta
    return R

def calc_mu_0(x1, x2, theta, mu):
    lower = -1e9
    upper = 1e9
    # lower = 0
    # upper = 1
    p = (1 - x1) * (1 - x2) / (1 - mu) + x1 * x2 / mu
    b = x1 * x2 / (x1 * x2 + (1 - x1) * (1 - x2) * mu / (1 - mu))
    q1 = mu / x1 if x1 >= mu else (1 - mu) / (1 - x1)
    q2 = mu / x2 if x2 >= mu else (1 - mu) / (1 - x2)
    # print(theta)
    # print(x1, x2, mu)
    # print(p, q1, q2)
    # return b - sqrt(theta / p / q1 / q2), b + sqrt(theta / p / q1 / q2)
    lower = max(lower, b - sqrt(theta / p / q1 / q2))
    upper = min(upper, b + sqrt(theta / p / q1 / q2))
    return lower, upper

def calc_mu_1(x1, x2, theta, mu):
    lower = -1e9
    upper = 1e9
    # lower = 0
    # upper = 1
    p = (1 - x1) * (1 - x2) / (1 - mu) + x1 * x2 / mu
    b = x1 * x2 / (x1 * x2 + (1 - x1) * (1 - x2) * mu / (1 - mu))
    q1 = mu / x1 if x1 >= mu else (1 - mu) / (1 - x1)
    q2 = mu / x2 if x2 >= mu else (1 - mu) / (1 - x2)
    if (x1 + x2 < 1) and (mu >= x1) and (mu <= x2) and (mu < 1 - x2):
        p1 = x2 * (1 - x2) / (1 - mu) + (1 - x2) * x2 / mu
        b1 = (1 - x2) * x2 / ((1 - x2) * x2 + x2 * (1 - x2) * mu / (1 - mu))
        q1 = (1 - x2 - mu) / (1 - x2 - x1)
        reg = theta - p1 * (1 - q1) * q2 * (0.5 - b1) * (0.5 - b1)
        if reg >= 0:
            delta = sqrt(reg / p / q1 / q2)
            lower = max(lower, b - delta)
            upper = min(upper, b + delta)
        else:
            # print(x1, x2, mu, theta)
            lower = 1e9
            upper = -1e9
        # else:
        #     delta = -sqrt(-reg / p / q1 / q2)
    return lower, upper

def calc_mu_2(x1, x2, theta, mu):
    lower = -1e9
    upper = 1e9
    # lower = 0
    # upper = 1
    p = (1 - x1) * (1 - x2) / (1 - mu) + x1 * x2 / mu
    b = x1 * x2 / (x1 * x2 + (1 - x1) * (1 - x2) * mu / (1 - mu))
    q1 = mu / x1 if x1 >= mu else (1 - mu) / (1 - x1)
    q2 = mu / x2 if x2 >= mu else (1 - mu) / (1 - x2)
    if (x1 + x2 > 1) and (mu >= x1) and (mu <= x2) and (mu > 1 - x1):
        p1 = x1 * (1 - x1) / (1 - mu) + (1 - x1) * x1 / mu
        b1 = (1 - x1) * x1 / ((1 - x1) * x1 + x1 * (1 - x1) * mu / (1 - mu))
        q2 = (mu - (1 - x1)) / (x2 - (1 - x1))
        reg = theta - p1 * q1 * (1 - q2) * (0.5 - b1) * (0.5 - b1)
        if reg >= 0:
            delta = sqrt(reg / p / q1 / q2)
            lower = max(lower, b - delta)
            upper = min(upper, b + delta)
        else:
            # print(x1, x2, mu, theta)
            lower = 1e9
            upper = -1e9
        # else:
        #     delta = -sqrt(-reg / p / q1 / q2)
        # lower = max(lower, b - delta)
        # upper = min(upper, b + delta)
       
    return lower, upper

def calc_theta(x1, x2, theta):
    lowermu_0 = search(0, 1, lambda mu: calc_mu_0(x1, x2, theta, mu)[0])
    uppermu_0 = search(0, 1, lambda mu: -calc_mu_0(x1, x2, theta, mu)[1])
    lowermu_1 = search(x1, min(1 - x2, x2), lambda mu: calc_mu_1(x1, x2, theta, mu)[0])
    uppermu_1 = search(x1, min(1 - x2, x2), lambda mu: -calc_mu_1(x1, x2, theta, mu)[1])
    lowermu_2 = search(max(x1, 1 - x1), x2, lambda mu: calc_mu_2(x1, x2, theta, mu)[0])
    uppermu_2 = search(max(x1, 1 - x1), x2, lambda mu: -calc_mu_2(x1, x2, theta, mu)[1])
    # print(lowermu, uppermu)
    # print(calc_mu(x1, x2, theta, lowermu)[0], calc_mu(x1, x2, theta, 3.485174545519908e-10)[0])
    # print(lowermu, uppermu)
    lower = max(calc_mu_0(x1, x2, theta, lowermu_0)[0], calc_mu_1(x1, x2, theta, lowermu_1)[0], calc_mu_2(x1, x2, theta, lowermu_2)[0])
    upper = min(calc_mu_0(x1, x2, theta, uppermu_0)[1], calc_mu_1(x1, x2, theta, uppermu_1)[1], calc_mu_2(x1, x2, theta, uppermu_2)[1])
    # lis = []
    # for mu in range(1, 100):
    #     lis.append(calc_mu(x1, x2, theta, mu / 100)[1])
    # print(lis)
    return lower, upper, upper - lower

def calc(x, theta=None):
    x1, x2 = x
    if x1 > x2:
        x2, x1 = x1, x2
    if (x1 * x2 == 0):
        # return 0
        return 0, 0, 0
    if ((1 - x1) * (1 - x2) == 0):
        # return 1
        return 1, 1, 0
    if theta == None:
        theta = binarysearch(0, 1, lambda theta: calc_theta(x1, x2, theta)[2])
    # print(theta)
    res = calc_theta(x1, x2, theta)
    # print(res)
    # if res[2] > 1e-3:
    #     print(res)
    # print(theta)
    # return (res[0] + res[1]) / 2
    return res[0], res[1], theta
    # return calc_theta(x1, x2, theta)
    


def calc_test(x, N=1000, theta=(5 * sqrt(5) - 11) / 8):
    x1, x2 = x
    lower = 0
    upper = 1
    U = []
    for uu in range(1, N):
        u = uu / N
        # if uu == 0:
        #     u = mu
        # if uu == N:
        #     u = 1 - mu
        U.append(u)
        p = (1 - x1) * (1 - x2) / (1 - u) + x1 * x2 / u
        b = x1 * x2 / (x1 * x2 + (1 - x1) * (1 - x2) * u / (1 - u))
        q1 = u / x1 if x1 >= u else (1 - u) / (1 - x1)
        q2 = u / x2 if x2 >= u else (1 - u) / (1 - x2)
        # lower.append(b - sqrt(theta / p / q1 / q2))
        # upper.append(b + sqrt(theta / p / q1 / q2))
        # lower = -1000
        # upper = 1000
        # lower = 0
        # upper = 1
        lower = max(lower, b - sqrt(theta / p / q1 / q2))
        upper = min(upper, b + sqrt(theta / p / q1 / q2))
        if (x1 + x2 < 1) and (u >= x1) and (u <= x2) and (u < 1 - x2):
            p1 = x2 * (1 - x2) / (1 - u) + (1 - x2) * x2 / u
            b1 = (1 - x2) * x2 / ((1 - x2) * x2 + x2 * (1 - x2) * u / (1 - u))
            q1 = (1 - x2 - u) / (1 - x2 - x1)
            reg = theta - p1 * (1 - q1) * q2 * (0.5 - b1) * (0.5 - b1)
            lower = max(lower, b - sqrt(reg / p / q1 / q2))
            upper = min(upper, b + sqrt(reg / p / q1 / q2))
        elif (x1 + x2 > 1) and (u >= x1) and (u <= x2) and (u > 1 - x1):
            p1 = x1 * (1 - x1) / (1 - u) + (1 - x1) * x1 / u
            b1 = (1 - x1) * x1 / ((1 - x1) * x1 + x1 * (1 - x1) * u / (1 - u))
            q2 = (u - (1 - x1)) / (x2 - (1 - x1))
            reg = theta - p1 * q1 * (1 - q2) * (0.5 - b1) * (0.5 - b1)
            lower = max(lower, b - sqrt(reg / p / q1 / q2))
            upper = min(upper, b + sqrt(reg / p / q1 / q2))
        # lowerlis.append(lower)
        # upperlis.append(upper)
    return lower, upper
    # print(x, upper, lower)
    # upper = min(upper, max(x1, x2))
    # lower = max(lower, min(x1, x2))
    # return (lower + upper) / 2


# def print_chart(path, data, labels=[]):
#     fig, ax = plt.subplots()
#     X = np.linspace(0, 1, len(data[0]))
#     if len(labels) == 0:
#         labels = range(len(data))
#     for i in range(len(data)):
#         # print(data[i], i)
#         plt.plot(data[i][0], data[i][1], label=labels[i])
#     #plt.plot(X, [data[i][len(X) - i - 1] for i in range(len(X))])
#     #plt.plot(X, [data[i][i] for i in range(len(X))])
#     # fig.legend()
#     fig.tight_layout()
#     fig.savefig(path + ".png", bbox_inches='tight', dpi = 400)
#     fig.savefig(path + ".pdf", bbox_inches='tight')

if __name__ == "__main__":
    # print(calc((0.02, 0.04)))
    # print(calc((0.04, 0.02)))
    # print(calc((0.97, 0.01)))
    # print(calc((0.03, 0.99)))
    # print(calc((0.01, 0.97)))
    # print(calc_theta(0.99, 0.03, theta=(5 * sqrt(5) - 11) / 8))
    # print(calc_theta(0.97, 0.01, theta=(5 * sqrt(5) - 11) / 8))
    # print(calc((0.02, 0.8)))
    #print(calc_theta(0.02, 0.8, 0.0224))
    # print(calc_theta(0.02, 0.52, 0.0206615738))
    # print(calc_theta(0.02, 0.52, 0.0225))

    N = 50
    func = np.zeros((N + 1, N + 1), dtype=float)
    lowerfunc = np.zeros((N + 1, N + 1), dtype=float)
    upperfunc = np.zeros((N + 1, N + 1), dtype=float)
    theta = np.zeros((N + 1, N + 1), dtype=float)
    for i in range(N + 1):
        for j in range(N + 1):
            res = calc((i / N, j / N))
            lowerfunc[i][j] = res[0] * 100
            upperfunc[i][j] = res[1] * 100
            if i + j <= N:
                func[i][j] = upperfunc[i][j]
            else:
                func[i][j] = lowerfunc[i][j]
            # func[i][j] = (lowerfunc[i][j] + upperfunc[i][j]) / 200
            theta[i][j] = res[2]
    heatmap.print_heatmap("Func", func, labels=[i * 100 / N for i in range(N + 1)], figsize=(20, 19))
    heatmap.print_heatmap("LowerFunc", lowerfunc, labels=[i * 100 / N for i in range(N + 1)], figsize=(20, 19))
    heatmap.print_heatmap("UpperFunc", upperfunc, labels=[i * 100 / N for i in range(N + 1)], figsize=(20, 19))
    mx = np.max(theta)
    print(mx)
    for i in range(N + 1):
        for j in range(N + 1):
            theta[i][j] = theta[i][j] / mx * 100
    heatmap.print_heatmap("Theta", theta, labels=[i * 100 / N for i in range(N + 1)], figsize=(20, 19))

    # np.savetxt("A.txt", func, fmt='%.10lf', delimiter=' ')
    # print(calc_theta(0.1, 0.1, theta=(5 * sqrt(5) - 11) / 8))
    # print(calc_theta(0.9997, 0.7346, theta=4e-8))
    # print(calc((0.9997, 0.7346)))
    # print(res)
    # 0.5946
    # 0.2518
    # 0.5994
    # 0.0694
    # 0.9333
    # 0.5615
    # 0.5613
    # 0.8788
    # 0.0852
    # 0.9231
    expert = Binary(mu=0.5946, reports=[[0.2518, 0.5994], [0.0694, 0.9333]])
    expert.calc_loss(BinaryAggregator, output=True)
    expert.calc_loss(calc, output=True)
    expert = Binary(mu=0.5615, reports=[[0.5613, 0.8788], [0.0852, 0.9231]])
    expert.calc_loss(BinaryAggregator, output=True)
    expert.calc_loss(calc, output=True)
    expert = Binary(mu=0.1666666666666666666666, reports=[[0, 0.2], [0.0, 0.7666666666666667]])
    expert.calc_loss(BinaryAggregator, output=True)
    expert.calc_loss(calc, output=True)
    expert = Binary(mu=0.8045, reports=[[0.5954, 0.9997], [0.3341, 0.9985]])
    expert.calc_loss(BinaryAggregator, output=True)
    expert.calc_loss(calc, output=True)
    expert = Binary(mu=0.7853, reports=[[0.6394, 0.9994], [0.3211, 0.9983]])
    expert.calc_loss(BinaryAggregator, output=True)
    expert.calc_loss(calc, output=True)
    expert = Binary(mu=0.7853, reports=[[0.6394, 0.9994], [0.3606, 1]])
    expert.calc_loss(BinaryAggregator, output=True)
    expert.calc_loss(calc, output=True)
    expert = Binary(mu=0.8170, reports=[[0.2654, 0.9997], [0.7346, 0.9997]])
    expert.calc_loss(BinaryAggregator, output=True)
    expert.calc_loss(calc, output=True)
    expert = Binary(mu=0.19098300562505257589770658281718, reports=[[0, .7], [0.1, .3]])
    expert.calc_loss(BinaryAggregator, output=True)
    expert.calc_loss(calc, output=True)
    expert = Binary(mu=0.19098300562505257589770658281718, reports=[[.1, .7], [.15, .3]])
    expert.calc_loss(BinaryAggregator, output=True)
    expert.calc_loss(calc, output=True)
    # print((res[0] + res[1]) / 2)
    # U = []
    # lower_0 = []
    # upper_0 = []
    # lower_1 = []
    # upper_1 = []
    # lower_2 = []
    # upper_2 = []
    # for u in range(1, 100):
    #     U.append(u / 100)
    #     lower_0.append(calc_mu_0(0.02, 0.52, 0.005857553333044052, u / 100)[0])
    #     upper_0.append(calc_mu_0(0.02, 0.52, 0.005857553333044052, u / 100)[1])
    #     lower_1.append(calc_mu_1(0.02, 0.52, 0.005857553333044052, u / 100)[0])
    #     upper_1.append(calc_mu_1(0.02, 0.52, 0.005857553333044052, u / 100)[1])
    #     lower_2.append(calc_mu_2(0.02, 0.52, 0.005857553333044052, u / 100)[0])
    #     upper_2.append(calc_mu_2(0.02, 0.52, 0.005857553333044052, u / 100)[1])
    # print_chart("B", [[U, lower_0], [U, upper_0], [U, lower_1], [U, upper_1], [U, lower_2], [U, upper_2]], ["lower_0", "upper_0", "lower_1", "upper_1", "lower_2", "upper_2"])
    # U, lower, upper = calc(0.1, 0.1)
    
# I draw a chart of all the discrete information structures with N=10 and the priors between 0 and 0.2, and the corresponding signaling probabilities
# Note that there are three large areas in the chart that do not have a corresponding discrete information structure