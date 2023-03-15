from math import sqrt
import numpy as np
from utils import *
from models.binary import Binary
mu = (3 - sqrt(5)) / 4
def BinaryAggregator(x, N=1000, theta=(5 * sqrt(5) - 11) / 8, with_lower_upper=False):
    x1 = min(x[0], x[1])
    x2 = max(x[0], x[1])
    if (x1 == 0) or (x2 == 0):
        if with_lower_upper:
            return .0, .0, .0
        else:
            return .0
    elif (x1 == 1) or (x2 == 1):
        if with_lower_upper:
            return 1., 1., 1.
        else:
            return 1.
    else:
        lower = 0
        upper = 1
        for uu in range(0, N + 1):
            u = uu / N
            if uu == 0:
                u = mu
            if uu == N:
                u = 1 - mu
            p = (1 - x1) * (1 - x2) / (1 - u) + x1 * x2 / u
            b = x1 * x2 / (x1 * x2 + (1 - x1) * (1 - x2) * u / (1 - u))
            q1 = u / x1 if x1 >= u else (1 - u) / (1 - x1)
            q2 = u / x2 if x2 >= u else (1 - u) / (1 - x2)
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
            
        # print(x, upper, lower)
        upper = min(upper, max(x1, x2))
        lower = max(lower, min(x1, x2))
        return (lower + upper) / 2


def BinaryAggregatorBinarySearch(x, N=1000, theta=(5 * sqrt(5) - 11) / 8, with_lower_upper=False):
    x1 = min(x[0], x[1])
    x2 = max(x[0], x[1])
    if (x1 == 0) or (x2 == 0):
        if with_lower_upper:
            return .0, .0, .0
        else:
            return .0
    elif (x1 == 1) or (x2 == 1):
        if with_lower_upper:
            return 1., 1., 1.
        else:
            return 1.
    else:
        lower = 0
        upper = 1
        for uu in range(0, N + 1):
            u = uu / N
            if uu == 0:
                u = mu
            if uu == N:
                u = 1 - mu
            p = (1 - x1) * (1 - x2) / (1 - u) + x1 * x2 / u
            b = x1 * x2 / (x1 * x2 + (1 - x1) * (1 - x2) * u / (1 - u))
            q1 = u / x1 if x1 >= u else (1 - u) / (1 - x1)
            q2 = u / x2 if x2 >= u else (1 - u) / (1 - x2)
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
            
        # print(x, upper, lower)
        upper = min(upper, max(x1, x2))
        lower = max(lower, min(x1, x2))
        return (lower + upper) / 2

        # return (lower + upper) / 2
        # return min(x1, x2)
        # if (x1 <= 0.5) and (x2 <= 0.5):
        #     upper = min(upper, (x1 + x2) / 2)
        # if (x1 >= 0.5) and (x2 >= 0.5):
        #     lower = max(lower, (x1 + x2) / 2)
        
        # if ((x1 >= 0.19) and (x1 <= 0.81)) or ((x2 >= 0.19) and (x2 <= 0.81)):
        #     return upper
        # return lower
        # return np.random.uniform(lower, upper)
        # return (x1 + x2) / 2       
        # return lower
        w1 = x1 + x2
        w2 = 2 - x1 - x2
        if with_lower_upper:
            return (w2 * lower + w1 * upper) / (w1 + w2), lower, upper
        else:
            return (w2 * lower + w1 * upper) / (w1 + w2)

def BinaryAggregatorTest(x, N=1000, theta=(5 * sqrt(5) - 11) / 8, eps=1e-9):
    lis = [[0, 1, 1 - x[0], x[1], 1 - x[1]], [0, 1, 1 - x[0], 1 - x[1]]]
    enu = Enumerator(end=(len(lis[0]) - 1, len(lis[1]) - 1))
    lower = min(x[0], x[1])
    upper = max(x[0], x[1])
    while True:
        cur = np.array(enu.cur)
        report1 = [min(x[0], lis[0][cur[0]]), max(x[0], lis[0][cur[0]])]
        report2 = [min(x[1], lis[1][cur[1]]), max(x[1], lis[1][cur[1]])]
        L = int(max(report1[0], report2[0]) * N)
        R = int(min(report1[1], report2[1]) * N)
        # print(report1, report2, L, R)
        for uu in range(L, R + 1):
            u = uu / N
            expert = Binary(mu=u, reports=[report1, report2])
            if expert.args != None:
                benchmark = 0
                p = 0
                regret = theta
                flag = True
                for report in expert.allreports:
                    if (report["report"] == x) or (report["report"] == [x[1], x[0]]):
                        benchmark = report["benchmark"]
                        p += report["p"]
                    elif (report["report"] == [1 - x[1], 1 - x[0]]) or (report["report"] == [1 - x[0], 1 - x[1]]):
                        p += report["p"]
                    elif (report["report"][0] == 0) or (report["report"][1] == 0) or (report["report"][0] == 1) or (report["report"][1] == 1):
                        pass
                    elif (abs(sum(report["report"]) - 1) < eps):
                        regret -= report["p"] * (report["benchmark"] - 0.5) * (report["benchmark"] - 0.5)
                    elif (abs(report["report"][0] - report["report"][1]) < eps):
                        regret -= report["p"] * (report["benchmark"] - report["report"][0]) * (report["benchmark"] - report["report"][0])
                    else:
                        flag = False
                if (flag) and (p > 0):
                    # print(expert.args)
                    lower = max(lower, benchmark - sqrt(regret / p))
                    upper = min(upper, benchmark + sqrt(regret / p))
        if not enu.step():
            break
    return (lower + upper) / 2
