def BinaryAggregator(x, N=10000, theta=0.02254248593736856025573354295705, with_lower_upper=False):
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
        for uu in range(1, N):
            u = uu / N
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
        w1 = x1 + x2
        w2 = 2 - x1 - x2
        if with_lower_upper:
            return (w2 * lower + w1 * upper) / (w1 + w2), lower, upper
        else:
            return (w2 * lower + w1 * upper) / (w1 + w2)
