import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def print_chart(path, data, Xlis=[]):
    fig, ax = plt.subplots()
    X = np.linspace(0, 1, len(data[0]))
    if len(Xlis) == 0:
        Xlis = range(len(data))
    for i in Xlis:
        print(data[i], i)
        plt.plot(X, data[i], label=str(int(100 * X[i] + 0.5)))
    #plt.plot(X, [data[i][len(X) - i - 1] for i in range(len(X))])
    #plt.plot(X, [data[i][i] for i in range(len(X))])
    fig.legend()
    fig.tight_layout()
    fig.savefig(path + ".png", bbox_inches='tight', dpi = 400)
    fig.savefig(path + ".pdf", bbox_inches='tight')