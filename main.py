from blocks.base import BaseBlock
from models.binary import Binary, BinaryOrder2IID
from online_learning import OnlineLearningForBlocks
from utils import Enumerator
import numpy as np

def binary_order_1():
    M = 100
    def construct_expert(params):
        return Binary(mu=params[0], reports=[[params[1], params[2]], [params[3], params[4]]])
    # expert = construct_expert([.5, .0, 1., .0, 1.])
    # print(expert.args)
    block = BaseBlock(bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], construct_expert=construct_expert, fixed=False)
    blocks = block.zoom_in([9, 9, 9, 9, 9])
    # for i in range(len(blocks)):
    #     for expert in blocks[i].experts:
    #         print(expert.args)

    def map2inputs(repo):
        return int(repo[0] * M + 0.5) * (M + 1) + int(repo[1] * M + 0.5)
    learner = OnlineLearningForBlocks(blocks, map2inputs, (M + 1) * (M + 1))
    learner.train(N=2000, info_epoch=100, thre=2)
    #minloss, minweight, func = learner.loss, learner.weight, learner.func
    learner.output_topweighted(output=True, k=20)

def binary_order_2_iid():
    M = 1000
    def construct_expert(params):
        return BinaryOrder2IID(reports=[[params[0], params[1], params[2]] for i in range(2)], E_0=[params[3], params[3]], E_1=[params[4], params[4]])
    # expert = construct_expert([.5, .0, 1., .0, 1.])
    # print(expert.args)
    block = BaseBlock(bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], construct_expert=construct_expert)
    blocks = block.zoom_in([4, 4, 4, 4, 4])
    # for i in range(len(blocks)):
    #     for expert in blocks[i].experts:
    #         print(expert.args)

    def map2inputs(repo):
        return int(repo[0] * M + 0.5) * (M + 1) + int(repo[2] * M + 0.5)
    learner = OnlineLearningForBlocks(blocks, map2inputs, (M + 1) * (M + 1))
    learner.train(N=2000, info_epoch=100, thre=0.01)
    #minloss, minweight, func = learner.loss, learner.weight, learner.func
    learner.output_topweighted(output=True, k=20)


def binary_order_2_iid(noise=0.0):
    M = 40
    N = 800
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

    binary_order_1()
    # binary_order_2_iid()