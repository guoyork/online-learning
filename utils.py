
class Enumerator(object):
    def __init__(self, start=None, end=(0, 0)):
        self.end = end
        if start == None:
            self.cur = tuple([0 for i in range(len(end))])
        else:
            self.cur = tuple(start)

    def step(self):
        x = len(self.end) - 1
        while (x >= 0) and (self.cur[x] >= self.end[x]):
            x = x - 1
        if x < 0:
            return False
        self.cur = self.cur[:x] + (self.cur[x] + 1,) + tuple([0 for i in range(len(self.end) - x - 1)])
        return True
