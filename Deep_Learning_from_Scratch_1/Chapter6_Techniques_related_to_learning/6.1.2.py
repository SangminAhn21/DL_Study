# 6.1.2 확률적 경사 하강법(SGD)


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= grads[key] * self.lr
