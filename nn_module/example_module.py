# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear1(x).clamp(min=0)
        return self.linear2(x)


if __name__ == '__main__':
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = Variable(torch.randn(N, D_in))
    y = Variable(torch.randn(N, D_out), requires_grad=False)
    model = TwoLayerNet(D_in, H, D_out)

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    num_epochs = 500
    for epoch in range(num_epochs):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        if epoch % 10 == 0:
            print(epoch, loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
