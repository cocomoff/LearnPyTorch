# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    )
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 500

for epoch in range(num_epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if epoch % 10 == 0:
        print(epoch, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
