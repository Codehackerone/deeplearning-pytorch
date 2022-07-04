import torch

x = torch.arange(9)

x_3x3 = x.view(3, 3)
print(x_3x3)
x_3x3 = x.reshape(3, 3)

# diff-> view works on contiguous

y = x_3x3.t()
## print(y.view()) # error

# use reshape

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

z = x1.view(-1)  # unroll/flatten
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)  # -1 on rest

# switch index

z = x.permute(0, 2, 1)  # T special case of permute

x = torch.arange(10)
print(x.unsqueeze(0).shape)  # {1,10}

x = torch.arange(10).unsqueeze(0).unsqueeze(0)
print(x.shape)
