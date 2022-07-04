import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)
z = x + y

# Subtraction

z = x - y

# Division

z = torch.true_divide(x, y)  # element wise if equal shape

# inplace ops (_ means inplace) (computationally efficient)

t = torch.zeros(3)
t.add_(x)
t += x

# Exponentiation

z = x.pow(2)
z = x ** 2
print(z)

# Simple Comparisons

z = x > 0
z = x < 0

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)

x3 = x1.mm(x2)

# Matrix exponentiation

matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)

# Element wise multiplication
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
# print(out_bmm)

# Example of broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2  # row will be expanded automatically
z = x1 ** x2

# Other useful operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)  # x.max(dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)

z = torch.argmax(x, dim=0)  # Returns position of max
z = torch.argmin(x, dim=0)

mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)  # false

sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10)  # check all elements less than 0 and makes it 0 (max and min)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)  # any true

z = torch.all(x)  # all true
