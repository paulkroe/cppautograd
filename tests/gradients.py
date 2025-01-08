import torch

a = torch.tensor([[1.], [2.]], requires_grad=True)
b = torch.tensor([[4.], [1.]], requires_grad=True)

d = a.t() @ b

d.backward()
print("result:", d)
print("a.grad:", a.grad)
print("b.grad:", b.grad)
