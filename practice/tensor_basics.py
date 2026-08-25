import torch


# 1. Creating a tensor from plain Python data — like a nested list
data = [[1,2],[5,4]]
x = torch.tensor(data)
print("tensor from list: \n",x)
print("Shape: ", x.shape)
print("Dtype: ", x.dtype)

# 2. Creating tensors with built-in generators
zeros = torch.zeros(2,3)
ones = torch.ones(2,3)
rand = torch.rand(2,3)
print("\n Zeros: \n", zeros)
print("ones: \n",ones)
print("random:\n",rand)

# 3. Moving a tensor to the GPU — this is the operation that matters most for us
print("\n Before move, device: ",x.device)
x_gpu = x.to("cuda")
print("After move, device:", x_gpu.device)

# 4. Basic math — element-wise operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0,5.0,6.0])
print("\na+b =",a+b)
print("a * b = ",a*b)
print("a @ b =", a@b)