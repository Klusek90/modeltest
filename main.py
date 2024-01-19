import pandas as pd
import numpy as np
import torch

#===================== precision in computuing=======================

# * Different tensor type and devices
# * conversion


# float_32_tensor = torch.tensor([3.0,6.0,9.0],
#                                dtype=torch.float32,  #tensor type
#                                device=None,          #device tensor is run on (CPU, GPU, CUDA)
#                                requires_grad=False)  #whether or no track gradiens
#
# print(float_32_tensor.dtype)
#
# # convert tensor type
# float_16_tensor= float_32_tensor.type(torch.float16)
# print(float_16_tensor.dtype)
#
# print(float_16_tensor * float_32_tensor)
#
#
# some_tensor = torch.randn(3,4)
# print(some_tensor)
# print(f"Datatype od tensor: {some_tensor.dtype}")
# print(f"Shape of tensor {some_tensor.size()}")

#========== Manipulation===============

# Tensor operation include:
# * Adddition
# * Subtraction
# * Muliplication (element-wise)
# * ZeroDivisionError
# * Matrix muliplication

tensor = torch.tensor([1,2,3])
# print(tensor+10)
# print(tensor*10)
# print(tensor-10)

# double_tensor = torch.tensor([[1,2,3],[4,5,6]])
# print(double_tensor +10)
# print(double_tensor *10)
# print(double_tensor -10)

# ==Mutliplication of MARIX==
tensor = torch.tensor([1,2,3])
print(tensor*tensor)