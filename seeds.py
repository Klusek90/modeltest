##====== Reproducbility (trying to take random out of random)===

# * reduomness in neural netrowks and PyTorch - conept of random seed

# random_tensor_A = torch.rand(3,4)
# random_tensor_B = torch.rand(3,4)
#
# print(random_tensor_A)
# print(random_tensor_B)
# print(random_tensor_A == random_tensor_B)

##making random but reproducable tensors

RANDOM_SEED =42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)

#random seed work only for one block of code. (if removed it wont work)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3,4)

print(random_tensor_D)
print(random_tensor_C)
print(random_tensor_D == random_tensor_C)