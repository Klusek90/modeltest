
# ========= tensor aggregation =====================
# x = torch.arange(1,100,10)

# print(x)
# print(torch.min(x))                       # x.min()
# print(torch.max(x))                       # x.max()
# print(torch.mean(x.type(torch.float32)))  # requiered to change to data type float
# print(torch.sum(x))

# print(x.argmin())  #minimum index value

## ========== reshaping , squeezing and unsqueezing
#
# * Reshape -reshape an input tensor to difine shape
# * View - return a view of a inpu tensror
# * Stacking - combine muliple tensors on top of each other
# * Squeeze - removes all '1' dimentions from basicTensorInformation
# * Unsqueeze - add a '1' dimentions to a target tensor
# * Permute - return a view of the input with dimentions

x = torch.arange(1.,10.)
# print(x, x.shape)
#
# # add extra dimention
reshape = x.reshape(1,9)
# print(reshape, reshape.shape)
#
# #change view -==share the same memory ==-
# z = x.view(1,9)
# print(z, z.shape)

#stack tensors on top of each other
# x_stack= torch.stack([x,x,x,x])
# print(x_stack)

#squeeze removing all single dimentions [1,9] -> [9]
# print(reshape.squeeze())
#
# #unsqueeze add single dimention to a target tensor
# print(f"previous target {reshape}")
# print(f"squieezed dimention {reshape.squeeze()}")
# print(f"unsqueezed dimention {reshape.unsqueeze(dim=0)}")
# print(f"other unsqueezed dimention {reshape.unsqueeze(dim=2)}")

##permute dimentions = change order od dimentions [1,2,3] -> [2,1,3]
x_orginal = torch.rand(size=(244,244,3))
x_permuted= x_orginal.permute(2,0,1)        #0->1 1->2 2->0
print(f"Previus shape {x_orginal.shape}")
print(f"New shape {x_permuted.shape}")