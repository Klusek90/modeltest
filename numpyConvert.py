
## from numpy to tensor
array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array)

print(f"from numpy: \n {array}   {tensor}")
##[1. 2. 3. 4. 5. 6. 7.] tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)
## by default is convered to float64 (long)

##from tensor to numpy
tensor =torch.ones(7)
numpy_tensor= tensor.numpy()
print(f"\n form tensor:\n {tensor}, {numpy_tensor}")