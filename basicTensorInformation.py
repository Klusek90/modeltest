
  ### ===========BASE INFO================
scalar = torch.tensor(7)
print(scalar)
#dimentions
print(scalar.ndim)

#get tensor back
print(scalar.item())

#vector (magnitued/ direction) -one dimentional tensor
vector= torch.tensor([7,7])
print(vector.shape)

#matix -two dimentional tensor
MATRIX = torch.tensor([[7,8],[9,10]])
print(MATRIX[1])

#TENSOR -any number of dimentions
TENSOR = torch.tensor([[[1,2,3],
                        [1,2,3],
                        [5,6,7]]])
print(TENSOR.ndim)

#random tensor
random_tens = torch.rand (3,4)
print(random_tens)
print(random_tens.ndim)

random_image =torch.rand(size=(3, 224,244)) #height ,width ,color changels (RGB)
print(random_image.ndim, random_image.shape)

# creating tensor witl all zeros --- they are in float format
zeros = torch.zeros(size=(3,4))
print(zeros)

ones = torch.ones([3,4])
print(ones)

#range
one_to_ten = torch.arange(1,11)
print(one_to_ten)

one_to_hundrend =  torch.arange(start=1, end=101, step=10)
print(one_to_hundrend)

#Tensor like
ten_zeros = torch.zeros_like(one_to_hundrend)
print(ten_zeros)