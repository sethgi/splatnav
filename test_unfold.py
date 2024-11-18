#%%

"""How to use ``unfoldNd``. A comparison with ``torch.nn.Unfold``."""

# imports, make this example deterministic
import torch

import unfoldNd

torch.manual_seed(0)

# random batched RGB 32x32 image-shaped input tensor of batch size 64
inputs = torch.randn((1, 1, 32, 32, 32))

# module hyperparameters
robot_mask = torch.randn((5, 3, 3))
kernel_size = tuple(robot_mask.shape)
dilation = 1
padding = tuple( (torch.tensor(robot_mask.shape) - 1) // 2)
stride = 1

# both modules accept the same arguments and perform the same operation
# torch_module = torch.nn.Unfold(
#     kernel_size, dilation=dilation, padding=padding, stride=stride
# )
padding = tuple((torch.tensor(kernel_size) - 1) // 2)
lib_module = unfoldNd.UnfoldNd(
    kernel_size, dilation=dilation, padding=padding, stride=stride
)

# forward pass
#torch_outputs = torch_module(inputs)
lib_outputs = lib_module(inputs)

print(lib_outputs.shape)
# # check
# if torch.allclose(torch_outputs, lib_outputs):
#     print("✔ Outputs of torch.nn.Unfold and unfoldNd.UnfoldNd match.")
# else:
#     raise AssertionError("❌ Outputs don't match")