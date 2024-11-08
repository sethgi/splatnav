#%%
import torch
import time
# Example 3D tensor of shape (H, W, C)
# H, W, C = 100, 100, 100
# D = 3
# tensor = torch.randn(H, W, C, D)

# # Fixed size of the slice for each dimension
# h_slice, w_slice, c_slice = 25, 25, 25

# B = 20000

# # Starting positions for each slice in (H, W, C) dimensions (B, 3)
# # Specify (start_h, start_w, start_c) for each slice
# start_positions = torch.randint(1, 20, (B, 3))

# # Create the indices for each dimension
# h_indices = torch.arange(h_slice).unsqueeze(0) + start_positions[:, 0].unsqueeze(1)  # Shape (B, h_slice)
# w_indices = torch.arange(w_slice).unsqueeze(0) + start_positions[:, 1].unsqueeze(1)  # Shape (B, w_slice)
# c_indices = torch.arange(c_slice).unsqueeze(0) + start_positions[:, 2].unsqueeze(1)  # Shape (B, c_slice)

# # Ensure indices are within bounds for each dimension
# h_indices = torch.clamp(h_indices, 0, H - 1)
# w_indices = torch.clamp(w_indices, 0, W - 1)
# c_indices = torch.clamp(c_indices, 0, C - 1)

# batch_indices = torch.arange(start_positions.size(0)).unsqueeze(1)  # Shape (B, 1)

# # Gather the slices using advanced indexing
# sliced_tensor = tensor[h_indices, :, :][batch_indices, :, w_indices, :][batch_indices, :, :, c_indices]

B = 200000
start_positions = torch.randint(1, 20, (B, 3))

X, Y, Z = torch.meshgrid(torch.arange(10), torch.arange(10), torch.arange(10))
points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

# Create the indices for each dimension
tnow = time.time()
torch.cuda.synchronize()
output = start_positions.unsqueeze(1) + points.unsqueeze(0)
torch.cuda.synchronize()
print('Time to slice:', time.time() - tnow)
raise
#%%