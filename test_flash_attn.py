import torch
import torch.nn.functional as F

q = torch.randn(16, 32, 64, device='cuda')
k = torch.randn(16, 32, 64, device='cuda')
v = torch.randn(16, 32, 64, device='cuda')

output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
print(output)
