from pytorch_memlab import profile
import torch 
import torch.nn as nn
import torch.nn.functional as F

@profile
def main():
    shape = (4, 8, 8620, 8)
    q = torch.randn(shape, dtype=torch.float32, device='cuda')
    k = torch.randn(shape, dtype=torch.float32, device='cuda')
    v = torch.randn(shape, dtype=torch.float32, device='cuda')
    attn_mask = torch.randint(0, 2, size=(8620, 8620), dtype=torch.bool, device='cuda')

    with torch.backends.cuda.sdp_kernel(
            enable_flash=False, 
            enable_math=False, 
            enable_mem_efficient=True
        ):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    target = torch.randn(out.shape, dtype=torch.float32, device='cuda')
    loss = F.mse_loss(out, target)
    loss.backward()


if __name__ == "__main__":
    main()