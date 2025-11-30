import torch
import torch.nn.functional as F

def binary_quantized(x, nbits):
    """ Convert integer quantized weights to binary representation."""
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    assert x.dim() == 2

    x_int = x.to(torch.int32)
    mask = (1 << nbits) - 1
    x_int = x_int & mask

    bit_idx = torch.arange(nbits, device=x.device)
    bits01 = ((x_int.unsqueeze(-1) >> bit_idx) & 1).float()

    weights = (1 << bit_idx).float().clone()
    weights[-1] *= -1

    weights = weights.view(1, 1, nbits)

    binary_quantized_x = (bits01 * weights).permute(2, 0, 1)

    return binary_quantized_x

def main():
    nbits = 5

    x = torch.rand(2, 3, 3) # (B, N, I)
    x = x.reshape(-1, x.shape[-1])
    print('Shape of x:', x.shape)
    print(x)

    w_q = torch.tensor([[-3, 5, 7], [8, 15, -13], [4, -2, 6], [1, 2, 3]]).float()  # (O, I)

    ref_res = F.linear(x, w_q)
    print('Reference result shape:', ref_res.shape)

    w_q = binary_quantized(w_q, nbits)

    assert w_q.shape == (nbits, 4, 3)

    res = torch.einsum('nd, kcd->knc', x, w_q)
    res = res.sum(dim=0)

    assert torch.allclose(ref_res, res, atol=1e-6)
    print('No Chunk Test passed!')

    out = []
    k = 2
    num_chunks = 2
    for i in range(k):
        noisy_x = x.unsqueeze(-2).expand(-1, num_chunks, -1)
        out.append(torch.einsum('ibk, dbk->dib', noisy_x, w_q[:, i * num_chunks: (i+1) * num_chunks, :]))
        
    chunk_res = torch.cat(out, 2).sum(dim=0)
    print(res)
    print(chunk_res)
    
    assert torch.allclose(ref_res, chunk_res, atol=1e-6)
    print('Chunk Test passed!')


if __name__ == "__main__":
    main()