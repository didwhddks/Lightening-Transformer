import torch

from ops.quantize import bit_serialized

def test_basic_examples():
    nbits = 4

    # Simple small vector including negatives
    x = torch.tensor([-8, -3, -1, 0, 1, 2, 3, 7], dtype=torch.int32)

    # Non-optimized: original behavior
    bs_non_opt = bit_serialized(x, nbits, optimized=False)
    x_recon_non_opt = bs_non_opt.sum(dim=0)
    assert torch.equal(x_recon_non_opt, x), (
        f"Non-optimized reconstruction failed:\n"
        f"original:     {x}\n"
        f"reconstructed:{x_recon_non_opt}"
    )

    # Optimized: positive-format for magnitude, then sign
    bs_opt = bit_serialized(x, nbits, optimized=True)
    x_recon_opt = bs_opt.sum(dim=0)
    assert torch.equal(x_recon_opt, x), (
        f"Optimized reconstruction failed:\n"
        f"original:     {x}\n"
        f"reconstructed:{x_recon_opt}"
    )

    # Check the specific example from the description:
    # 4-bit, -3 should be decomposed as -2 + (-1) when optimized
    x_example = torch.tensor([-5], dtype=torch.int32)
    bs_example_opt = bit_serialized(x_example, nbits, optimized=True).squeeze(1)  # shape [nbits]

    # Bits are ordered as [2^0, 2^1, ..., 2^(nbits-2), negative_MSB]
    # For -3: magnitude 3 -> 2 + 1, then sign=-1 -> [-1, -2, 0, 0] which sums to -3
    print("Optimized bit decomposition for -5 (4-bit):", bs_example_opt)

    assert bs_example_opt.sum().item() == -5, "Sum of optimized decomposition for -3 is incorrect"

    print("test_basic_examples passed.")


def test_random_integers(nbits=4, num_samples=1000):
    qmin = -(1 << (nbits - 1))
    qmax = (1 << (nbits - 1)) - 1

    # Random integers in the valid quantization range
    x = torch.randint(low=qmin, high=qmax + 1, size=(num_samples,), dtype=torch.int32)

    # Non-optimized
    bs_non_opt = bit_serialized(x, nbits, optimized=False)
    x_recon_non_opt = bs_non_opt.sum(dim=0)
    assert torch.equal(x_recon_non_opt, x), "Non-optimized random reconstruction failed."

    # Optimized
    bs_opt = bit_serialized(x, nbits, optimized=True)
    x_recon_opt = bs_opt.sum(dim=0)
    assert torch.equal(x_recon_opt, x), "Optimized random reconstruction failed."

    print(f"test_random_integers (nbits={nbits}, num_samples={num_samples}) passed.")


if __name__ == "__main__":
    torch.manual_seed(0)

    test_basic_examples()
    test_random_integers(nbits=4, num_samples=1000)
    test_random_integers(nbits=8, num_samples=1000)

    print("All bit_serialized tests passed.")