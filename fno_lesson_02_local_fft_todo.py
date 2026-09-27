"""FNO lesson 2: the local branch and the Fourier transform.

Goal
----
Build one educational FNO-style block that has two parallel paths:

    x -> local 1x1 convolution -----------+
                                           + -> GELU -> output
    x -> rFFT -> keep low modes -> irFFT --+

Important: the Fourier path in this lesson is only a non-learned low-pass
filter. In lesson 3, you will replace it with learned complex Fourier weights.

Rules
-----
1. Complete TODO 1 through TODO 10.
2. Replace the placeholder on each TODO with exactly one Python statement.
3. Do not modify anything below the automatic-tests line.
4. Do not hard-code BATCH_SIZE, GRID_SIZE, WIDTH, or MODES in a solution.
5. Run the entire file, then answer the five questions printed at the end.

Permitted PyTorch tools
-----------------------
Tensor.permute, nn.Conv1d, torch.fft.rfft, torch.zeros_like,
tensor slicing, torch.fft.irfft, and torch.nn.functional.gelu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


# B = functions in a batch, N = spatial points, WIDTH = hidden channels.
BATCH_SIZE = 4
GRID_SIZE = 128
WIDTH = 16
MODES = 12


# Construct four smooth periodic input functions rather than pure noise.
# grid shape: (B, N, 1)
grid = torch.arange(GRID_SIZE, dtype=torch.float32) / GRID_SIZE
grid = grid.view(1, GRID_SIZE, 1).repeat(BATCH_SIZE, 1, 1)

frequencies = torch.arange(
    1, BATCH_SIZE + 1, dtype=torch.float32
).view(BATCH_SIZE, 1, 1)

# u0 shape: (B, N, 1)
u0 = (
    torch.sin(2 * torch.pi * frequencies * grid)
    + 0.30 * torch.cos(2 * torch.pi * (frequencies + 2) * grid)
)

# At each x, the two input features are [u0(x), x].
# model_input shape: (B, N, 2)
model_input = torch.cat((u0, grid), dim=-1)


class LiftingLayer(nn.Module):
    """Completed lesson 1: lift two input features to WIDTH features."""

    def __init__(self, input_dimension: int, width: int):
        super().__init__()
        self.linear = nn.Linear(input_dimension, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def channels_last_to_first(x: torch.Tensor) -> torch.Tensor:
    """Change (B, N, C) into the Conv1d/FFT layout (B, C, N)."""

    # TODO 1: Reorder the three axes. Do not reshape or hard-code sizes.
    x.view(BATCH_SIZE, 2, GRID_SIZE)


class LocalBranch(nn.Module):
    """Pointwise channel mixing: R^WIDTH -> R^WIDTH at every x."""

    def __init__(self, width: int):
        super().__init__()

        # TODO 2: Store a width -> width Conv1d whose kernel has length one.
        self.local=nn.Conv1d(2, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 3: Apply the layer stored in self.local to x.
        return self.local(x)


class FourierLowPass1d(nn.Module):
    """Transform to Fourier space, retain low modes, then reconstruct."""

    def __init__(self, modes: int):
        super().__init__()
        self.modes = modes

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstructed signal, full spectrum, filtered spectrum."""

        # TODO 4: Compute the real FFT along the spatial (last) dimension.
        x_fourier = torch.zeroes(self.modes)
        x_fourier = torch.fft.rfft(x, dim=-1)

        if self.modes > x_fourier.size(-1):
            raise ValueError(
                f"modes={self.modes} exceeds the available "
                f"Fourier coefficients={x_fourier.size(-1)}"
            )

        # TODO 5: Create a zero tensor with the same shape, dtype, and device.
        filtered_fourier = torch.zeroes(BATCH_SIZE, WIDTH, self.modes, dtype= torch.float32)
        

        # TODO 6: Copy only the first self.modes coefficients into it.
        filtered_fourier[:,:] = x_fourier

        # TODO 7: Invert the real FFT. Preserve the original spatial length.
        reconstructed = torch.fft.ifft(x_fourier)
        

        return reconstructed, x_fourier, filtered_fourier


class StarterFNOBlock(nn.Module):
    """Parallel local and non-learned Fourier paths."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.local = LocalBranch(width)
        self.fourier = FourierLowPass1d(modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 8: Send x through the local branch.
        local_part = self.local(x)
        if local_part is None:
            raise NotImplementedError("TODO 8 is incomplete")

        # TODO 9: Send x through the Fourier branch and unpack its 3 outputs.
        fourier_part = self.fourier(x)
        if fourier_part is None:
            raise NotImplementedError("TODO 9 is incomplete")

        # TODO 10: Add both paths, apply GELU, and store the result.
        output = nn.GELU(local_part + fourier_part[1])
        if output is None:
            raise NotImplementedError("TODO 10 is incomplete")

        return output


# -----------------------------------------------------------------------------
# Automatic tests: do not modify anything below this line.
# -----------------------------------------------------------------------------

lifting = LiftingLayer(input_dimension=2, width=WIDTH)

# nn.Linear expects features last: (B, N, 2) -> (B, N, WIDTH).
hidden_channels_last = lifting(model_input)

# Conv1d and our FFT code expect spatial points last: (B, WIDTH, N).
hidden = channels_last_to_first(hidden_channels_last)

assert tuple(hidden.shape) == (BATCH_SIZE, WIDTH, GRID_SIZE), (
    "TODO 1 produced the wrong shape: "
    f"got {tuple(hidden.shape)}, expected {(BATCH_SIZE, WIDTH, GRID_SIZE)}."
)

# A second shape catches solutions that hard-code the lesson constants.
shape_probe = torch.randn(2, 31, 7)
shape_probe_result = channels_last_to_first(shape_probe)
assert tuple(shape_probe_result.shape) == (2, 7, 31), (
    "TODO 1 must work for arbitrary B, N, and C."
)


# Test the complete local branch.
local_branch = LocalBranch(WIDTH)
local_output = local_branch(hidden)

assert isinstance(local_branch.local, nn.Conv1d), (
    "TODO 2 must store an nn.Conv1d in self.local."
)
assert local_branch.local.kernel_size == (1,), (
    "The local convolution must have kernel_size=1."
)
assert tuple(local_branch.local.weight.shape) == (WIDTH, WIDTH, 1), (
    "The local weight tensor must have shape (WIDTH, WIDTH, 1)."
)
assert tuple(local_output.shape) == tuple(hidden.shape), (
    "The local branch must preserve (B, WIDTH, N)."
)

local_parameter_count = sum(
    parameter.numel() for parameter in local_branch.parameters()
)
assert local_parameter_count == WIDTH * WIDTH + WIDTH, (
    "A WIDTH -> WIDTH kernel-1 Conv1d needs WIDTH*WIDTH weights "
    "and WIDTH biases."
)


# Test the Fourier transform and low-mode reconstruction.
fourier_branch = FourierLowPass1d(MODES)
fourier_output, full_spectrum, filtered_spectrum = fourier_branch(hidden)

expected_spectrum_shape = (
    BATCH_SIZE,
    WIDTH,
    GRID_SIZE // 2 + 1,
)

assert tuple(full_spectrum.shape) == expected_spectrum_shape, (
    f"Wrong rFFT shape: got {tuple(full_spectrum.shape)}, "
    f"expected {expected_spectrum_shape}."
)
assert torch.is_complex(full_spectrum), "torch.fft.rfft must produce complex values."
assert tuple(filtered_spectrum.shape) == tuple(full_spectrum.shape), (
    "The filtered spectrum must have the same shape as the full spectrum."
)
assert filtered_spectrum.dtype == full_spectrum.dtype, (
    "The filtered spectrum must preserve the complex dtype."
)
assert torch.allclose(
    filtered_spectrum[..., :MODES],
    full_spectrum[..., :MODES],
), "The retained low modes do not match the original spectrum."
assert torch.count_nonzero(filtered_spectrum[..., MODES:]).item() == 0, (
    "Every Fourier coefficient after MODES must be zero."
)
assert tuple(fourier_output.shape) == tuple(hidden.shape), (
    "The Fourier path must return to the original spatial shape."
)
assert not torch.is_complex(fourier_output), (
    "torch.fft.irfft must reconstruct real values."
)

expected_low_pass = torch.fft.irfft(
    filtered_spectrum,
    n=hidden.size(-1),
    dim=-1,
)
assert torch.allclose(fourier_output, expected_low_pass), (
    "The reconstructed output is not the inverse FFT of the filtered spectrum."
)

# Full rFFT -> irFFT must reconstruct the original tensor.
full_round_trip = torch.fft.irfft(
    full_spectrum,
    n=hidden.size(-1),
    dim=-1,
)
round_trip_error = (full_round_trip - hidden).abs().max().item()
assert round_trip_error < 1e-5, (
    f"FFT round-trip error is too large: {round_trip_error:.3e}."
)

# Odd N catches an irfft call that forgets n=x.size(-1).
odd_input = torch.randn(2, 5, 127)
odd_branch = FourierLowPass1d(modes=9)
odd_output, _, _ = odd_branch(odd_input)
assert tuple(odd_output.shape) == tuple(odd_input.shape), (
    "For odd N, irfft must receive n=x.size(-1)."
)


# Test the combined educational block.
block = StarterFNOBlock(width=WIDTH, modes=MODES)
block_output = block(hidden)

assert tuple(block_output.shape) == tuple(hidden.shape), (
    "The block must preserve (B, WIDTH, N)."
)
assert torch.isfinite(block_output).all(), (
    "The block output contains NaN or infinity."
)

expected_block_output = F.gelu(
    block.local(hidden) + block.fourier(hidden)[0]
)
assert torch.allclose(block_output, expected_block_output), (
    "The block must return GELU(local_part + fourier_part)."
)

# A second block catches hard-coded WIDTH and N.
block_probe = StarterFNOBlock(width=7, modes=5)
block_probe_output = block_probe(torch.randn(2, 7, 30))
assert tuple(block_probe_output.shape) == (2, 7, 30), (
    "The block must work for arbitrary valid widths and spatial lengths."
)


retained_energy_ratio = (
    fourier_output.square().mean() / hidden.square().mean()
).item()

print("model input:       ", tuple(model_input.shape))
print("after lifting:      ", tuple(hidden_channels_last.shape))
print("channels first:     ", tuple(hidden.shape))
print("local output:       ", tuple(local_output.shape))
print("rFFT spectrum:      ", tuple(full_spectrum.shape))
print("spectrum dtype:     ", full_spectrum.dtype)
print("Fourier output:     ", tuple(fourier_output.shape))
print("block output:       ", tuple(block_output.shape))
print("local parameters:   ", local_parameter_count)
print("FFT round-trip err: ", f"{round_trip_error:.3e}")
print("retained energy:    ", f"{retained_energy_ratio:.3f}")
print("\nSUCCESS: lesson 2 is complete.")


print("\nAnswer these questions before lesson 3:")
print("1. Why does rFFT change N=128 into 65 Fourier coefficients?")
print("2. Why are Fourier coefficients complex numbers?")
print("3. Why must irFFT receive n=x.size(-1), especially when N is odd?")
print("4. Why is kernel_size=1 called the local or pointwise branch?")
print("5. What trainable object is still missing from this Fourier branch?")
