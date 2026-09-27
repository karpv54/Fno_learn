"""Lesson 3: learn channel mixing at each retained Fourier mode.

Input and output: (B, C, N), real float32 tensors.
Weights: (C_in, C_out, M), complex64. Here C_in = C_out = width.
Replace each NotImplementedError with exactly one Python statement.
Do not change the automatic tests or hard-code the example dimensions.
Tools: nn.Parameter, torch.randn, torch.cfloat, torch.fft.rfft,
torch.zeros_like, slicing, torch.einsum, and torch.fft.irfft.
Run this file to check shapes, known signals, phase, and backpropagation.
"""

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        if width < 1 or modes < 1:
            raise ValueError('width and modes must be positive')
        self.width = width
        self.modes = modes
        scale = 1 / width

        # TODO 1: Set self.weights to an nn.Parameter containing scaled complex
        # random values, shape (width, width, modes). Use the supplied scale.
        self.weights = nn.Parameter(scale * torch.randn(width, width, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.width:
            raise ValueError('Expected input shape (B, width, N)')

        # TODO 2: Store the spatial-axis rFFT in x_fourier.
        x_fourier = torch.fft.rfft(x, dim=-1)
        if self.modes > x_fourier.size(-1):
            raise ValueError('modes exceeds the available rFFT coefficients')

        # TODO 3: Set output_fourier to zeros matching x_fourier.
        output_fourier = torch.zeros_like(x_fourier)

        # TODO 4: Set low_modes to the first self.modes entries of the last axis.
        low_modes = x_fourier[..., :self.modes]

        # TODO 5: Fill output_fourier[..., :self.modes] using low_modes and
        # self.weights. Sum input channels i; preserve batch b and mode k;
        # produce output channels o. Translate this into an einsum.
        output_fourier[..., :self.modes] = torch.einsum("ijk,jlk->ilk", low_modes, self.weights)

        # TODO 6: Return a real signal with the original number of spatial points.
        return torch.fft.irfft(output_fourier, n=x.shape[-1], dim=-1)


# Automatic tests: do not modify anything below this line.
def run_tests():
    torch.manual_seed(3)
    for n in (31, 64, 127):
        layer = SpectralConv1d(width=3, modes=7)
        x = torch.randn(2, 3, n)
        y = layer(x)
        assert y.shape == x.shape
        assert not y.is_complex()
        assert torch.isfinite(y).all()
        assert dict(layer.named_parameters())['weights'] is layer.weights
        assert layer.weights.shape == (3, 3, 7)
        assert layer.weights.is_complex()

    # With identity channel matrices at every mode, this becomes lesson 2.
    layer = SpectralConv1d(3, 7)
    with torch.no_grad():
        layer.weights.zero_()
        for channel in range(3):
            layer.weights[channel, channel, :] = 1
    spectrum = torch.fft.rfft(x)
    spectrum[..., 7:] = 0
    torch.testing.assert_close(layer(x), torch.fft.irfft(spectrum, n=x.size(-1)))

    # Known gain and channel routing: channel 0 -> channel 1, multiplied by 2.
    n = 64
    grid = torch.arange(n) / n
    sine = torch.sin(2 * torch.pi * 2 * grid)
    probe = torch.zeros(1, 2, n)
    probe[0, 0] = sine
    router = SpectralConv1d(2, 5)
    with torch.no_grad():
        router.weights.zero_()
        router.weights[0, 1, 2] = 2
    routed = router(probe)
    torch.testing.assert_close(routed[0, 0], torch.zeros(n), atol=2e-6, rtol=0)
    torch.testing.assert_close(routed[0, 1], 2 * sine, atol=2e-6, rtol=0)

    # Complex weights can change phase: +i times a cosine's coefficient -> -sin.
    probe[0, 0] = torch.cos(2 * torch.pi * 2 * grid)
    with torch.no_grad():
        router.weights[0, 1, 2] = 1j
    torch.testing.assert_close(router(probe)[0, 1], -sine, atol=2e-6, rtol=0)

    # A frequency outside the retained band disappears from this branch.
    probe[0, 0] = torch.sin(2 * torch.pi * 10 * grid)
    torch.testing.assert_close(router(probe), torch.zeros_like(probe), atol=2e-6, rtol=0)

    # The DC mode represents a constant. Its imaginary part cannot affect real output.
    dc = SpectralConv1d(1, 4)
    with torch.no_grad():
        dc.weights.zero_()
        dc.weights[0, 0, 0] = 3 + 4j
    torch.testing.assert_close(dc(torch.ones(1, 1, 31)), torch.full((1, 1, 31), 3.0))

    # A real-valued loss must reach both input and Fourier parameters.
    layer = SpectralConv1d(3, 7)
    x = torch.randn(2, 3, 31, requires_grad=True)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    before = layer.weights.detach().clone()
    loss = (layer(x) - torch.randn_like(x)).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weights.grad is not None
    assert torch.isfinite(layer.weights.grad).all()
    assert layer.weights.grad[..., 1:].abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, layer.weights)

    try:
        SpectralConv1d(2, 9)(torch.randn(1, 2, 8))
    except ValueError:
        pass
    else:
        raise AssertionError('Too many modes must produce a clear error')

    print('SUCCESS: lesson 3 shapes, filtering, gain, phase, gradients, and update passed.')
    print('1. What does each of b, i, o, k mean in the einsum?')
    print('2. Which axis is summed, and which axes are preserved?')
    print('3. Why must the weights be an nn.Parameter?')
    print('4. What weights reproduce the low-pass filter from lesson 2?')
    print('5. Does this Fourier multiplication mix different frequencies directly?')


if __name__ == '__main__':
    run_tests()
