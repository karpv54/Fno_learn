"""Lesson 4: assemble an FNO for Burgers u(x,0) -> u(x,1).

Fixed problem: u_t + u*u_x = 0.01*u_xx, x in [0,1), periodic boundaries.
Today we build the architecture. Burgers labels and training come in lessons 5-6.

Input: u0 with shape (B,N,1). Output: prediction with shape (B,N,1).
The model builds its own x-coordinate feature. Keep this file beside lesson 3.

Exercise rules: replace each NotImplementedError with one Python statement;
keep the automatic tests unchanged. Do not hard-code batch size, grid size,
width, modes, depth, or projection_width. Feature counts 2 and 1 are allowed:
they mean [u0,x] and the scalar predicted field, respectively.

Tools: your SpectralConv1d, nn.Conv1d, nn.Linear, nn.ModuleList, nn.Sequential,
nn.GELU, F.gelu, torch.cat, tensor.permute, and ordinary Python loops.
The grid construction and the loop over blocks are provided.
"""

import torch
from torch import nn
from torch.nn import functional as F
from lesson3_exercise import SpectralConv1d


class FNOBlock(nn.Module):
    """Two parallel paths; both return (B,width,N)."""

    def __init__(self, width: int, modes: int):
        super().__init__()

        # TODO 1: Store the width -> width pointwise convolution in self.local.
        self.local = nn.Conv1d(width, width, kernel_size=1)

        # TODO 2: Store your learned Fourier layer in self.spectral.
        self.spectral = SpectralConv1d(width, modes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # TODO 3: Apply both paths to hidden, add their outputs, and return GELU.
        return F.gelu(self.local(hidden) + self.spectral(hidden))


class FNO1d(nn.Module):
    def __init__(self, width=16, modes=12, depth=3, projection_width=32):
        super().__init__()
        if min(width, modes, depth, projection_width) < 1:
            raise ValueError('All architecture settings must be positive')

        # TODO 4: Store a Linear layer mapping [u0,x] to width in self.lift.
        self.lift = nn.Linear(2, width)

        # TODO 5: Store depth independently created FNOBlock objects in
        # self.blocks, using nn.ModuleList and a list comprehension.
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(depth)])

        # TODO 6: Store a Sequential projection in self.project:
        # Linear(width, projection_width) -> GELU -> Linear(projection_width, 1).
        # The final layer is linear, so the prediction can have either sign.
        self.project = nn.Sequential(nn.Linear(width, projection_width), nn.GELU(), nn.Linear(projection_width, 1))

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        if u0.ndim != 3 or u0.size(-1) != 1:
            raise ValueError('Expected u0 with shape (B,N,1)')

        # Provided: uniform periodic grid, without duplicating the endpoint x=1.
        batch, n, _ = u0.shape
        grid = torch.arange(n, dtype=u0.dtype, device=u0.device) / n
        grid = grid.view(1, n, 1).expand(batch, n, 1)

        # TODO 7: Set features to [u0,grid], joined along the last axis.
        features = torch.cat((u0, grid), dim=-1)

        # TODO 8: Lift features, then swap to channels first; store in hidden.
        hidden = self.lift(features).permute(0, 2, 1)

        for block in self.blocks:
            # TODO 9: Update hidden by applying this block to it.
            hidden = block(hidden)

        # TODO 10: Swap hidden back to channels last, apply self.project, return.
        return self.project(hidden.permute(0, 2, 1))


# Automatic tests: do not modify anything below this line.
def run_tests():
    torch.manual_seed(4)
    torch.set_num_threads(2)

    # Known input: the local branch preserves both frequencies; the spectral
    # branch passes only the lower frequency. This catches sequential wiring.
    block = FNOBlock(width=2, modes=6)
    assert isinstance(block.local, nn.Conv1d), 'TODO 1: use Conv1d'
    assert block.local.weight.shape == (2, 2, 1), 'TODO 1: wrong channel/kernel sizes'
    assert isinstance(block.spectral, SpectralConv1d), 'TODO 2: reuse lesson 3'
    with torch.no_grad():
        block.local.weight.zero_()
        block.local.bias.fill_(0.1)
        block.spectral.weights.zero_()
        for c in range(2):
            block.local.weight[c, c, 0] = 2
            block.spectral.weights[c, c, :] = 1
    grid = torch.arange(64) / 64
    low = torch.sin(2 * torch.pi * 2 * grid)
    high = 0.3 * torch.cos(2 * torch.pi * 12 * grid)
    hidden = (low + high).view(1, 1, 64).expand(2, 2, 64).contiguous()
    expected = F.gelu(3 * low + 2 * high + 0.1).view(1, 1, 64).expand_as(hidden)
    torch.testing.assert_close(block(hidden), expected, atol=3e-6, rtol=3e-6)

    # Several shapes, odd lengths, and different depths catch hard-coded sizes.
    for batch, n, width, modes, depth, projection_width in (
        (4, 128, 16, 12, 3, 32),
        (2, 31, 7, 5, 2, 11),
        (1, 65, 5, 7, 1, 9),
    ):
        model = FNO1d(width, modes, depth, projection_width)
        assert isinstance(model.lift, nn.Linear), 'TODO 4: use Linear'
        assert model.lift.weight.shape == (width, 2), 'TODO 4: lift two features'
        assert isinstance(model.blocks, nn.ModuleList), 'TODO 5: use ModuleList'
        assert len(model.blocks) == depth, 'TODO 5: wrong number of blocks'
        assert len({id(b) for b in model.blocks}) == depth, 'Create a new block on each iteration'
        assert len({b.spectral.weights.data_ptr() for b in model.blocks}) == depth, 'Blocks must have independent weights'
        assert isinstance(model.project, nn.Sequential), 'TODO 6: use Sequential'
        assert len(model.project) == 3, 'TODO 6: expected Linear, GELU, Linear'
        assert isinstance(model.project[0], nn.Linear)
        assert isinstance(model.project[1], nn.GELU)
        assert isinstance(model.project[2], nn.Linear)
        assert model.project[0].weight.shape == (projection_width, width)
        assert model.project[2].weight.shape == (1, projection_width)

        # Inspect what is actually sent to the lifting layer.
        captured = []
        hook = model.lift.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        u0 = torch.randn(batch, n, 1)
        prediction = model(u0)
        hook.remove()
        assert len(captured) == 1, 'Apply the lifting layer exactly once'
        assert captured[0].shape == (batch, n, 2), 'TODO 7: wrong feature layout'
        torch.testing.assert_close(captured[0][..., :1], u0)
        expected_grid = (torch.arange(n) / n).view(1, n).expand(batch, n)
        torch.testing.assert_close(captured[0][..., 1], expected_grid)
        assert prediction.shape == (batch, n, 1), 'TODOs 8-10: wrong output layout'
        assert not prediction.is_complex() and torch.isfinite(prediction).all()

        registered = dict(model.named_parameters())
        for index, current_block in enumerate(model.blocks):
            assert registered[f'blocks.{index}.spectral.weights'] is current_block.spectral.weights

    # One model can accept a new uniform spatial resolution on the same domain.
    assert model(torch.randn(3, 127, 1)).shape == (3, 127, 1)

    # Dummy targets check gradient connections only; these are NOT Burgers labels.
    model = FNO1d(width=7, modes=5, depth=3, projection_width=11)
    u0 = torch.randn(2, 31, 1, requires_grad=True)
    dummy_target = torch.randn_like(u0)
    loss = F.mse_loss(model(u0), dummy_target)
    loss.backward()
    assert u0.grad is not None and torch.isfinite(u0.grad).all()
    assert u0.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f'{name} did not participate in the prediction'
        assert torch.isfinite(parameter.grad).all(), f'{name} has non-finite gradients'
        assert parameter.grad.abs().sum() > 0, f'{name} has only zero gradients'

    # A signed output must survive the projection without a final activation.
    with torch.no_grad():
        model.project[-1].weight.zero_()
        model.project[-1].bias.fill_(-2)
    torch.testing.assert_close(model(u0.detach()), torch.full_like(u0, -2))

    print('SUCCESS: lesson 4 branch wiring, shapes, registration, gradients, and projection passed.')
    print('This network is untrained; these checks do not establish Burgers prediction accuracy.')
    print('1. Why does the lifting layer take two features and the projection return one?')
    print('2. Why use ModuleList for the FNO blocks? Does ModuleList apply them automatically?')
    print('3. Where do we swap axes, and why?')
    print('4. Why is there no activation after the final Linear layer?')
    print('5. Does depth=3 mean three physical time steps?')
    print('6. What information is missing before this model can learn u(x,0) -> u(x,1)?')


if __name__ == '__main__':
    run_tests()
