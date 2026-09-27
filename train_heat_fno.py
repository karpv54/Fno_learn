"""A first complete FNO: u0(x) -> u(x, T) for periodic 1D heat flow.

Equation: u_t = diffusivity * u_xx, x in [0, 1), periodic boundary conditions.
The equation, diffusivity and final time are fixed for each trained model.
Training labels come from the exact decay of a finite Fourier series.
Python 3.10+ and PyTorch are required. No external dataset or GPU is needed.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from fno_model import FNO1d, evaluate, relative_l2


def sample_heat_pairs(count, n, seed, diffusivity=0.01, final_time=0.5, harmonics=8):
    """Independent random initial functions and exact final-time solutions.

    u0 = c + sum_k [a_k sin(2*pi*k*x) + b_k cos(2*pi*k*x)]
    uT = c + sum_k exp(-diffusivity*(2*pi*k)^2*T) * [same kth term]

    All harmonics lie strictly below the grid's Nyquist frequency.
    Using the same seed at a different n samples the SAME continuous functions.
    """
    if count < 1 or harmonics < 1 or n <= 2 * harmonics:
        raise ValueError('Need count >= 1, harmonics >= 1, and n > 2*harmonics')
    if diffusivity <= 0 or final_time < 0:
        raise ValueError('Need positive diffusivity and nonnegative final_time')
    generator = torch.Generator().manual_seed(seed)
    k = torch.arange(1, harmonics + 1, dtype=torch.float32)
    coefficients = torch.randn(count, 2, harmonics, generator=generator) / k
    mean = 0.2 * torch.randn(count, 1, generator=generator)
    grid = torch.arange(n, dtype=torch.float32) / n
    angles = 2 * torch.pi * k[:, None] * grid[None, :]
    sine, cosine = angles.sin(), angles.cos()
    decay = torch.exp(-diffusivity * (2 * torch.pi * k).square() * final_time)
    u0 = mean + coefficients[:, 0] @ sine + coefficients[:, 1] @ cosine
    uT = mean + (coefficients[:, 0] * decay) @ sine + (coefficients[:, 1] * decay) @ cosine
    return u0.unsqueeze(-1), uT.unsqueeze(-1)


def check_data_generator():
    """Compare the sine/cosine construction with an independent FFT solution."""
    u0, target = sample_heat_pairs(4, 63, seed=19)
    k = torch.fft.rfftfreq(63, d=1 / 63)
    decay = torch.exp(-0.01 * (2 * torch.pi * k).square() * 0.5)
    fft_solution = torch.fft.irfft(torch.fft.rfft(u0.squeeze(-1)) * decay, n=63)
    torch.testing.assert_close(fft_solution, target.squeeze(-1), atol=3e-6, rtol=3e-6)
    # Periodic heat flow preserves the mean and dissipates squared L2 energy.
    torch.testing.assert_close(u0.mean(1), target.mean(1), atol=2e-6, rtol=2e-6)
    assert (target.square().mean(1) <= u0.square().mean(1) + 1e-6).all()
    initial, at_zero = sample_heat_pairs(4, 64, seed=19, final_time=0)
    torch.testing.assert_close(initial, at_zero)
    print('Data checks passed: exact Fourier decay, initial condition, mass, and energy.')


def load_trained_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = FNO1d(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['equation']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'heat_results')
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error('epochs and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    check_data_generator()

    # Split by whole functions, never by spatial points from the same function.
    train_x, train_y = sample_heat_pairs(256, 128, seed=10)
    valid_x, valid_y = sample_heat_pairs(64, 128, seed=20)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(40), num_workers=0,
    )
    architecture = {'width': 16, 'modes': 12, 'depth': 3}
    model = FNO1d(**architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0003)
    best_error, best_state, best_epoch = float('inf'), None, None
    history = []
    untrained_validation = evaluate(model, valid_x, valid_y)
    print(f'Untrained validation relative L2: {untrained_validation["mean_relative_l2"]:.4f}', flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = F.mse_loss(prediction, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError('Training loss became non-finite')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch_x.size(0)
        scheduler.step()
        validation = evaluate(model, valid_x, valid_y)
        error = validation['mean_relative_l2']
        history.append({'epoch': epoch, 'train_mse': loss_sum / len(train_x), **validation})
        if error < best_error:
            best_error = error
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f'Epoch {epoch:3d}: train MSE={loss_sum / len(train_x):.6f}; validation relative L2={error:.4f}', flush=True)

    # Freeze model selection before generating or evaluating the test functions.
    model.load_state_dict(best_state)
    test_x, test_y = sample_heat_pairs(64, 128, seed=30)
    fine_x, fine_y = sample_heat_pairs(64, 256, seed=30)
    test = evaluate(model, test_x, test_y)
    fine = evaluate(model, fine_x, fine_y)
    identity_baseline = relative_l2(test_x, test_y).mean().item()
    mean_baseline = relative_l2(test_x.mean(dim=1, keepdim=True).expand_as(test_x), test_y).mean().item()
    equation = {
        'name': 'periodic 1D heat equation', 'domain_length': 1.0,
        'diffusivity': 0.01, 'final_time': 0.5, 'boundary': 'periodic',
    }
    metrics = {
        'torch_version': str(torch.__version__), 'device': 'cpu',
        'epochs': args.epochs, 'best_epoch': best_epoch,
        'untrained_validation': untrained_validation,
        'test_128': test, 'same_test_functions_256': fine,
        'identity_baseline_relative_l2': identity_baseline,
        'spatial_mean_baseline_relative_l2': mean_baseline,
        'equation': equation, 'architecture': architecture,
        'distribution': {'harmonics': 8, 'coefficient_std': '1/k', 'mean_std': 0.2},
        'splits': {'train': 256, 'validation': 64, 'test': 64},
        'seeds': {'model': 42, 'train': 10, 'validation': 20, 'test': 30, 'shuffle': 40},
        'scope': 'Fixed diffusivity, final time and periodic domain; synthetic smooth inputs.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / 'heat_fno.pt'
    torch.save({'model_state': best_state, 'architecture': architecture, 'equation': equation}, checkpoint_path)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (args.output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')

    # Check saved-model inference on a fresh function that was in no split.
    reloaded, _ = load_trained_model(checkpoint_path)
    new_x, new_y = sample_heat_pairs(1, 128, seed=1234)
    with torch.no_grad():
        predicted = reloaded(new_x)
        torch.testing.assert_close(predicted, model(new_x))
    with (args.output / 'new_prediction.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['x', 'initial_u0', 'exact_uT', 'predicted_uT'])
        for j in range(128):
            writer.writerow([j / 128, new_x[0, j, 0].item(), new_y[0, j, 0].item(), predicted[0, j, 0].item()])

    print(f'Best epoch: {best_epoch}')
    print(f'Test mean relative L2 at 128 points: {test["mean_relative_l2"]:.4%}')
    print(f'Same test functions at 256 points:  {fine["mean_relative_l2"]:.4%}')
    print(f'Identity baseline relative L2:      {identity_baseline:.4%}')
    print(f'Saved and reloaded model successfully: {checkpoint_path.resolve()}')


if __name__ == '__main__':
    main()
