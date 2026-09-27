"""Train u(x,0) -> u(x,1) for periodic viscous Burgers' equation.

u_t + u*u_x = 0.01*u_xx on x in [0,1), with periodic boundaries.
The FNO architecture is shared with the earlier heat example; targets are
generated specifically for Burgers using a checked Cole-Hopf reference solver.
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
from burgers_data import sample_burgers_pairs, check_reference


def sample_pairs(count, n, seed):
    arrays = sample_burgers_pairs(count, n, seed)
    return tuple(torch.from_numpy(array) for array in arrays)


def load_trained_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = FNO1d(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['equation']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'burgers_results')
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error('epochs and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    reference_checks = check_reference()

    # Split by whole functions, never by spatial points from the same function.
    train_x, train_y = sample_pairs(512, 128, seed=10)
    valid_x, valid_y = sample_pairs(64, 128, seed=20)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(40), num_workers=0,
    )
    architecture = {'width': 24, 'modes': 16, 'depth': 4}
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
    test_x, test_y = sample_pairs(64, 128, seed=30)
    fine_x, fine_y = sample_pairs(64, 256, seed=30)
    test = evaluate(model, test_x, test_y)
    fine = evaluate(model, fine_x, fine_y)
    identity_baseline = relative_l2(test_x, test_y).mean().item()
    mean_baseline = relative_l2(test_x.mean(dim=1, keepdim=True).expand_as(test_x), test_y).mean().item()
    equation = {
        'name': 'periodic 1D viscous Burgers equation', 'domain_length': 1.0,
        'viscosity': 0.01, 'initial_time': 0.0, 'final_time': 1.0, 'boundary': 'periodic',
    }
    metrics = {
        'torch_version': str(torch.__version__), 'device': 'cpu',
        'epochs': args.epochs, 'best_epoch': best_epoch,
        'untrained_validation': untrained_validation,
        'test_128': test, 'same_test_functions_256': fine,
        'identity_baseline_relative_l2': identity_baseline,
        'spatial_mean_baseline_relative_l2': mean_baseline,
        'equation': equation, 'architecture': architecture,
        'distribution': {'harmonics': 8, 'coefficient_raw_std': '1/k^2', 'coefficient_l1_norm': 'uniform[0.2,1.0]', 'mean': 'uniform[-0.3,0.3]'},
        'splits': {'train': 512, 'validation': 64, 'test': 64},
        'seeds': {'model': 42, 'train': 10, 'validation': 20, 'test': 30, 'shuffle': 40},
        'scope': 'Fixed viscosity=0.01, T=1, periodic unit domain; bounded synthetic smooth inputs.',
        'reference_checks': reference_checks,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / 'burgers_fno.pt'
    torch.save({'model_state': best_state, 'architecture': architecture, 'equation': equation}, checkpoint_path)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (args.output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')

    # Check saved-model inference on a fresh function that was in no split.
    reloaded, _ = load_trained_model(checkpoint_path)
    new_x, new_y = sample_pairs(1, 128, seed=1234)
    with torch.no_grad():
        predicted = reloaded(new_x)
        torch.testing.assert_close(predicted, model(new_x))
    with (args.output / 'new_prediction.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['x', 'initial_u0', 'reference_u_T1', 'predicted_uT'])
        for j in range(128):
            writer.writerow([j / 128, new_x[0, j, 0].item(), new_y[0, j, 0].item(), predicted[0, j, 0].item()])

    print(f'Best epoch: {best_epoch}')
    print(f'Test mean relative L2 at 128 points: {test["mean_relative_l2"]:.4%}')
    print(f'Same test functions at 256 points:  {fine["mean_relative_l2"]:.4%}')
    print(f'Identity baseline relative L2:      {identity_baseline:.4%}')
    print(f'Saved and reloaded model successfully: {checkpoint_path.resolve()}')


if __name__ == '__main__':
    main()
