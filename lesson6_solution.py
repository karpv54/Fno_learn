"""Lesson 6: train, evaluate, save, and reload your Burgers FNO.

Problem: u_t + u*u_x = 0.01*u_xx, periodic x in [0,1), u(x,0) -> u(x,1).
Uses YOUR lesson4_exercise.FNO1d and lesson5_exercise data loaders.

Complete the 12 TODOs, one statement per TODO. Use the supplied arguments.
Read TRAINING_CHEAT_SHEET.md and LESSON6.md first.
Run: python lesson6_exercise.py --test-only
Then: python lesson6_exercise.py --epochs 128
The tests check your training code; the real run measures prediction quality.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from lesson4_exercise import FNO1d
from lesson5_exercise import make_loaders, make_pairs


def make_optimizer(model, learning_rate):
    """Adam must receive the parameters belonging to this model."""
    # TODO 1: Return torch.optim.Adam for model.parameters(), lr=learning_rate.
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch(model, loader, optimizer):
    """Visit the training examples once; return their weighted mean batch MSE."""
    # TODO 2: Switch the model to training mode (this alone does not train it).
    model.train()
    loss_sum, count = 0.0, 0
    for batch_u0, batch_u1 in loader:
        # TODO 3: Clear previous gradients with optimizer.zero_grad().
        optimizer.zero_grad()
        # TODO 4: Set prediction by applying model to batch_u0.
        prediction = model(batch_u0)
        # TODO 5: Set loss to F.mse_loss(prediction, batch_u1).
        loss = F.mse_loss(prediction, batch_u1)
        if not torch.isfinite(loss):
            raise RuntimeError('Training loss is not finite; check your learning rate and data')
        # TODO 6: Compute gradients by calling loss.backward().
        loss.backward()
        # TODO 7: Update the weights by calling optimizer.step().
        optimizer.step()
        # Provided: account for a smaller final batch. item() is for logging only.
        loss_sum += loss.item() * batch_u0.size(0)
        count += batch_u0.size(0)
    if count == 0:
        raise ValueError('Training loader must contain examples')
    return loss_sum / count


def evaluate(model, loader):
    """Measure error without changing weights or recording gradients."""
    # TODO 8: Switch the model to evaluation mode.
    model.eval()
    mse_sum, relative_sum, worst, count = 0.0, 0.0, 0.0, 0
    # Provided: eval() and no_grad() do different jobs; see the cheat sheet.
    with torch.no_grad():
        for batch_u0, batch_u1 in loader:
            prediction = model(batch_u0)
            if not torch.isfinite(prediction).all():
                raise RuntimeError('Evaluation produced non-finite predictions')
            relative = relative_l2(prediction, batch_u1)
            mse_sum += F.mse_loss(prediction, batch_u1).item() * batch_u0.size(0)
            relative_sum += relative.sum().item()
            worst = max(worst, relative.max().item())
            count += batch_u0.size(0)
    if count == 0:
        raise ValueError('Evaluation loader must contain examples')
    return {'mse': mse_sum / count, 'mean_relative_l2': relative_sum / count,
            'max_relative_l2': worst}


def snapshot_weights(model):
    """Freeze a copy; later optimizer steps must not change this snapshot."""
    # TODO 9: Return copy.deepcopy(model.state_dict()).
    return copy.deepcopy(model.state_dict())


def save_checkpoint(path, model, architecture, metadata):
    """Save an inference checkpoint containing weights and their settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {'model_state': model.state_dict(),
                  'architecture': architecture, 'metadata': metadata}
    # TODO 10: Save checkpoint to path with torch.save.
    torch.save(checkpoint, path)


def load_checkpoint(path):
    """Rebuild the same architecture and insert the saved trained weights."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = FNO1d(**checkpoint['architecture'])
    # TODO 11: Load checkpoint['model_state'] into model with load_state_dict.
    model.load_state_dict(checkpoint['model_state'])
    # TODO 12: Switch the reloaded model to evaluation mode.
    model.eval()
    return model, checkpoint['metadata']


# Provided metrics, tests, and training driver: do not modify below this line.
ARCHITECTURE = {'width': 16, 'modes': 12, 'depth': 3, 'projection_width': 32}
EXPECTED_EQUATION = {'name': 'viscous Burgers', 'viscosity': 0.01,
                     'initial_time': 0.0, 'final_time': 1.0,
                     'domain_length': 1.0, 'boundary': 'periodic'}


def relative_l2(prediction, target):
    """One error per curve; 0.02 means 2% relative L2 error, not 98% accuracy."""
    if prediction.shape != target.shape or target.ndim != 3:
        raise ValueError('Expected matching (examples,N,1) tensors')
    error_norm = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    target_norm = torch.linalg.vector_norm(target.flatten(1), dim=1)
    return error_norm / target_norm.clamp_min(1e-8)


def run_tests():
    torch.set_num_threads(2)
    torch.manual_seed(6)
    small = nn.Linear(1, 1, bias=False)
    optimizer = make_optimizer(small, learning_rate=0.0123)
    assert isinstance(optimizer, torch.optim.Adam), 'TODO 1: use Adam'
    assert all(group['lr'] == 0.0123 for group in optimizer.param_groups), 'Use learning_rate'
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {
        id(p) for p in small.parameters()}, 'Optimize the supplied model'

    # A known scalar learning problem catches missing steps and accumulated grads.
    small.eval()
    with torch.no_grad():
        small.weight.zero_()
    small.weight.grad = torch.full_like(small.weight, 11.0)
    x = torch.ones(5, 3, 1)
    y = 2 * x
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    measured_loss = train_one_epoch(small, loader, torch.optim.SGD(small.parameters(), lr=0.1))
    # Three updates: w = 0 -> .4 -> .72 -> .976. Last batch has one example.
    torch.testing.assert_close(small.weight, torch.tensor([[0.976]]))
    assert math.isclose(measured_loss, (2*4 + 2*2.56 + 1.6384)/5, rel_tol=1e-6)
    assert small.training, 'TODO 2: call model.train()'

    class EvalProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, values):
            assert not self.training, 'TODO 8: call model.eval()'
            assert not torch.is_grad_enabled(), 'Keep the provided no_grad context'
            return values * self.scale

    probe = EvalProbe()
    probe.scale.grad = torch.tensor(7.0)
    probe_x = torch.tensor([2.0, 2.0, 1.25]).view(3, 1, 1).expand(3, 4, 1)
    probe_y = torch.ones_like(probe_x)
    metrics = evaluate(probe, DataLoader(TensorDataset(probe_x, probe_y), batch_size=2))
    assert math.isclose(metrics['mse'], 0.6875, rel_tol=1e-6)
    assert math.isclose(metrics['mean_relative_l2'], 0.75, rel_tol=1e-6)
    assert metrics['max_relative_l2'] == 1.0
    assert probe.scale.item() == 1.0 and probe.scale.grad.item() == 7.0
    torch.testing.assert_close(relative_l2(probe_y, probe_y), torch.zeros(3))

    frozen = snapshot_weights(small)
    before = small.weight.detach().clone()
    with torch.no_grad():
        small.weight.add_(10)
    torch.testing.assert_close(frozen['weight'], before, msg='TODO 9: use deepcopy')

    # The real network must propagate gradients through all of its main branches.
    architecture = {'width': 4, 'modes': 5, 'depth': 2, 'projection_width': 7}
    model = FNO1d(**architecture)
    inputs, targets = make_pairs(count=5, n=32, seed=6006, solver_n=256)
    original = copy.deepcopy(model.state_dict())
    train_one_epoch(model, DataLoader(TensorDataset(inputs, targets), batch_size=3),
                    make_optimizer(model, 0.001))
    for key in ('lift.weight', 'blocks.0.local.weight', 'blocks.0.spectral.weights',
                'project.2.weight'):
        assert not torch.equal(original[key], model.state_dict()[key]), f'No update: {key}'
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        expected = model(inputs)
    with tempfile.TemporaryDirectory(prefix='fno_lesson6_') as temporary:
        path = Path(temporary) / 'test_checkpoint.pt'
        metadata = {'equation': EXPECTED_EQUATION, 'purpose': 'test'}
        save_checkpoint(path, model, architecture, metadata)
        restored, restored_metadata = load_checkpoint(path)
        assert restored_metadata == metadata
        assert not restored.training, 'TODO 12: call model.eval()'
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs), expected, atol=0, rtol=0)
    print('SUCCESS: lesson 6 updates, gradients, evaluation, snapshots, and save/load passed.', flush=True)


def read_dataset(path):
    if not path.is_file():
        raise FileNotFoundError(f'Run your completed lesson5_exercise.py first: missing {path}')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = saved['metadata']
    if metadata['equation'] != EXPECTED_EQUATION or metadata['normalization'] != 'none':
        raise ValueError('Use the unnormalized periodic Burgers dataset from lesson 5')
    partitions = saved['splits']
    if set(partitions) != {'train', 'validation', 'test'}:
        raise ValueError('Expected three lesson 5 partitions')
    for name, (u0, u1) in partitions.items():
        if (u0.shape != u1.shape or u0.ndim != 3 or u0.size(-1) != 1
                or len(u0) != metadata['split_sizes'][name] or len(u0) == 0
                or u0.size(1) != metadata['sampling']['grid_points']):
            raise ValueError(f'Invalid {name} tensor shapes')
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (u0, u1)):
            raise ValueError(f'Expected finite float32 {name} tensors')
    return partitions, metadata


def save_plots(output, history, u0, target, prediction):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib is unavailable; numerical results are still saved.', flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].semilogy([r['epoch'] for r in history], [r['train_mse'] for r in history], label='Training')
    axes[0].semilogy([r['epoch'] for r in history], [r['validation_mse'] for r in history], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Mean squared error', title='Learning from the lesson 5 data')
    axes[0].legend()
    n = u0.size(1)
    grid = torch.arange(n) / n
    axes[1].plot(grid, u0[0, :, 0], color='0.65', label='Initial curve, t=0')
    axes[1].plot(grid, target[0, :, 0], color='#147d73', linewidth=2.5, label='Reference, t=1')
    axes[1].plot(grid, prediction[0, :, 0], '--', color='#bd532f', linewidth=1.8, label='FNO, t=1')
    axes[1].set(xlabel='x', ylabel='u', title='A new initial condition')
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output / 'training_and_prediction.png', dpi=160)
    plt.close(fig)


def run_training(dataset_path, output, epochs, learning_rate):
    if (output / 'burgers_fno.pt').exists():
        raise FileExistsError('A model already exists here. Use --output lesson6_try2 for another run.')
    partitions, data_metadata = read_dataset(dataset_path)
    # The real run resets seeds after the automatic tests.
    torch.manual_seed(42)
    loaders = make_loaders(partitions, batch_size=32, seed=82)
    model = FNO1d(**ARCHITECTURE)
    optimizer = make_optimizer(model, learning_rate)
    # Provided: gradually reduce the learning rate, after each full epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate/10)
    initial_validation = evaluate(model, loaders['validation'])
    print(f'Untrained validation relative L2: {initial_validation["mean_relative_l2"]:.2%}', flush=True)
    best_error, best_epoch, best_state = float('inf'), None, None
    history = []
    for epoch in range(1, epochs + 1):
        train_mse = train_one_epoch(model, loaders['train'], optimizer)
        validation = evaluate(model, loaders['validation'])
        history.append({'epoch': epoch, 'train_mse': train_mse,
                        'validation_mse': validation['mse'],
                        'validation_relative_l2': validation['mean_relative_l2']})
        if validation['mean_relative_l2'] < best_error:
            best_error = validation['mean_relative_l2']
            best_state = snapshot_weights(model)
            best_epoch = epoch
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'Epoch {epoch:3d}/{epochs}: train MSE={train_mse:.6f}; '
                  f'validation relative L2={validation["mean_relative_l2"]:.2%}', flush=True)

    # Select only with validation. The test set is first evaluated after this choice.
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders['test'])
    test_u0, test_u1 = partitions['test']
    identity_error = relative_l2(test_u0, test_u1).mean().item()
    mean_prediction = test_u0.mean(dim=1, keepdim=True).expand_as(test_u1)
    mean_error = relative_l2(mean_prediction, test_u1).mean().item()
    metadata = {'equation': EXPECTED_EQUATION, 'data_metadata': data_metadata,
                'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                'architecture': ARCHITECTURE, 'epochs': epochs, 'best_epoch': best_epoch,
                'initial_learning_rate': learning_rate, 'batch_size': 32,
                'model_seed': 42, 'loader_seed': 82, 'device': 'cpu',
                'torch_version': str(torch.__version__), 'checkpoint_purpose': 'inference'}
    save_checkpoint(output / 'burgers_fno.pt', model, ARCHITECTURE, metadata)
    restored, restored_metadata = load_checkpoint(output / 'burgers_fno.pt')
    assert restored_metadata == metadata
    # A fresh curve for demonstration; it does not participate in model selection.
    new_u0, new_u1 = make_pairs(count=1, n=test_u0.size(1), seed=9001,
                              solver_n=data_metadata['sampling']['solver_grid_points'])
    with torch.no_grad():
        prediction = restored(new_u0)
        torch.testing.assert_close(prediction, model(new_u0), atol=0, rtol=0)
    new_error = relative_l2(prediction, new_u1).item()
    report = {'initial_validation': initial_validation, 'best_validation_relative_l2': best_error,
              'best_epoch': best_epoch, 'test': test_metrics,
              'copy_input_baseline_relative_l2': identity_error,
              'spatial_mean_baseline_relative_l2': mean_error,
              'new_example_seed': 9001, 'new_example_relative_l2': new_error,
              'metadata': metadata}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    torch.save({'u0': new_u0, 'reference_u1': new_u1, 'predicted_u1': prediction},
               output / 'new_example.pt')
    with (output / 'new_prediction.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['x', 'initial_u0', 'reference_u1', 'predicted_u1'])
        n = new_u0.size(1)
        for j in range(n):
            writer.writerow([j/n, new_u0[0,j,0].item(), new_u1[0,j,0].item(), prediction[0,j,0].item()])
    save_plots(output, history, new_u0, new_u1, prediction)
    print(f'Best validation epoch: {best_epoch}', flush=True)
    print(f'Test mean relative L2: {test_metrics["mean_relative_l2"]:.2%}', flush=True)
    print(f'Test worst relative L2: {test_metrics["max_relative_l2"]:.2%}', flush=True)
    print(f'Copy-input baseline: {identity_error:.2%}; mean baseline: {mean_error:.2%}', flush=True)
    print(f'New example relative L2: {new_error:.2%}', flush=True)
    print(f'Model saved and reloaded: {(output / "burgers_fno.pt").resolve()}', flush=True)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--dataset', type=Path, default=directory / 'lesson5_data' / 'burgers_pairs.pt')
    default_output = 'lesson6_reference_results' if 'solution' in Path(__file__).stem else 'lesson6_results'
    parser.add_argument('--output', type=Path, default=directory / default_output)
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('epochs and learning-rate must be positive and finite')
    run_tests()
    if not args.test_only:
        run_training(args.dataset, args.output, args.epochs, args.learning_rate)


if __name__ == '__main__':
    main()
