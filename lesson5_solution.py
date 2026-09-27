"""Lesson 5: prepare paired Burgers data for u(x,0) -> u(x,1).

Equation: u_t + u*u_x = 0.01*u_xx, x in [0,1), periodic boundaries.
The reference solver is provided in burgers_data.py; it generates labels.
Today we prepare data. We do not train the network yet.

Exercise rules:
1. Replace each NotImplementedError with one Python statement (possibly multiline).
2. Keep the automatic tests unchanged.
3. Use function arguments, not hard-coded example sizes or seeds.
4. Keep every initial profile and its final profile paired.

New tools: torch.from_numpy, Tensor.float, Tensor.unsqueeze, torch.randperm,
TensorDataset, DataLoader, tensor indexing, and Python dictionaries.
Keep this file beside burgers_data.py and your completed lesson4_exercise.py.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from burgers_data import sample_coefficients, initial_profile, solve_burgers, check_reference
from lesson4_exercise import FNO1d


VISCOSITY = 0.01
FINAL_TIME = 1.0


def make_pairs(count, n, seed, solver_n=512):
    """Return u0 and u1, both real float32 tensors of shape (count,n,1)."""
    if count < 1 or n <= 16 or solver_n < n or solver_n % n != 0:
        raise ValueError('Require count > 0, n > 16, and solver_n a multiple of n')

    # Provided: coefficients and means specify the same continuous input functions
    # regardless of which uniform grid we sample them on.
    coefficients, mean = sample_coefficients(count, seed)
    stride = solver_n // n

    # TODO 1: Set u0_numpy to initial_profile(coefficients, mean, n).
    # It contains the initial values on the model grid, shape (count,n).
    u0_numpy = initial_profile(coefficients, mean, n)

    # TODO 2: Set u1_fine using solve_burgers with coefficients and mean,
    # n=solver_n, viscosity=VISCOSITY, and final_time=FINAL_TIME.
    u1_fine = solve_burgers(coefficients, mean, n=solver_n, viscosity=VISCOSITY, final_time=FINAL_TIME)

    # TODO 3: Set u1_numpy to every stride-th spatial value of u1_fine.
    # Keep ALL examples. The result must have shape (count,n).
    u1_numpy = u1_fine[:, ::stride]

    # TODO 4: Set u0 by converting u0_numpy with torch.from_numpy, converting
    # to float32 using .float(), and adding a final size-one axis with .unsqueeze(-1).
    u0 = torch.from_numpy(u0_numpy).float().unsqueeze(-1)

    # TODO 5: Perform the same conversion on u1_numpy, storing the result in u1.
    u1 = torch.from_numpy(u1_numpy).float().unsqueeze(-1)

    return u0, u1


def split_pairs(u0, u1, n_train, n_validation, seed):
    """Shuffle whole-example IDs once, then form three non-overlapping splits."""
    if u0.shape != u1.shape or u0.ndim != 3 or u0.size(-1) != 1:
        raise ValueError('Expected matching (examples,N,1) input and target tensors')
    total = u0.size(0)
    if n_train < 1 or n_validation < 1 or n_train + n_validation >= total:
        raise ValueError('Training, validation, and test sets must all be nonempty')
    generator = torch.Generator().manual_seed(seed)

    # TODO 6: Set order to a random permutation of integers 0 through total-1,
    # using torch.randperm and the supplied generator.
    order = torch.randperm(total, generator=generator)

    # Provided: reserve different whole examples for the three purposes.
    train_ids = order[:n_train]
    validation_ids = order[n_train:n_train + n_validation]
    test_ids = order[n_train + n_validation:]

    # TODO 7: Set partitions to a dictionary with keys 'train', 'validation', 'test'.
    # Each value is a tuple (selected_u0, selected_u1), using the SAME corresponding
    # ID tensor on both u0 and u1. Select examples along axis 0.
    partitions = {'train': (u0[train_ids], u1[train_ids]), 'validation': (u0[validation_ids], u1[validation_ids]), 'test': (u0[test_ids], u1[test_ids])}

    return partitions


def make_loader(u0, u1, batch_size, shuffle, seed):
    """Package paired examples and deliver them in mini-batches."""
    generator = torch.Generator().manual_seed(seed)

    # TODO 8: Store a TensorDataset containing u0 and u1 in dataset.
    dataset = TensorDataset(u0, u1)

    # TODO 9: Store a DataLoader in loader, using dataset, batch_size, shuffle,
    # generator=generator, num_workers=0, and drop_last=False.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)

    return loader


def make_loaders(partitions, batch_size, seed):
    """Training shuffles paired examples; validation/test preserve their order."""
    loaders = {}
    for name, pair in partitions.items():
        # TODO 10: Set loaders[name] by calling make_loader on pair[0], pair[1],
        # batch_size, shuffle=(name == 'train'), and seed.
        loaders[name] = make_loader(pair[0], pair[1], batch_size, shuffle=(name == 'train'), seed=seed)
    return loaders


# Automatic tests and demonstration: do not modify anything below this line.
def run_tests():
    torch.set_num_threads(2)
    reference_checks = check_reference()

    # Check actual t=0/t=1 pairing and spatial coordinates at multiple resolutions.
    for count, n, solver_n, seed in ((5, 128, 512, 51), (3, 64, 256, 52), (2, 31, 248, 53)):
        u0, u1 = make_pairs(count, n, seed, solver_n)
        for tensor in (u0, u1):
            assert tensor.shape == (count, n, 1), 'TODOs 1-5: expected (examples,N,1)'
            assert tensor.dtype == torch.float32, 'TODOs 4-5: match model float32 weights'
            assert not tensor.is_complex() and torch.isfinite(tensor).all()
        # Evaluate the reference at a DIFFERENT finer resolution and compare values
        # at the same x positions. This also catches wrong times and wrong slicing.
        coefficients, mean = sample_coefficients(count, seed)
        expected_u0 = torch.from_numpy(initial_profile(coefficients, mean, n)).float().unsqueeze(-1)
        finer = solve_burgers(coefficients, mean, n=2 * solver_n, viscosity=0.01, final_time=1.0)
        expected_u1 = torch.from_numpy(finer[:, ::2 * solver_n // n].copy()).float().unsqueeze(-1)
        torch.testing.assert_close(u0, expected_u0)
        torch.testing.assert_close(u1, expected_u1, atol=2e-6, rtol=2e-5)
        assert not torch.allclose(u0, u1), 'These nonconstant inputs should evolve by t=1'
        repeat_u0, repeat_u1 = make_pairs(count, n, seed, solver_n)
        torch.testing.assert_close(u0, repeat_u0, atol=0, rtol=0)
        torch.testing.assert_close(u1, repeat_u1, atol=0, rtol=0)

    # Synthetic IDs check bookkeeping only; these are not physical PDE labels.
    total, n = 17, 9
    ids = torch.arange(total, dtype=torch.float32).view(total, 1, 1).expand(total, n, 1).clone()
    answers = 7 * ids + 3
    splits = split_pairs(ids, answers, n_train=9, n_validation=4, seed=61)
    assert set(splits) == {'train', 'validation', 'test'}
    expected_order = torch.randperm(total, generator=torch.Generator().manual_seed(61))
    expected_ids = {'train': expected_order[:9], 'validation': expected_order[9:13], 'test': expected_order[13:]}
    recovered_ids = []
    for name, (inputs, targets) in splits.items():
        assert inputs.shape == targets.shape == (len(expected_ids[name]), n, 1)
        torch.testing.assert_close(targets, 7 * inputs + 3, msg='Input/target pairing was broken')
        torch.testing.assert_close(inputs[:, 0, 0], expected_ids[name].float())
        recovered_ids.append(inputs[:, 0, 0])
    all_ids = torch.cat(recovered_ids)
    torch.testing.assert_close(all_ids.sort().values, torch.arange(total).float())
    assert all_ids.unique().numel() == total, 'An example appears in more than one split'
    repeated = split_pairs(ids, answers, 9, 4, seed=61)
    changed = split_pairs(ids, answers, 9, 4, seed=62)
    for name in splits:
        torch.testing.assert_close(repeated[name][0], splits[name][0], atol=0, rtol=0)
    assert not torch.equal(changed['train'][0], splits['train'][0]), 'TODO 6: use the seed'
    assert torch.equal(answers, 7 * ids + 3), 'Do not mutate the supplied examples'

    # Shuffle pairs together, retain every example, and keep short final batches.
    loaders = make_loaders(splits, batch_size=4, seed=71)
    from torch.utils.data import RandomSampler, SequentialSampler
    assert isinstance(loaders['train'].sampler, RandomSampler), 'Shuffle training examples'
    assert isinstance(loaders['validation'].sampler, SequentialSampler), 'Keep validation order'
    assert isinstance(loaders['test'].sampler, SequentialSampler), 'Keep test order'
    for name, loader in loaders.items():
        assert isinstance(loader.dataset, TensorDataset), 'TODO 8: use TensorDataset'
        assert loader.num_workers == 0 and not loader.drop_last
        batches = list(loader)
        inputs = torch.cat([batch[0] for batch in batches])
        targets = torch.cat([batch[1] for batch in batches])
        torch.testing.assert_close(targets, 7 * inputs + 3)
        torch.testing.assert_close(inputs[:, 0, 0].sort().values, splits[name][0][:, 0, 0].sort().values)
        if name != 'train':
            torch.testing.assert_close(inputs, splits[name][0])
    assert [len(x) for x, _ in make_loader(*splits['train'], 4, False, 72)] == [4, 4, 1]
    first = list(make_loader(*splits['train'], 4, True, 73))
    second = list(make_loader(*splits['train'], 4, True, 73))
    torch.testing.assert_close(torch.cat([x for x, _ in first]), torch.cat([x for x, _ in second]))

    print('SUCCESS: lesson 5 targets, shapes, splits, reproducibility, and paired batches passed.')
    return reference_checks


def prepare_lesson6_data(reference_checks):
    """Provided: save the completed dataset, with the equation and sampling metadata."""
    settings = {'examples': 640, 'grid_points': 128, 'solver_grid_points': 512,
                'data_seed': 2026, 'split_seed': 81, 'loader_seed': 82}
    u0, u1 = make_pairs(settings['examples'], settings['grid_points'], settings['data_seed'], settings['solver_grid_points'])
    partitions = split_pairs(u0, u1, n_train=512, n_validation=64, seed=settings['split_seed'])
    loaders = make_loaders(partitions, batch_size=32, seed=settings['loader_seed'])
    batch_u0, batch_u1 = next(iter(loaders['train']))
    torch.manual_seed(5)
    model = FNO1d(width=16, modes=12, depth=3, projection_width=32)
    model.eval()
    with torch.no_grad():
        prediction = model(batch_u0)
    assert prediction.shape == batch_u1.shape == (32, 128, 1)
    assert torch.isfinite(prediction).all()

    equation = {'name': 'viscous Burgers', 'viscosity': 0.01, 'initial_time': 0.0,
                'final_time': 1.0, 'domain_length': 1.0, 'boundary': 'periodic'}
    metadata = {'equation': equation, 'sampling': settings, 'reference_checks': reference_checks,
                'split_sizes': {name: len(pair[0]) for name, pair in partitions.items()},
                'normalization': 'none', 'dtype': 'float32',
                'input_distribution': {'harmonics': 8, 'raw_coefficient_std': '1/k^2',
                    'coefficient_l1_norm': 'uniform[0.2,1.0]', 'mean': 'uniform[-0.3,0.3]'}}
    output = Path(__file__).parent / 'lesson5_data'
    output.mkdir(exist_ok=True)
    dataset_path = output / 'burgers_pairs.pt'
    torch.save({'metadata': metadata, 'splits': partitions}, dataset_path)
    restored = torch.load(dataset_path, map_location='cpu', weights_only=True)
    assert restored['metadata'] == metadata
    for name, pair in partitions.items():
        for index in (0, 1):
            torch.testing.assert_close(restored['splits'][name][index], pair[index], atol=0, rtol=0)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Dataset splits:', metadata['split_sizes'])
    print('Training input batch:', tuple(batch_u0.shape), 'target batch:', tuple(batch_u1.shape))
    print('Your lesson 4 model accepts these batches. It is still untrained.')
    print('Saved and reloaded dataset:', dataset_path.resolve())


if __name__ == '__main__':
    checks = run_tests()
    prepare_lesson6_data(checks)
    print('\nQuestions before lesson 6:')
    print('1. What do the three axes of u0 and u1 represent?')
    print('2. Why must inputs and targets use the same example IDs when splitting?')
    print('3. Why split whole functions rather than spatial points?')
    print('4. What is the difference between TensorDataset and DataLoader?')
    print('5. Does shuffling training data change the order of spatial points?')
    print('6. How do training, validation, and test data have different roles?')
    print('7. Why are the reference solver and an optimizer still needed in supervised operator learning?')
