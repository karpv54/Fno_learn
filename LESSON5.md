# Lesson 5: give your Burgers FNO the right examples

Your lesson 4 model has the complete architecture. Today you will create the
dataset it will learn from. After this, one core lesson remains: training,
evaluation, saving, and prediction in lesson 6.

Our equation and settings are fixed:

`u_t + u*u_x = 0.01*u_xx`, for `x in [0,1)`, with periodic boundaries.

Every example is a pair: **the full initial profile u(x,0)** and **its matching
final profile u(x,1)**. The two arrays describe the same problem instance and
the same spatial sample points, at different times.

Open **lesson5_exercise.py** and complete its ten TODOs. Keep it beside
`burgers_data.py`, `lesson4_exercise.py`, and `lesson3_exercise.py`. A separate
`lesson5_solution.py` is available for checking your work.

## Where the answers come from

The provided `burgers_data.py` contains a numerical reference calculation
based on the Cole-Hopf transformation. It uses the equation to produce target
solutions. You do not need to reimplement the solver in this lesson.

Three provided functions have different jobs:

| Function | Job |
| --- | --- |
| `sample_coefficients(count, seed)` | Define random smooth initial functions, independently of the grid |
| `initial_profile(coefficients, mean, n)` | Sample those initial functions at n spatial points |
| `solve_burgers(coefficients, mean, n=..., viscosity=..., final_time=...)` | Compute their solution profiles at the requested time |

Use the **same coefficients and means** to produce an input and its target.
Generating new coefficients for the target would pair unrelated problems.

The reference uses 512 grid points by default. The model uses 128. Their
grids are `j/512` and `j/128`, so selecting fine-grid indices 0,4,8,... gives
the same spatial positions as the model grid. In general:

```python
stride = solver_n // n
coarse_values = fine_values[:, ::stride]
```

The first axis is the example axis; the second is space. This is spatial
sampling, not Fourier-mode truncation. The code checks that the grids align.

The solver checks initial conditions, constant solutions, conservation of
the spatial mean, energy dissipation, and agreement under grid refinement.
An independent Fourier/RK4 solver also checks its answer at t=1. The labels
are numerically evaluated reference solutions, not assumed to be error-free.

Using a reference solver for supervised data is expected: it supplies examples
to train the network. After training, a prediction on a new initial profile
uses only the FNO. This dataset is a bounded synthetic teaching distribution,
not the published Burgers benchmark and not every possible initial condition.

## Give the data the shape your model expects

The solver returns NumPy arrays with shape `(examples, N)` and float64 values.
Your model expects PyTorch float32 tensors with a final feature axis:

```text
NumPy array               (examples, N)
torch.from_numpy(...)     (examples, N)
.float()                  (examples, N), float32
.unsqueeze(-1)            (examples, N, 1)
```

For example:

```python
values = torch.from_numpy(numpy_values).float().unsqueeze(-1)
```

`torch.from_numpy` converts the array to a tensor (initially sharing its
storage). `.float()` gives the float32 dtype used by our model. `unsqueeze(-1)`
adds the missing one-feature axis; it preserves the spatial points.
See [PyTorch from_numpy](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html).

Both `u0` and `u1` should have shape `(examples,N,1)`. Do not add the coordinate
feature here: your lesson 4 model already creates and appends it internally.

## Split entire examples, keeping the pairs together

Training, validation, and test data serve three different purposes:

| Set | Role in lesson 6 |
| --- | --- |
| Training | Calculate the losses used to update model weights |
| Validation | Choose the saved epoch and compare development choices |
| Test | Measure performance on held-out functions after choices are fixed |

We will use 640 independently sampled functions: 512 for training, 64 for
validation, and 64 for testing. The split operates on axis 0, the example axis.

Create a random ordering with `torch.randperm`. For six examples, one possible
ordering is `[4,1,5,0,3,2]`: every example ID occurs exactly once. A seeded
generator makes the ordering repeatable. See
[PyTorch randperm](https://docs.pytorch.org/docs/stable/generated/torch.randperm.html).

```python
generator = torch.Generator().manual_seed(seed)
order = torch.randperm(number_of_examples, generator=generator)
```

After selecting IDs for each partition, use them on both tensors:

```python
selected_inputs = u0[example_ids]
selected_targets = u1[example_ids]
```

Each selected row still contains all N spatial points. Splitting the points
of one initial function between training and test would not test a new initial
condition. Shuffling inputs and targets independently would break the answer
pairing even if their shapes remained correct.

In the exercise, store the three pairs in a dictionary. A dictionary associates
a name with a value; here the value is a tuple of tensors. For example:

```python
one_partition = {'train': (train_inputs, train_targets)}
pair = one_partition['train']
inputs, targets = pair
```

## TensorDataset and DataLoader

`TensorDataset` associates matching rows from your tensors:

```python
dataset = TensorDataset(inputs, targets)
one_input, one_target = dataset[0]
```

That returns the input and target for example 0, each shaped `(N,1)`.

`DataLoader` delivers groups of those paired examples:

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
batch_inputs, batch_targets = next(iter(loader))
```

Each full batch has shape `(32,N,1)` for inputs and targets. The batch dimension
is the number of functions processed together. Shuffling chooses the order
of examples; it does not reorder the spatial points within a function.
Both objects are described in the [PyTorch data guide](https://docs.pytorch.org/docs/stable/data.html).

For this lesson, shuffle training examples and keep validation/test order fixed.
Use `num_workers=0` for a straightforward Windows setup. Use `drop_last=False`
so a short final batch is retained: 70 examples with batch size 32 give batches
of sizes 32,32,6. The supplied generator makes shuffling reproducible across
fresh runs. Reusing a training loader across epochs advances its generator and
can produce a new ordering each epoch.

## Your ten TODOs

1. Sample the initial functions on the model grid.
2. Compute their final solutions on the finer reference grid at t=1.
3. Select the final values at the model's spatial points.
4. Convert inputs into float32 tensors of shape `(examples,N,1)`.
5. Convert targets to the same shape and dtype.
6. Create a shuffled ordering of whole-example IDs.
7. Build three paired partitions using the corresponding IDs.
8. Wrap an input/target pair in TensorDataset.
9. Create a DataLoader with the requested settings.
10. Create loaders for all partitions, enabling shuffle only for training.

Replace each `NotImplementedError` with one Python statement. Dictionary and
function-call statements can span several lines inside braces/parentheses.
The checks and demonstration below the marked line are provided; keep them
unchanged. The tests use artificial ID-valued arrays to check pairing; those
test fixtures are separate from the real Burgers data.

Run the exercise in your existing Python environment. In a terminal opened
in the lessons folder:

```text
python lesson5_exercise.py
```

The existing workspace launcher also supports:

```powershell
.\run_lesson.ps1 lesson5_exercise.py
```

Once all TODOs pass, the provided demonstration generates the 640-example
dataset, checks a batch against your lesson 4 model, saves it, and verifies
that loading it restores identical tensors and metadata. It writes:

- `lesson5_data/burgers_pairs.pt`: the three paired tensor partitions and metadata.
- `lesson5_data/metadata.json`: readable equation, sampling, and split settings.

The generation seed is 2026 and split seed is 81. This is a new dataset from
the same teaching distribution as the earlier full reference example; the
earlier trained model's reported accuracy is not an evaluation of this split.
No normalization is applied in this lesson; the generated inputs are bounded.
No model weights are updated today.

## Questions before lesson 6

1. What do the three axes of u0 and u1 represent?
2. Why must inputs and targets use the same example IDs when splitting?
3. Why split whole functions rather than spatial points?
4. What is the difference between TensorDataset and DataLoader?
5. Does shuffling training data change the order of spatial points?
6. How do training, validation, and test data have different roles?
7. Why are the reference solver and an optimizer still needed in supervised
   operator learning?

Lesson 6 will use these saved pairs to train your FNO, choose weights using
validation error, evaluate on the held-out test set, and predict new solutions.
