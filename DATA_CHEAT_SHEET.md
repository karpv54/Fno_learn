# Lesson 5 — data cheat sheet

**One example = one complete initial curve and its matching final curve.**

Goal: **u(x,0) → u(x,1)**. Viscosity **0.01**; periodic domain **[0,1)**.

```text
Choose initial curves → solve Burgers → convert to tensors → split → batch
```

Run the Python blocks in order, beside `burgers_data.py`.

## 1. What are we importing?

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from burgers_data import (
    sample_coefficients, initial_profile, solve_burgers, check_reference
)
```

**`burgers_data`** is our local Python file. It uses **NumPy** (`np`) for numerical arrays. **PyTorch** (`torch`) uses tensors: arrays suited to neural networks.

`import` makes tools available; `solve_burgers(...)` calls the function.

## 2. The four Burgers functions

| Call | Meaning | Returns |
|---|---|---|
| `sample_coefficients(count=10, seed=42)` | Choose ingredients for 10 initial curves. | `coefficients, mean` |
| `initial_profile(coefficients, mean, n=128)` | Sample initial curves at 128 points. | NumPy array `(10,128)` at **t=0** |
| `solve_burgers(...)` | Compute their final curves; example below. | NumPy array `(10,512)` at **t=1** |
| `check_reference()` | Test the provided solver. | Results dictionary; an error if checks fail. |

**`coefficients`** are sine/cosine strengths defining the initial curves, not FNO weights. Shape `(10,2,8)`: 10 curves, sine/cosine, 8 frequencies.

**`mean`** is one average value per curve, shape `(10, 1)`.

**`count`** = curves; **`n`** = spatial points. Same sampler arguments, including **`seed`**, reproduce the curves.

```python
coefficients, mean = sample_coefficients(count=10, seed=42)
u0_np = initial_profile(coefficients, mean, n=128)
u1_fine = solve_burgers(
    coefficients, mean, n=512, viscosity=0.01, final_time=1.0
)
stride = 512 // 128                # 4
u1_np = u1_fine[:, ::stride]       # (10, 128)
```

Both times use the **same coefficients and mean**. This solver takes those ingredients, not `u0_np`.

`[:, ::4]` = **all curves, every fourth point**. It matches the initial grid: 0, 1/128, …, 127/128.

## 3. Convert arrays into model inputs

```python
u0 = torch.from_numpy(u0_np).float().unsqueeze(-1)
u1 = torch.from_numpy(u1_np).float().unsqueeze(-1)
```

| Operation | Effect |
|---|---|
| `torch.from_numpy(...)` | NumPy array → PyTorch tensor; keeps the shape. |
| `.float()` | Uses 32-bit decimal numbers, matching our model. |
| `.unsqueeze(-1)` | Adds a last axis: `(10,128)` → `(10,128,1)`. |

Shape = **(examples, spatial points, channels)**. One channel means one value of **u** per point. [Conversion](https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html) and [unsqueeze reference](https://docs.pytorch.org/docs/2.14/generated/torch.unsqueeze.html).

`u0.shape` gives all sizes; `u0.size(0)` is 10; `u0.ndim` is 3. `u0[0, :, 0]` selects the first complete curve.

## 4. Split whole examples

```python
generator = torch.Generator().manual_seed(81)
order = torch.randperm(10, generator=generator)
train_ids, val_ids, test_ids = order[:6], order[6:8], order[8:]
partitions = {
    "train": (u0[train_ids], u1[train_ids]),
    "validation": (u0[val_ids], u1[val_ids]),
    "test": (u0[test_ids], u1[test_ids]),
}
```

[`randperm(10)`](https://docs.pytorch.org/docs/2.14/generated/torch.randperm.html) shuffles IDs 0–9 without repeats. `manual_seed` seeds its generator. `[:6]` takes six IDs; `[6:8]` takes the next two.

**Use the same IDs for inputs and answers. Keep each curve's points in order.**

| Split | Purpose |
|---|---|
| Training: 6 examples | Update the model's weights. |
| Validation: 2 examples | Choose a model during training. |
| Test: 2 examples | Evaluate on unseen examples. |

The **dictionary** stores named **tuples** `(inputs, answers)`: `pair[0]` = inputs; `pair[1]` = answers.

## 5. Make batches

```python
loaders = {}
for name, pair in partitions.items():
    loaders[name] = DataLoader(
        TensorDataset(pair[0], pair[1]),
        batch_size=4, shuffle=(name == "train"),
        generator=torch.Generator().manual_seed(82),
        num_workers=0, drop_last=False,
    )

for batch_u0, batch_u1 in loaders["train"]:
    print(batch_u0.shape, batch_u1.shape)
```

`.items()` visits each name and pair. Only training shuffles.

**`TensorDataset`** pairs matching rows. **`DataLoader`** groups pairs into batches. `num_workers=0` uses the current process; `drop_last=False` keeps incomplete batches. [PyTorch reference](https://docs.pytorch.org/docs/2.14/data.html).

Six training examples give batch shapes **`(4,128,1)` then `(2,128,1)`**, for both inputs and answers.

## 6. The remaining names in your lesson

| Name | Meaning |
|---|---|
| `FNO1d` | Your lesson 4 network class. |
| `Path` | Build file/folder paths. |
| `json` | Read/write settings such as viscosity and seed. |
| `torch.save` / `torch.load` | Save/reload tensors and metadata. |

**Lesson 5 prepares the data. Lesson 6 trains the FNO using these answers.**
