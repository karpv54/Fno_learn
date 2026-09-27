# Lesson 6 — train and use your Burgers FNO

**This is the final core lesson. You now teach your network to predict the solution.**

We keep the same problem: **u(x,0) → u(x,1)** for viscosity **0.01**, on **[0,1)** with periodic boundaries.

Start with [TRAINING_CHEAT_SHEET.md](TRAINING_CHEAT_SHEET.md), then fill the 12 TODOs in [lesson6_exercise.py](lesson6_exercise.py). The driver and tests below the marked line are provided. The completed [solution](lesson6_solution.py) is available for comparison.

## What you already built

| Earlier lesson | What lesson 6 uses |
|---|---|
| Lesson 3 | Your learned Fourier channel mixing. |
| Lesson 4 | Your `FNO1d`: width 16, 12 modes, 3 blocks, projection 16 → 32 → 1. |
| Lesson 5 | Your saved input/answer pairs and `make_loaders`. |

The data contains **512 training**, **64 validation**, and **64 test** examples, each shaped `(128,1)`. A usual training batch is `(32,128,1)`.

The initial curve is the input. Its matching curve at t=1 is the target, also called the label. Training changes the FNO's weights; the reference targets stay fixed.

## 1. What training actually does

For each batch:

```text
Clear old gradients → predict → measure loss → calculate gradients → update weights
```

The **loss** is one number measuring prediction error:

```python
loss = F.mse_loss(prediction, batch_u1)
```

This calculates the mean of `(prediction - target)**2` over all values in the batch. Smaller is better; zero means every sampled value matches.

A **gradient** measures how the loss changes when a weight changes. `loss.backward()` computes gradients for the learnable parameters, including the real and imaginary parts of the Fourier weights. `optimizer.step()` uses them to update the weights.

Our optimizer is **Adam**, with initial learning rate `0.003`. The learning rate controls update size. Adam also adapts updates using gradient history. The provided scheduler gradually reduces the learning rate across the run.

One **epoch** is one pass through the training examples. With 512 examples and batches of 32, one epoch has 16 updates; 128 epochs have 2,048 updates. Epochs are repeated learning passes, not physical time steps. Every target remains t=1. The FNO still predicts t=1 in one forward pass.

The operations and their roles follow the [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

## 2. The 12 TODOs

| TODO | Your task |
|---|---|
| 1 | Create Adam using this model's parameters and the supplied learning rate. |
| 2 | Set training mode. |
| 3 | Clear previous gradients. |
| 4 | Predict from the input batch. |
| 5 | Compute MSE against the answer batch. |
| 6 | Compute gradients. |
| 7 | Update weights. |
| 8 | Set evaluation mode. |
| 9 | Copy the current weights so the best version is preserved. |
| 10 | Save the supplied checkpoint dictionary to the supplied path. |
| 11 | Insert saved weights into the rebuilt model. |
| 12 | Put the loaded model in evaluation mode. |

Replace each `NotImplementedError` with one statement. Use function arguments: for example, `lr=learning_rate`, not a fixed number copied from an example.

**Common mistakes:**

- `model.train()` only selects a mode; the training loop does the learning.
- Gradients accumulate by default, so clear them for each ordinary training batch.
- Keep `loss` as a tensor for `backward()`. Use `loss.item()` only to obtain a number for logging.
- Call the optimizer's `step()` after `backward()`.
- `model.eval()` does not disable gradient tracking. The supplied `with torch.no_grad():` does that.

This FNO has no dropout or batch normalization, so switching modes currently does not change its prediction formula. Keeping the modes correct makes the training/evaluation code usable when the architecture changes.

## 3. Check progress without teaching from the test set

Training examples update weights. After each epoch, validation examples measure progress without updating weights. The driver preserves the weights with the **lowest mean validation relative L2 error**.

```text
relative L2 for one curve = length(prediction - target) / length(target)
```

Here “length” means the Euclidean norm of the curve's sampled values. The code calculates this separately for each curve, then averages the errors. It also reports the worst curve and MSE. A small denominator floor prevents division by zero; relative error is difficult to interpret for almost-zero targets, so read MSE too.

**`0.02`, displayed as `2%`, means 2% relative L2 error. It is not classification accuracy.**

The test set is evaluated after the best weights have been selected. The driver also compares two simple predictions: copying u0 unchanged and predicting its spatial mean everywhere. A useful model should improve substantially on these baselines. Choose training settings using validation; keep the test set for the final assessment.

## 4. Preserve and save what the model learned

`model.state_dict()` is a dictionary of parameter and buffer tensors. Use `copy.deepcopy(...)` to freeze a snapshot of the best weights: retaining the original dictionary alone would let later training change its tensors.

`torch.save` writes the checkpoint. Loading rebuilds the same architecture, calls `load_state_dict`, and selects evaluation mode. The checkpoint also records the equation, dataset settings, dataset fingerprint, and selected epoch. See [PyTorch saving and loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).

This checkpoint is for prediction. Exact training resumption would additionally require optimizer, scheduler, and random-generator states.

## 5. Run your exercise

In the VS Code terminal opened in your **Project A** folder:

```powershell
python lesson6_exercise.py --test-only
```

The checks cover actual weight updates, clearing gradients, correct MSE, evaluation without updates, independent snapshots, Fourier gradients, and identical predictions after saving/loading. They do not establish Burgers prediction accuracy.

When they pass:

```powershell
python lesson6_exercise.py --epochs 128
```

The supplied driver uses the CPU with two threads. It reads `lesson5_data/burgers_pairs.pt`; if missing, run your completed lesson 5 first. The existing PowerShell launcher also accepts `lesson6_exercise.py` and its arguments.

Your run creates **lesson6_results/**:

| File | What it contains |
|---|---|
| `burgers_fno.pt` | Your selected trained model and its settings. |
| `metrics.json` | Validation/test errors, baselines, and run settings. |
| `history.json` | Loss and validation error across epochs. |
| `new_example.pt` / `new_prediction.csv` | A fresh initial curve, reference answer, and FNO prediction. |
| `training_and_prediction.png` | Learning curves and a solution comparison, if Matplotlib is installed. |

The completed solution uses **lesson6_reference_results/** so your work has its own output folder. If your output already contains a checkpoint, use a new folder for another run:

```powershell
python lesson6_exercise.py --epochs 128 --output lesson6_try2
```

**Verified reference run:** the supplied solution completed 128 epochs using your lesson 4 architecture and lesson 5 dataset. Validation selected epoch 127. Test mean relative L2 was **1.94%**, with **11.92%** on the worst test curve; the two baselines were **98.19%** (copy input) and **62.49%** (spatial mean). Full numbers and settings are in [the reference metrics](lesson6_reference_results/metrics.json). These are measured errors for this dataset, not a guarantee for arbitrary inputs. Exact results can vary with software versions.

## 6. Use the trained AI

After your training run:

```powershell
python lesson6_predict.py
```

This provided tool loads your saved weights, generates a fresh demo curve, and compares the FNO with a reference answer. It saves results in **lesson6_prediction/**. There are no more TODOs in this tool.

To supply your own curve, first save 128 values sampled at `x = 0, 1/128, ..., 127/128`. For example, in a small Python script in Project A:

```python
import numpy as np
x = np.arange(128) / 128
u0 = 0.1 + 0.5 * np.sin(2 * np.pi * x)
np.save("initial.npy", u0)
```

Then run:

```powershell
python lesson6_predict.py --input initial.npy --output my_prediction
```

This path uses the **FNO alone to predict u(x,1)**; it does not compute a reference solution. `predicted_u1.npy` has shape `(1,128,1)`. The tool also accepts `(128,1)` or batches shaped `(B,128,1)` and saves a CSV; its plot shows the first curve.

The learned operator applies to our fixed viscosity, time, domain, and periodic boundaries. Its measured errors concern the smooth, bounded initial curves from lesson 5. Other equations or substantially different inputs need appropriate data, training, and evaluation.

## Before calling the project complete

Explain these in your own words:

1. What is the difference between `backward()` and `step()`?
2. Why clear gradients each batch?
3. Why use both `eval()` and `no_grad()`?
4. Why choose weights with validation rather than test error?
5. Why copy the best weights instead of retaining a live `state_dict()`?
6. What must be known about a new initial curve before using this checkpoint?

After the TODOs pass, your training run finishes, and you can load its checkpoint to predict a new curve, **you have completed the core FNO project**. Broader resolution tests and harder initial conditions are optional extensions.
