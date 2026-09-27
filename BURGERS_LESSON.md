# Your target: Burgers from t=0 to t=1

We are learning the solution operator for

\[
u_t + u u_x = 0.01 u_{xx},\qquad x\in[0,1),\quad t\in[0,1],
\]

with periodic boundary conditions and initial condition `u(x,0)=u0(x)`.
The viscosity, domain, and boundaries are the defaults you confirmed.
The target time is **1**. This is the initial-to-final mapping described in
the [FNO Burgers example](https://neuraloperator.github.io/dev/theory_guide/fno.html#burgers-equation).

The network receives an entire sampled initial profile and predicts an entire
sampled final profile:

```text
u0[b, j, 0] = u for example b, at point x_j, at time 0
target[b, j, 0] = u for the same example and point, at time 1
```

Both tensors have shape `(B, N, 1)`. A time value of zero alone does not specify
an initial condition. The input must contain all the values of `u0` on the grid.
Our lifting layer combines those values with position, giving `(B, N, 2)`.

## How this connects to lessons 2 and 3

Your lesson 2 corrections and lesson 3 exercise remain applicable. The FNO
block structure does not prescribe which PDE is learned. The training pairs,
loss, and training determine the learned solution map.

In lesson 2, the Fourier branch copies low frequencies. In lesson 3, it learns
complex matrices that mix hidden channels at each retained frequency. A full
network stacks these blocks and projects back to one output value per point.

```text
[u0(x), x] -> lifting -> FNO blocks -> projection -> predicted u(x,1)
```

An FNO block is a hidden computational layer. Four blocks do not mean four
time steps or times 0.25, 0.5, 0.75, and 1. The trained network predicts the final
profile directly. Since all labels are at t=1, this example needs no explicit
time feature. Predicting arbitrary times would require a different training
setup, such as including time as an input and training at multiple times.

The nonlinear term `u*u_x` lets the profile transport and steepen itself, while
the diffusion term `0.01*u_xx` smooths gradients. This creates a nonlinear
solution map. The simple frequency-decay formula applied directly to `u0` in
the heat example does not produce Burgers labels.

## Your immediate exercise

Continue with `lesson3_exercise.py`. The six TODOs build the learned spectral
layer needed by the Burgers FNO. Its full solution is `lesson3_solution.py`.
Before moving on, explain the following shapes without looking at the code:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Raw profile | `(B,N,1)` | Values at t=0 |
| Input with coordinates | `(B,N,2)` | Initial values and x |
| Lifted, channels first | `(B,C,N)` | Hidden spatial features |
| Full rFFT spectrum | `(B,C,N//2+1)` | Complex Fourier coefficients |
| Learned weights | `(C,C,M)` | Channel matrix for each retained mode |
| Final prediction | `(B,N,1)` | Predicted values at t=1 |

The original lesson uses `C=16` and `M=12`. The complete training example uses
`C=24` and `M=16`; these are model settings, not PDE parameters.

## The complete training reference

- `fno_model.py`: the shared FNO architecture and error measures.
- `burgers_data.py`: initial conditions and a checked Burgers reference solver.
- `train_burgers_fno.py`: training, validation, final testing, and checkpoint saving.
- `predict_burgers.py`: load the checkpoint and predict at t=1 from a new profile.

The dataset contains 512 training functions, 64 validation functions, and
64 held-out test functions. These are independently sampled finite sums of
eight sine/cosine harmonics, with means between -0.3 and 0.3. The sum of absolute
oscillatory coefficients is sampled between 0.2 and 1.0. This bounds the example
distribution; it is not the published FNO benchmark dataset.

Reference solutions use a 512-point grid, then are sampled at the model's
128 grid points. A 1024-point reference checks spatial convergence. A separate
dealiased Fourier/RK4 solver checks the Cole-Hopf implementation at t=1. We
also verify constants, the initial condition, mass conservation, and energy
dissipation. The trained network is evaluated on 256 points using the same
held-out continuous functions as the 128-point test.

The reference solver uses the Cole-Hopf transformation: for zero mean, a
nonlinear change of variable turns Burgers into a heat equation for an
auxiliary function. A moving coordinate handles nonzero means. Transforming
back gives Burgers solutions. This auxiliary heat equation is part of the
Burgers reference calculation; the training labels are velocity profiles for
Burgers. See the [NYU derivation](https://math.nyu.edu/~tabak/PDEs/The_Burgers-Equation.pdf).

The analytic transformation is evaluated numerically with FFTs, so the label
accuracy is checked rather than treated as exact. Its finite-precision
conditioning also depends on viscosity and input amplitude; the current
implementation is designed and checked for the fixed example distribution.
Neither the reference solver nor a PDE formula is called by the FNO forward
method. They generate/check answers used for training and evaluation.

For supervised training, the central operation is:

```python
prediction_at_t1 = model(initial_profile_at_t0)
loss = F.mse_loss(prediction_at_t1, target_profile_at_t1)
```

Use validation error to choose the saved epoch. Test errors are measured after
that choice. Report relative L2 error rather than classification accuracy.

## Run the lessons

In a terminal opened in this folder, use your Python with PyTorch and NumPy:

```text
python lesson3_exercise.py
python burgers_data.py
python train_burgers_fno.py
python predict_burgers.py
```

The exercise intentionally stops until its TODOs are complete. The full
training reference imports the completed spectral layer from lesson 3.
Training defaults to CPU, 128 epochs, and two computation threads.

The existing workspace launcher also supports the new files:

```powershell
.\run_lesson.ps1 train_burgers_fno.py
.\run_lesson.ps1 predict_burgers.py
```

Results are saved under `burgers_results/`: `burgers_fno.pt`, `metrics.json`,
`history.json`, and `new_prediction.csv`. The checkpoint records viscosity,
initial time, final time, boundary type, domain length, and architecture.
The previous heat-equation checkpoint does not apply to this PDE.

The completed CPU run used 128 epochs and selected epoch 123 by validation
error. On 64 held-out initial functions, mean relative L2 error was **1.3440%**
at 128 grid points (worst case 4.5967%), and **1.3820%** on the same functions
at 256 points. Copying the initial profile gave 95.7301% mean relative L2 error.
Saving, reloading, and inference passed; the separate example in
`predict_burgers.py` had 1.4533% relative L2 error. These results apply to the
synthetic input distribution above, not arbitrary Burgers initial conditions.

After completing lesson 3, your next lesson is to assemble `FNO1d` and explain
how it returns one value at every spatial point. Then work through the paired
Burgers dataset and the training loop. The full training file is a reference
for that destination, not a requirement to learn every part at once.
