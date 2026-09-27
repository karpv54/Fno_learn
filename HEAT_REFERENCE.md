# From your lesson 2 to a trained Fourier Neural Operator

Start with `lesson2_corrected.py`, then complete `lesson3_exercise.py`.
Use `lesson3_solution.py` to check your work. `train_heat_fno.py` is a complete
reference for the later steps: data, architecture, training, evaluation, saving,
and inference. You do not need to understand that entire file before lesson 3.

Verified in this workspace with PyTorch 2.8.0+cpu: all original lesson 2 tests
pass, and lesson 3 passes its signal and gradient checks. A 100-epoch run gives
**0.8336% mean relative L2 error on 64 held-out functions at 128 grid points**,
and **0.8339% on the same continuous functions at 256 points**. Copying the
initial condition gives 80.2225% on the 128-point test set. The checkpoint was
saved, reloaded, and used for a separate known input (0.6836% relative L2 error).
These measurements describe this synthetic heat-equation distribution only.
Exact metrics and the training history are in `heat_results/`.

Your first concrete target is the operator **initial temperature function →
temperature function at a fixed later time**, for a periodic 1D heat equation.
An operator maps a whole function to another function. The model receives
sampled numerical values, rather than an equation written as text. FNOs learn
families of solution maps from examples; this is the central idea of the
[original FNO paper](https://arxiv.org/abs/2010.08895).

## Your lesson 2 corrections

TODOs **3 and 8 are correct**. Your `rfft` statement in TODO 4 is also correct;
the unnecessary line before it is the problem.

| TODO | Correct statement | Why your version fails |
| --- | --- | --- |
| 1 | `return x.permute(0, 2, 1)` | Your input is already lifted to `(B, N, WIDTH)`. The requested view `(B, 2, N)` has the wrong number of elements. A view also cannot swap the meaning of axes. There is no return statement, so even a successful view would leave the function returning `None`. |
| 2 | `self.local = nn.Conv1d(width, width, kernel_size=1)` | `2` describes the raw input features. The local branch receives `width` hidden channels after lifting. |
| 3 | `return self.local(x)` | Correct. |
| 4 | `x_fourier = torch.fft.rfft(x, dim=-1)` | Delete `torch.zeroes(...)`: this function does not exist, and rFFT creates its own output tensor. |
| 5 | `filtered_fourier = torch.zeros_like(x_fourier)` | Keep the full spectrum shape, complex dtype, and device. With `N=128`, its last dimension is 65, even if only 12 coefficients will be kept. Allocating 12 float32 values both changes the shape and loses imaginary information. |
| 6 | `filtered_fourier[..., :self.modes] = x_fourier[..., :self.modes]` | `[:,:]` selects all of the first two axes and leaves the last axis unrestricted. Your statement copies every frequency rather than keeping only low frequencies. |
| 7 | `reconstructed = torch.fft.irfft(filtered_fourier, n=x.size(-1), dim=-1)` | `ifft` expects a full complex spectrum. You have the one-sided spectrum from `rfft`. You also inverted the unfiltered spectrum. |
| 8 | `local_part = self.local(x)` | Correct. |
| 9 | `fourier_part, _, _ = self.fourier(x)` | The function returns three tensors: spatial reconstruction, full spectrum, filtered spectrum. Unpack the first for addition to the local branch. |
| 10 | `output = F.gelu(local_part + fourier_part)` | `nn.GELU(...)` constructs a module; its first argument configures the approximation. It does not apply GELU to that tensor. Also, element `[1]` of your original tuple is a complex spectrum of length 65, which cannot be added to spatial values of length 128. |

In tuple unpacking, `_` is an ordinary Python variable used by convention for
an unneeded value. It does not do any special tensor processing.

The axis numbers passed to `permute` refer to the **old** axes in their new
order: keep batch axis 0, then channel axis 2, then spatial axis 1.
See [PyTorch's permute documentation](https://docs.pytorch.org/docs/stable/generated/torch.permute.html).

For example, two points with features `[a, b, c]` and `[d, e, f]` must become
three channel sequences `[a, d]`, `[b, e]`, `[c, f]`. Merely regrouping the
flattened values into three rows gives `[a, b]`, `[c, d]`, `[e, f]` instead.
Correct dimensions alone are therefore insufficient: axis meaning matters.

The local branch computes, at each point `j`,

`local[b, o, j] = bias[o] + sum_i weight[o, i, 0] * hidden[b, i, j]`.

It mixes features at the same point and uses the same weights everywhere.
Its kernel length of one does not make it an identity: with width 16 it has
256 weights and 16 biases. See [Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html).

The corrected block follows this shape flow:

```text
raw [u0(x), x] : (B, N, 2)
       linear : (B, N, C)
      permute : (B, C, N)
                    |
           +--------+----------+
           |                   |
      Conv1d(C,C,1)            rFFT
       (B,C,N)         (B,C,N//2+1), complex
           |             keep indices 0..M-1
           |             zero all other indices
           |                  irFFT(n=N)
           |                  (B,C,N), real
           +--------- add -----+
                       |
                      GELU
                   (B,C,N)
```

Both branches must be real spatial tensors of the same shape **before** the
addition. Your Fourier branch has no learnable parameters yet; the local
branch and lifting layer already do.

## Answers to the five lesson 2 questions

1. **Why 65 coefficients?** For a real signal, negative-frequency coefficients
   are complex conjugates of positive-frequency coefficients. rFFT stores only
   indices `0` through `N//2`: `128//2 + 1 = 65`. Index 0 is the constant (DC)
   component; index 64 is the Nyquist component for this even grid size.
   [PyTorch rFFT](https://docs.pytorch.org/docs/stable/generated/torch.fft.rfft.html)
   explains the conjugate symmetry.
2. **Why complex?** A Fourier coefficient represents both the size of an
   oscillation and its phase. Equivalently, real and imaginary components
   encode cosine and sine contributions with a convention-dependent sign.
3. **Why specify `n`?** A one-sided spectrum length does not uniquely determine
   whether the input had an odd or even length. An input of length 127 has 64
   rFFT coefficients; without `n=127`, irFFT defaults to 126 output points.
   See [PyTorch irFFT](https://docs.pytorch.org/docs/stable/generated/torch.fft.irfft.html).
4. **Why pointwise?** A kernel of length one reads only the channels at the
   current spatial point. It does not read neighboring points.
5. **What is missing?** Trainable complex weights that mix channels at each
   retained frequency. Simply copying low frequencies always applies the same
   filter, regardless of the desired solution.

`MODES=12` means **12 retained indices: 0 through 11**, including the constant
component. It does not mean retaining frequencies 1 through 12.

## Lesson 3: learn what to do with each frequency

Lesson 2 uses `output_fourier[..., :M] = x_fourier[..., :M]`.
Lesson 3 changes this to a trainable linear transformation of channels at each
mode, following the [FNO Fourier-layer construction](https://neuraloperator.github.io/dev/theory_guide/fno.html):

`output_fourier[b, o, k] = sum_i x_fourier[b, i, k] * weights[i, o, k]`.

The letters mean:

| Letter | Meaning | Example size |
| --- | --- | --- |
| b | Function in the batch | 4 |
| i | Input hidden channel | 16 |
| o | Output hidden channel | 16 |
| k | Retained Fourier mode | 12 |

The learnable tensor has shape `(16, 16, 12)`. Think of **12 separate 16-by-16
complex matrices**, one per mode. The trainable weights contain no batch axis
and no spatial-grid-size axis.

The compact implementation is:

```python
self.weights = nn.Parameter(
    (1 / width) * torch.randn(width, width, modes, dtype=torch.cfloat)
)

# Inside forward, after rFFT and allocating the full output spectrum:
output_fourier[..., :self.modes] = torch.einsum(
    'bik,iok->bok',
    x_fourier[..., :self.modes],
    self.weights,
)
```

In `einsum`, `i` is absent from the output labels, so it is summed over.
The batch `b` and mode `k` are preserved; `o` selects the output channel.
This operation mixes **channels at one frequency**. It does not directly mix
different frequencies. Nonlinear activations in physical space can generate
new frequencies and couple frequency content between layers.
See [PyTorch einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html)
for the index notation.

`nn.Parameter` registers the tensor so that `model.parameters()` exposes it to
the optimizer. The FFT itself is not trained. Backpropagation passes through
the FFT operations to update the spectral weights. Use a real scalar loss,
such as the mean squared error of the real reconstructed output.

With a complex scalar weight `2+0j`, the corresponding coefficient doubles.
With `0+1j`, its phase shifts. For a cosine, the latter produces negative sine
under PyTorch's FFT convention. The exercise tests both of these behaviors.

Two details for later: DC and, for even N, Nyquist coefficients must be real
for a real signal. `irfft` ignores their imaginary parts, so those particular
imaginary parameters have no effect. Also, zeroing high modes only constrains
the spectral branch; the local branch and activation mean the entire FNO
block is not necessarily band limited.

Complete the six TODOs in `lesson3_exercise.py`, one statement each. Keep the
tests unchanged. New tools are `nn.Parameter`, `torch.randn`, complex dtype,
and `torch.einsum`. Read the solution only after trying the exercise.

You are ready for lesson 4 when you can explain why:

- identity channel matrices at each mode reproduce lesson 2;
- the parameters do not grow when N changes from 128 to 256;
- the returned signal is real even though the weights are complex;
- an optimizer must update registered weights, not the temporary FFT output.

## Lessons 4–6: build and train your first complete solver

The reference in `train_heat_fno.py` implements these steps. Work through the
functions in the following order.

**Lesson 4 — assemble `FNO1d`.** Lift two input features to 16 hidden channels,
apply three FNO blocks, then project back to one output value per spatial
point. The final projection has no final GELU: the solution must be able to
take positive and negative values. Hidden layers still use GELU.

**Lesson 5 — understand `sample_heat_pairs`.** The PDE is

`u_t = alpha * u_xx`, with `alpha = 0.01`, `T = 0.5`, and `x in [0, 1)`.

The boundary is periodic: the value and spatial derivative agree at the two
ends of the continuous interval. The sample grid omits x=1 because it is the
same periodic point as x=0. Adjacent endpoint samples need not be identical.

For a Fourier basis function `exp(i*2*pi*k*x)`, differentiating twice in x
multiplies it by `-(2*pi*k)^2`. Substituting into the PDE gives the scalar ODE

`d u_hat_k / dt = -alpha * (2*pi*k)^2 * u_hat_k`.

Its solution is

`u_hat_k(T) = exp(-alpha * (2*pi*k)^2 * T) * u_hat_k(0)`.

That gives exact labels, up to floating-point rounding, for the finite sums
of sines and cosines in this dataset. Every input has independently sampled
coefficients. This heat problem has a simple exact solver; it is a teaching
benchmark, not evidence that an FNO is preferable to that solver here.

**Lesson 6 — train, validate, then test.** The training loop performs:

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(batch_x)
loss = F.mse_loss(prediction, batch_y)
loss.backward()
optimizer.step()
```

The network sees `batch_x`. The correct answers `batch_y` are used only to
calculate the loss. The PDE label formula is not inside the network's forward
method. Checkpoint inference calls only the trained network.

The 256 training, 64 validation, and 64 test examples are different initial
functions. The best validation epoch chooses the saved weights. The test set
is evaluated after that choice. Do not split spatial points from one function
across training and test sets: that would not test a new PDE instance.

Mean squared error is the training loss. We also report each function's
relative L2 error, `norm(prediction - target) / norm(target)`, averaged over
functions, plus its maximum and the mean absolute error in conserved mass.
This is prediction error, not classification accuracy. Relative errors are
unstable near zero targets; the implementation guards the denominator.

The script compares with copying the initial state and predicting its spatial
mean. It also evaluates the same continuous test functions on 256 points.
An FNO can accept a new uniform resolution on this same domain, but its
accuracy there must be measured. Accepting a shape is not a guarantee of
accurate resolution transfer.

The coordinate feature follows your original `[u0, x]` design. The ramp `x`
itself is not periodic and can introduce a jump in the FFT representation.
For this homogeneous periodic equation, try removing coordinates entirely
(change the lifting input size to 1), or use periodic coordinate features
`sin(2*pi*x)` and `cos(2*pi*x)` (lifting input size 3). Compare validation errors.

## Running the files

Use your Python environment with PyTorch. In a terminal opened in this folder:

```text
python lesson2_corrected.py
python lesson3_exercise.py
python lesson3_solution.py
python train_heat_fno.py
python predict_heat.py
```

The exercise intentionally raises `NotImplementedError` until its TODOs are
completed. All other files are complete. A reproducible dependency is listed
in `requirements.txt`; an existing compatible PyTorch installation can also
be used. Training defaults to CPU and 100 epochs. `--epochs 2` checks that a
short run completes, but does not establish useful prediction quality.

The training script writes `heat_results/heat_fno.pt`, `metrics.json`,
`history.json`, and `new_prediction.csv`. The saved checkpoint includes the
architecture, equation settings, and learned weights. `predict_heat.py`
shows how to load it and supply an initial condition of shape `(B, N, 1)`.

This workspace also has `run_lesson.ps1`, which uses the available bundled
Python and the project-local PyTorch installation. For example:

```powershell
.\run_lesson.ps1 lesson2_corrected.py
.\run_lesson.ps1 lesson3_solution.py
.\run_lesson.ps1 train_heat_fno.py
.\run_lesson.ps1 predict_heat.py
```

## After the first trained heat model

1. **Inspect generalization.** Plot several test predictions against exact
   solutions. Look at the worst cases as well as the average. Test frequencies
   and amplitudes outside the training distribution, and report those errors
   separately. Compare validation results when changing modes, width, and depth.
2. **Condition on equation parameters.** To predict different diffusivities
   or final times with one model, sample them during training and include
   them as input features broadcast across x. Retrain; changing checkpoint
   metadata alone does not change what the model learned.
3. **Move to viscous Burgers' equation.** Use
   `u_t + u*u_x = nu*u_xx` with a clearly specified periodic domain and positive
   viscosity. Generate labels with a validated numerical solver and check
   resolution/time-step convergence before trusting its data. This introduces
   nonlinear solution behavior while keeping the architecture familiar.
4. **Predict trajectories.** Choose direct prediction conditioned on time, or
   train a short-step map and roll it forward. For the latter, test long
   rollouts: small step errors can accumulate. A fixed-final-time heat model
   does not automatically solve arbitrary trajectories.
5. **Extend the geometry and boundary conditions.** A 2D grid needs 2D spectral
   layers and careful frequency indexing. Nonperiodic boundaries need an
   explicit treatment; adding coordinates or using a 1x1 convolution does
   not enforce a boundary condition. Unstructured meshes need a suitable
   representation beyond this standard uniform-grid FFT implementation.

The completed example solves one specified family approximately: smooth
periodic heat problems with fixed diffusivity and final time. A model for
your eventual equation will require the corresponding inputs, outputs,
training distribution, boundary treatment, and validation.
