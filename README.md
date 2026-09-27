# Your FNO lessons: Burgers from t=0 to t=1

Your agreed target is **u(x,0) → u(x,1)** for the viscous Burgers equation
`u_t + u*u_x = 0.01*u_xx` on the periodic unit interval `x in [0,1)`.
The viscosity, domain, and boundary conditions are the defaults you confirmed.

Start with `lesson2_corrected.py`, then complete `lesson3_exercise.py`.
Use `lesson3_solution.py` to check your work. Those layers apply to Burgers
without changing the lesson 2 corrections or the lesson 3 exercise.

**Current exercise: [lesson 6 — the final core lesson](LESSON6.md)**.
Your completed lesson 5 passes its checks. Start with
[the training cheat sheet](TRAINING_CHEAT_SHEET.md), then complete the 12 TODOs
in `lesson6_exercise.py` to train, evaluate, save, and reload your FNO.
The provided `lesson6_predict.py` loads a checkpoint to predict new curves.
The tested solution and reference results are separate from your exercise outputs.

Read **[the Burgers lesson](BURGERS_LESSON.md)** for the input/output meaning,
the full architecture, checked training data, training, and saved-model inference.
The complete reference is `train_burgers_fno.py` and the inference example is
`predict_burgers.py`. The earlier heat example and its measurements are retained
in `HEAT_REFERENCE.md`; they do not measure Burgers performance.

An operator maps a whole function to another function. In this task, each input
is a sampled initial velocity profile and each target is its solution at t=1.
All original lesson 2 tests and the lesson 3 signal/gradient tests passed with
PyTorch 2.8.0+cpu.

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

## Continue toward your Burgers solver

1. Complete the learned spectral layer in `lesson3_exercise.py`.
2. Assemble lifting, FNO blocks, and projection; study `fno_model.py`.
3. Understand the paired `(u0, u1)` dataset in `burgers_data.py`.
4. Train and validate with `train_burgers_fno.py`.
5. Load the checkpoint and predict from a new initial function using `predict_burgers.py`.

The complete explanation and runnable commands are in [BURGERS_LESSON.md](BURGERS_LESSON.md).
Hidden FNO layers are not physical time steps: this model learns a direct
initial-to-final map at fixed viscosity and fixed final time.

For the existing workspace installation, a terminal in this folder can run:

```powershell
.\run_lesson.ps1 lesson2_corrected.py
.\run_lesson.ps1 lesson3_exercise.py
.\run_lesson.ps1 train_burgers_fno.py
.\run_lesson.ps1 predict_burgers.py
```

The exercise intentionally stops until its TODOs are complete. The training
reference uses the completed spectral layer. With your own Python environment,
run the same Python files directly; dependencies are in `requirements.txt`.

Results and the trained checkpoint are saved in `burgers_results/`.
The verified 128-epoch run achieved **1.3440% mean relative L2 test error** on
64 held-out functions at 128 points, and 1.3820% at 256 points. Saving and
reloading the Burgers model for inference passed.
The reference solver is checked for the chosen smooth input distribution;
performance on other initial conditions or viscosities needs separate evaluation.
