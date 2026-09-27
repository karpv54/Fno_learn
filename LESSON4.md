# Lesson 4: build the complete FNO

You have three core lessons left, including this one:

| Lesson | What you will build | Completion point |
| --- | --- | --- |
| 4 | Lifting, parallel FNO blocks, and output projection | A correctly connected, trainable network |
| 5 | Burgers initial/final training pairs | Checked data and independent train/validation/test sets |
| 6 | Training, evaluation, saving, and prediction | Your own trained u(x,0) -> u(x,1) solver |

After these, an optional lesson 7 can explore resolution transfer, difficult
initial conditions, and improving accuracy. The three core lessons cover the
first solver for our fixed equation, rather than every possible FNO application.

## Today's target

Build a network whose input and output both have shape `(B,N,1)`.
Its intended task is Burgers with viscosity 0.01 on the periodic unit interval:

`u_t + u*u_x = 0.01*u_xx`, with input at t=0 and output at t=1.

The forward method takes only `u0`, the sampled initial profile. It creates the
coordinate feature internally. The number 0 alone is not an initial profile;
`u0` contains one value of the function at each spatial point.

Complete **lesson4_exercise.py**. It imports your completed `SpectralConv1d`
from `lesson3_exercise.py`, so keep the files together. The separate
`lesson4_solution.py` is available for checking your attempt.

## The complete path through the network

```text
u0                     (B,N,1)
add coordinate x       (B,N,2)
lift                   (B,N,C)
swap axes              (B,C,N)
apply each FNO block    (B,C,N)
swap axes back         (B,N,C)
project                (B,N,1)
```

For today's defaults, `C=16`, `modes=12`, and `depth=3`.
Within each FNO block, the same hidden tensor enters both the local and
learned Fourier paths. Add their real spatial outputs, then apply GELU:

`next_hidden = GELU(local(hidden) + spectral(hidden))`.

The spectral layer from lesson 3 now returns one reconstructed tensor. It
does not return the three-item diagnostic tuple from lesson 2.

The two branches within one block run in parallel. The complete blocks run
one after another, so each block processes the previous block's output.
Each block learns its own parameters. Depth counts hidden transformations,
not physical time steps. All blocks together will learn the direct t=0 to t=1
mapping once trained.

## The two new containers

`nn.ModuleList` stores a collection of layers and registers their parameters.
You explicitly decide how to use them in `forward`. A normal Python list alone
does not register its contained layers with the parent model. Registration is
what makes those parameters available through `model.parameters()` and the
model state dictionary.

For example, this stores three separate linear layers:

```python
self.layers = nn.ModuleList([
    nn.Linear(5, 5) for _ in range(3)
])
```

The list comprehension calls the constructor each time, giving each layer
independent weights. Apply the stored layers explicitly:

```python
for layer in self.layers:
    values = layer(values)
```

Use the same pattern for your FNO blocks, with `depth` controlling their count.
See [PyTorch ModuleList](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html).

`nn.Sequential` registers layers and also applies them in their listed order.
This suits the projection, which is a simple chain:

```python
self.example_chain = nn.Sequential(
    nn.Linear(5, 8),
    nn.GELU(),
    nn.Linear(8, 1),
)
```

Calling `self.example_chain(values)` runs all three operations. `nn.GELU()`
constructs the activation module placed inside this container; `F.gelu(values)`
directly applies the activation to values. See
[PyTorch Sequential](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html).

Your actual projection is `width -> projection_width -> 1`, with GELU between
the two linear layers. The default `projection_width` is 32.

## Why projection is needed

Each spatial point currently has C hidden features. Projection learns how to
turn those features into one scalar prediction. It applies the same learned
transformation at every point; it preserves the batch and spatial axes.

`nn.Linear` operates on the last axis, so change `(B,C,N)` back to `(B,N,C)`
before projecting. There is no need to flatten the spatial grid.
See [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).

The final layer is linear, with no activation after it. This allows unrestricted
signed values of u. GELU can produce some negative values, but applying it to
the final output would alter and restrict the possible predictions.

## Your ten TODOs

1. Create the local branch in `self.local`.
2. Create the learned spectral branch in `self.spectral`.
3. Combine both branches with GELU and return the block output.
4. Create the lifting layer in `self.lift`.
5. Create independent FNO blocks inside `self.blocks` using ModuleList.
6. Create the three-layer projection in `self.project` using Sequential.
7. Combine the initial values and supplied coordinates into `features`.
8. Lift those features and reorder their axes into `hidden`.
9. Apply the current block to `hidden` inside the supplied loop.
10. Reorder the axes back, project, and return the prediction.

Replace each `NotImplementedError` with one Python statement. A statement may
span several lines inside parentheses. Keep the automatic tests unchanged.
Only the input feature count 2 and output feature count 1 are fixed; use the
provided variables for widths, modes, and depth. The grid and block loop are
already provided.

In a terminal opened in the lesson folder, run:

```text
python lesson4_exercise.py
```

With this workspace's existing runtime, you can also use:

```powershell
.\run_lesson.ps1 lesson4_exercise.py
```

The file intentionally raises `NotImplementedError` until the TODOs are filled.
The completed solution was run with PyTorch 2.8.0+cpu and passed all checks.
The checks cover known frequency content passing through both branches,
coordinates, odd and even grids, independent registered blocks, gradients,
and an unrestricted signed output. Passing them means the architecture works;
the dummy targets in the gradient check do not train a Burgers solver.

## Questions to answer after the exercise

1. Why does lifting take two features while projection returns one?
2. Why use ModuleList for the blocks? Does it apply the blocks automatically?
3. Where do we swap axes, and why?
4. Why is there no activation after the final linear layer?
5. Does depth=3 mean three physical time steps?
6. What information is missing before this network can learn the Burgers map?

Next, lesson 5 will supply the paired initial and final profiles that make
this architecture a learner for our specific equation.
