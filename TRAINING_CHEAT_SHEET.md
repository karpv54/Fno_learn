# Lesson 6 — training cheat sheet

**The FNO guesses u(x,1); training adjusts its weights using the reference answer.**

## The tools

| Code | Meaning |
|---|---|
| `model.parameters()` | The model's learnable weights and biases. |
| `torch.optim.Adam(model.parameters(), lr=learning_rate)` | Create the optimizer that will update them. |
| `model.train()` | Select training mode; it does not perform learning. |
| `optimizer.zero_grad()` | Clear gradients left by the previous batch. |
| `prediction = model(batch_u0)` | Predict the final curves. |
| `loss = F.mse_loss(prediction, batch_u1)` | Average squared error against the correct answers. |
| `loss.backward()` | Calculate how the loss depends on each weight. |
| `optimizer.step()` | Use those gradients to update the weights. |
| `loss.item()` | Convert a scalar tensor to a Python number for printing. |

**Order matters:** clear → predict → loss → backward → step. [PyTorch training reference](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

## Checking predictions

```python
model.eval()
with torch.no_grad():
    prediction = model(batch_u0)
```

`eval()` selects evaluation mode. `no_grad()` stops recording gradients, saving memory. Both appear in validation, testing, and prediction; no optimizer update happens there.

## Saving and loading

| Code | Meaning |
|---|---|
| `model.state_dict()` | A dictionary containing current weights and buffers. |
| `copy.deepcopy(model.state_dict())` | An independent snapshot that later updates cannot change. |
| `torch.save(checkpoint, path)` | Write the checkpoint dictionary to a file. |
| `torch.load(path, map_location="cpu", weights_only=True)` | Read the saved checkpoint onto the CPU. |
| `FNO1d(**architecture)` | Unpack the settings dictionary into the model constructor. |
| `model.load_state_dict(checkpoint["model_state"])` | Insert the saved weights into that model. |

The driver supplies `checkpoint`, `path`, and `architecture`. Use those arguments in your TODOs. [PyTorch checkpoint reference](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).

## Four words to remember

| Word | Meaning in this lesson |
|---|---|
| Batch | 32 paired curves processed together. |
| Epoch | One pass over all 512 training examples: 16 updates. |
| Learning rate | Controls optimizer update size; initially 0.003. |
| Relative L2 error | Error norm divided by target norm; 0.02 means 2% error. |

Train updates weights. Validation chooses the best saved version. Test measures that chosen version on unseen examples. **All targets are t=1, regardless of epoch number.**
