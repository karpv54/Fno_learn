"""Reload the trained FNO and predict from a sampled initial condition."""

from pathlib import Path
import torch
from train_heat_fno import load_trained_model, relative_l2


if __name__ == '__main__':
    model, equation = load_trained_model(Path(__file__).parent / 'heat_results' / 'heat_fno.pt')
    n = 128
    x = torch.arange(n, dtype=torch.float32) / n
    # Replace this formula with your own real float32 samples on [0, 1).
    u0 = (torch.sin(2 * torch.pi * x) + 0.3 * torch.cos(6 * torch.pi * x)).view(1, n, 1)
    with torch.no_grad():
        prediction = model(u0)
    alpha_t = equation['diffusivity'] * equation['final_time']
    exact = (
        torch.exp(torch.tensor(-alpha_t * (2 * torch.pi) ** 2)) * torch.sin(2 * torch.pi * x)
        + 0.3 * torch.exp(torch.tensor(-alpha_t * (6 * torch.pi) ** 2)) * torch.cos(6 * torch.pi * x)
    ).view(1, n, 1)
    print('Equation settings:', equation)
    print('Predicted solution shape:', tuple(prediction.shape))
    print('Relative L2 error for this known example:', relative_l2(prediction, exact).item())
    print('First five predicted values:', prediction[0, :5, 0].tolist())
