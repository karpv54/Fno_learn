"""Use the trained Burgers FNO on an initial profile at t=0."""

from pathlib import Path
import numpy as np
import torch
from burgers_data import initial_profile, solve_burgers
from fno_model import relative_l2
from train_burgers_fno import load_trained_model


if __name__ == '__main__':
    torch.set_num_threads(2)
    model, equation = load_trained_model(Path(__file__).parent / 'burgers_results' / 'burgers_fno.pt')
    # This example starts from u(x,0) = 0.1 + 0.5*sin(2*pi*x) + 0.1*cos(4*pi*x).
    coefficients = np.zeros((1, 2, 8))
    coefficients[0, 0, 0] = 0.5
    coefficients[0, 1, 1] = 0.1
    mean = np.array([[0.1]])
    u0 = torch.from_numpy(initial_profile(coefficients, mean, 128)[..., None].astype(np.float32))
    with torch.no_grad():
        prediction_at_t1 = model(u0)
    print('Equation settings:', equation)
    print('Input u(x,0):', tuple(u0.shape))
    print('Predicted u(x,1):', tuple(prediction_at_t1.shape))
    # The reference is used here only to measure the example's prediction error.
    reference = solve_burgers(coefficients, mean, n=512)[:, ::4]
    reference = torch.from_numpy(reference[..., None].astype(np.float32))
    print('Relative L2 error:', relative_l2(prediction_at_t1, reference).item())
