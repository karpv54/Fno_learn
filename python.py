import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
import fastai
import torch
import pandas as pd




from pathlib import Path
from urllib.request import urlretrieve
from scipy.io import loadmat

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"
data_path = data_dir / "burgers_data_R10.mat"

url = (
    "https://ssd.mathworks.com/supportfiles/"
    "nnet/data/burgers1d/burgers_data_R10.mat"
)

data_dir.mkdir(exist_ok=True)

if not data_path.exists():
    print("Téléchargement du dataset...")
    urlretrieve(url, data_path)
    print("Téléchargement terminé.")

data = loadmat(data_path)

u0 = data["a"]
uT = data["u"]

print("u0 :", u0.shape)
print("uT :", uT.shape)










nu = 0.1


# Dérivée première avec conditions périodiques
def first_derivative_periodic(q, dx):
    return (np.roll(q, -1) - np.roll(q, 1)) / (2 * dx)


# Dérivée seconde avec conditions périodiques
def second_derivative_periodic(q, dx):
    return (
        np.roll(q, -1)
        - 2 * q
        + np.roll(q, 1)
    ) / dx**2


# F(u) tel que u_t = F(u)
def burgers_rhs(u, dx):
    diffusion = nu * second_derivative_periodic(u, dx)

    flux = 0.5 * u**2
    convection = first_derivative_periodic(flux, dx)

    return diffusion - convection


# Une étape d'Euler explicite
def euler_step(u, dt, dx):
    return u + dt * burgers_rhs(u, dx)


def exact_burgers_solution(x, t, a=0.5):
    z = a * np.exp(-4 * np.pi**2 * nu * t)

    numerator = (
        4 * np.pi * nu * z * np.sin(2 * np.pi * x)
    )

    denominator = (
        1 + z * np.cos(2 * np.pi * x)
    )

    return numerator / denominator


