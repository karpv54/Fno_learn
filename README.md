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

