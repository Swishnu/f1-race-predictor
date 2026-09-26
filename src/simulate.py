"""Monte Carlo race simulation: predicted positions -> win / podium probabilities.

Each simulated race:
  1. every driver's "race score" = model's predicted finishing position + random noise
     (noise = everything the model can't see: strategy, tyre deg, mistakes, luck)
  2. some drivers DNF at random (crashes, reliability)
  3. optionally a safety car happens, which adds extra noise to shuffle the order
  4. sort by score -> finishing order

Repeat 20,000 times and count how often each driver wins or makes the podium.

The noise size (sigma) is not guessed. It's calibrated on the backtest:
we pick the sigma that gave past winners the highest probability.
"""
import numpy as np


def simulate(pred_pos, sigma, p_dnf, p_sc=0.0, sc_mult=1.0, n=20_000, seed=0):
    """
    pred_pos : array of predicted finishing positions (lower = better), one per driver
    sigma    : noise std-dev, in finishing positions
    p_dnf    : per-driver retirement probability
    p_sc     : probability of a safety car in the race
    sc_mult  : noise multiplier when a safety car happens
    Returns (p_win, p_podium) arrays aligned with pred_pos.
    """
    rng = np.random.default_rng(seed)
    pred_pos = np.asarray(pred_pos, dtype=float)
    k = len(pred_pos)

    sc = rng.random(n) < p_sc
    scale = np.where(sc, sigma * sc_mult, sigma)[:, None]         # (n, 1)
    score = pred_pos[None, :] + rng.normal(size=(n, k)) * scale   # (n, k)
    score[rng.random((n, k)) < p_dnf] = np.inf                    # DNFs go to the back

    order = np.argsort(score, axis=1)
    p_win = np.bincount(order[:, 0], minlength=k) / n
    p_podium = np.bincount(order[:, :3].ravel(), minlength=k) / n
    return p_win, p_podium
