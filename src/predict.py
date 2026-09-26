"""Backtest on 2026 so far, calibrate the simulation, and predict the next race.

Walk-forward backtest: to "predict" round N we train only on rounds 1..N-1,
exactly as if we'd run this the night before that race. No peeking.

Run:  python -m src.predict     -> prints results, writes docs/data/prediction.{json,js}
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import features, model
from .simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "prediction.json"

FIRST_BACKTEST_ROUND = 5          # need a few races of history before predicting
SIGMAS = np.arange(0.5, 6.01, 0.25)

# Baku assumptions (not learned from 2026 data, since there's only one race per track):
# safety cars are common on street circuits, and when one happens the order gets shuffled.
BAKU_P_SC = 0.6
BAKU_SC_MULT = 1.3


def dnf_rate(df):
    done = df[df["finish_pos"].notna()]
    return float(done["dnf"].astype(float).mean())


def walk_forward(df, cols):
    """Predicted finishing positions for each past race, trained only on earlier races."""
    rows = []
    last = int(df.loc[df["finish_pos"].notna(), "round"].max())
    for rnd in range(FIRST_BACKTEST_ROUND, last + 1):
        train, race = df[df["round"] < rnd], df[df["round"] == rnd].copy()
        m = model.fit(train, cols)
        race["pred_pos"] = model.predict(m, race, cols)
        race["p_dnf"] = dnf_rate(train)
        rows.append(race)
    return pd.concat(rows)


def calibrate_sigma(bt):
    """Pick the noise level whose podium probabilities best matched reality
    (minimise log-loss of "made the podium: yes/no" for every driver).

    Why podium and not winner?  Scoring only the winner uses 1 outcome per race.
    In a season where pole usually wins, that keeps rewarding smaller noise all
    the way to zero, i.e. an overconfident model. Podium gives 3x the evidence
    and punishes overconfidence on the misses too, so it finds a real optimum.
    """
    scores = {}
    for sigma in SIGMAS:
        ll = []
        for _, race in bt.groupby("round"):
            _, p_pod = simulate(race["pred_pos"], sigma, race["p_dnf"].iat[0], n=5_000)
            p_pod = np.clip(p_pod, 1e-4, 1 - 1e-4)
            pod = race["finish_pos"].values <= 3
            ll.append(-np.log(p_pod[pod]).mean() - np.log(1 - p_pod[~pod]).mean())
        scores[float(sigma)] = float(np.mean(ll))
    best = min(scores, key=scores.get)
    return best, scores


def backtest_report(bt, sigma):
    races = []
    for rnd, race in bt.groupby("round"):
        p_win, _ = simulate(race["pred_pos"], sigma, race["p_dnf"].iat[0])
        race = race.assign(p_win=p_win).sort_values("p_win", ascending=False)
        winner = race[race["won"] == 1].iloc[0]
        races.append({
            "round": int(rnd),
            "event": winner["event"],
            "winner": winner["driver"],
            "winner_grid": int(winner["grid"]),
            "model_pick": race.iloc[0]["driver"],
            "p_winner": round(float(winner["p_win"]), 3),
            "winner_model_rank": int((race["driver"] == winner["driver"]).values.argmax()) + 1,
        })
    r = pd.DataFrame(races)
    summary = {
        "races": len(r),
        "top_pick_correct": int((r["model_pick"] == r["winner"]).sum()),
        "winner_in_top3": int((r["winner_model_rank"] <= 3).sum()),
        "pole_sitter_won": int((r["winner_grid"] == 1).sum()),   # the naive baseline
        "avg_p_winner": round(float(r["p_winner"].mean()), 3),
    }
    return r, summary


def main():
    df = features.build()
    cols = model.usable_features(df)
    target_round = int(df["round"].max())
    target = df[df["round"] == target_round].copy()
    history = df[df["round"] < target_round]

    # Before qualifying there's no grid: stand in with the practice pace order.
    provisional = target["grid"].isna().all()
    if provisional:
        target["grid"] = target["pace_gap_pct"].rank(method="first")
        target["quali_gap_pct"] = target["pace_gap_pct"]

    # 1. backtest + calibrate
    bt = walk_forward(history, cols)
    sigma, _ = calibrate_sigma(bt)
    bt_races, bt_summary = backtest_report(bt, sigma)

    # 2. fit on everything, predict the target race
    m = model.fit(history, cols)
    target["pred_pos"] = model.predict(m, target, cols)
    p_win, p_pod = simulate(target["pred_pos"], sigma, dnf_rate(history),
                            p_sc=BAKU_P_SC, sc_mult=BAKU_SC_MULT)
    target["p_win"], target["p_podium"] = p_win, p_pod
    target = target.sort_values("p_win", ascending=False)

    # ---- print
    pd.set_option("display.width", 200)
    print(f"\nFeatures used: {cols}")
    print(f"Coefficients (per +1 std, in finishing positions): {model.coefficients(m, cols)}")
    print(f"Calibrated noise sigma: {sigma} positions\n")
    print("Backtest (walk-forward):")
    print(bt_races.to_string(index=False))
    s = bt_summary
    print(f"\n  top pick correct : {s['top_pick_correct']}/{s['races']}"
          f"   (baseline 'pole sitter wins': {s['pole_sitter_won']}/{s['races']})")
    print(f"  winner in top 3  : {s['winner_in_top3']}/{s['races']}")
    print(f"  avg prob given to actual winner: {s['avg_p_winner']:.1%}\n")
    label = "PROVISIONAL (pre-qualifying, grid = practice pace order)" if provisional else "grid from qualifying"
    print(f"{target['event'].iat[0]}: {label}")
    show = target[["driver", "team", "grid", "pace_gap_pct", "pred_pos", "p_win", "p_podium"]].head(10)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---- export for the website
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "season": 2026,
        "round": target_round,
        "event": target["event"].iat[0],
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "provisional": bool(provisional),
        "sigma": sigma,
        "assumptions": {"p_safety_car": BAKU_P_SC, "sc_noise_mult": BAKU_SC_MULT,
                        "p_dnf": round(dnf_rate(history), 3)},
        "coefficients": {k: float(v) for k, v in model.coefficients(m, cols).items()},
        "drivers": [
            {"driver": r.driver, "name": r.name, "team": r.team, "grid": int(r.grid),
             "pace_gap_pct": round(float(r.pace_gap_pct), 3),
             "pred_pos": round(float(r.pred_pos), 2),
             "p_win": round(float(r.p_win), 4), "p_podium": round(float(r.p_podium), 4)}
            for r in target.itertuples()
        ],
        "backtest": {"summary": bt_summary, "races": bt_races.to_dict(orient="records")},
    }
    OUT.write_text(json.dumps(out, indent=2))
    # Same data as a script, so docs/index.html also works when opened straight from disk.
    OUT.with_suffix(".js").write_text(f"window.PREDICTION = {json.dumps(out)};\n")
    print(f"\nWrote {OUT.relative_to(ROOT)} (+ .js)")


if __name__ == "__main__":
    main()
