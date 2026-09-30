"""Backtest on 2026 so far, calibrate the simulation, and predict the next race.

Walk-forward backtest: to "predict" round N we train only on rounds 1..N-1,
exactly as if we'd run this the night before that race. No peeking.

Live predictions are frozen in docs/data/races/round-N.json. They are rewritten
as the weekend goes on (after FP3, after qualifying), but once the race has a
result the round is never predicted again, so the file is exactly what was
published before lights out. Frozen predictions are then scored against the
real result.

Run:  python -m src.predict     -> prints results, writes docs/data/site.{json,js}
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from . import data, features, model
from .simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
SITE_DATA = ROOT / "docs" / "data"
RACES = SITE_DATA / "races"

MODEL_VERSION = "v1"

FIRST_BACKTEST_ROUND = 5          # need a few races of history before predicting
SIGMAS = np.arange(0.5, 6.01, 0.25)

# The calibrated noise already covers an average 2026 race. Street circuits get
# extra chaos on top: a likely safety car that shuffles the order. These are
# assumptions, not learned (2026 has only one race per track).
STREET_CIRCUITS = {"Azerbaijan Grand Prix", "Singapore Grand Prix", "Las Vegas Grand Prix", "Monaco Grand Prix"}
STREET_P_SC = 0.6
STREET_SC_MULT = 1.3


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


def predict_race(target, history, cols, sigma):
    """Win / podium probabilities for an upcoming race, as a JSON-ready dict."""
    target = target.copy()
    # Before qualifying there's no grid: stand in with the practice pace order.
    provisional = bool(target["grid"].isna().all())
    if provisional:
        target["grid"] = target["pace_gap_pct"].rank(method="first")
        target["quali_gap_pct"] = target["pace_gap_pct"]

    event = target["event"].iat[0]
    street = event in STREET_CIRCUITS
    p_sc, sc_mult = (STREET_P_SC, STREET_SC_MULT) if street else (0.0, 1.0)

    m = model.fit(history, cols)
    target["pred_pos"] = model.predict(m, target, cols)
    target["p_win"], target["p_podium"] = simulate(target["pred_pos"], sigma, dnf_rate(history),
                                                   p_sc=p_sc, sc_mult=sc_mult)
    target = target.sort_values("p_win", ascending=False)
    return {
        "season": data.SEASON,
        "round": int(target["round"].iat[0]),
        "event": event,
        "model_version": MODEL_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "provisional": provisional,
        "sigma": sigma,
        "assumptions": {"p_safety_car": p_sc, "sc_noise_mult": sc_mult,
                        "p_dnf": round(dnf_rate(history), 3)},
        "coefficients": {k: float(v) for k, v in model.coefficients(m, cols).items()},
        "drivers": [
            {"driver": r.driver, "name": r.name, "team": r.team, "grid": int(r.grid),
             "pace_gap_pct": round(float(r.pace_gap_pct), 3),
             "pred_pos": round(float(r.pred_pos), 2),
             "p_win": round(float(r.p_win), 4), "p_podium": round(float(r.p_podium), 4)}
            for r in target.itertuples()
        ],
    }


def score_live(df):
    """Attach the real result to every frozen prediction whose race has happened."""
    scored = []
    for path in sorted(RACES.glob("round-*.json"), key=lambda p: int(p.stem.split("-")[1])):
        pred = json.loads(path.read_text())
        res = df[(df["round"] == pred["round"]) & df["finish_pos"].notna()].set_index("driver")
        if res.empty:
            continue
        for d in pred["drivers"]:
            if d["driver"] in res.index:
                r = res.loc[d["driver"]]
                d["finish_pos"] = int(r["finish_pos"])
                d["race_grid"] = int(r["grid"])
                d["dnf"] = bool(r["dnf"])
        ranked = pred["drivers"]  # already sorted by p_win
        winner = next(d for d in ranked if d.get("finish_pos") == 1)
        podium = [d for d in ranked if d.get("finish_pos", 99) <= 3]
        pred["result"] = {
            "winner": winner["driver"],
            "model_pick": ranked[0]["driver"],
            "pick_won": ranked[0]["driver"] == winner["driver"],
            "p_winner": winner["p_win"],
            "winner_model_rank": ranked.index(winner) + 1,
            "podium": [d["driver"] for d in sorted(podium, key=lambda d: d["finish_pos"])],
            "podium_hits": sum(d["driver"] in {x["driver"] for x in ranked[:3]} for d in podium),
        }
        scored.append(pred)
    return scored


def next_event(after_round):
    sched = fastf1.get_event_schedule(data.SEASON, include_testing=False)
    nxt = sched[sched["RoundNumber"] > after_round].head(1)
    if nxt.empty:
        return None
    ev = nxt.iloc[0]
    return {"round": int(ev.RoundNumber), "event": ev.EventName,
            "date": ev.EventDate.strftime("%Y-%m-%d")}


def main():
    df = features.build()
    cols = model.usable_features(df)
    done = df[df.groupby("round")["finish_pos"].transform("count") > 0]
    upcoming = df[~df["round"].isin(done["round"])]

    # 1. backtest + calibrate on every finished race
    bt = walk_forward(done, cols)
    sigma, _ = calibrate_sigma(bt)
    bt_races, bt_summary = backtest_report(bt, sigma)

    pd.set_option("display.width", 200)
    print(f"\nModel {MODEL_VERSION}, features: {cols}")
    print(f"Calibrated noise sigma: {sigma} positions\n")
    print("Backtest (walk-forward):")
    print(bt_races.to_string(index=False))
    s = bt_summary
    print(f"\n  top pick correct : {s['top_pick_correct']}/{s['races']}"
          f"   (baseline 'pole sitter wins': {s['pole_sitter_won']}/{s['races']})")
    print(f"  winner in top 3  : {s['winner_in_top3']}/{s['races']}")
    print(f"  avg prob given to actual winner: {s['avg_p_winner']:.1%}\n")

    # 2. predict the upcoming race, if its weekend has started
    prediction = None
    if not upcoming.empty:
        rnd = int(upcoming["round"].min())
        prediction = predict_race(upcoming[upcoming["round"] == rnd], done, cols, sigma)
        RACES.mkdir(parents=True, exist_ok=True)
        (RACES / f"round-{rnd}.json").write_text(json.dumps(prediction, indent=2))
        label = "PROVISIONAL (pre-qualifying)" if prediction["provisional"] else "grid from qualifying"
        print(f"{prediction['event']}: {label}")
        print(pd.DataFrame(prediction["drivers"])[["driver", "team", "grid", "pred_pos", "p_win", "p_podium"]]
              .head(10).to_string(index=False))
    else:
        print("No upcoming race data yet (next weekend hasn't started).")

    # 3. score frozen predictions against real results
    live = score_live(df)
    for p in live:
        r = p["result"]
        print(f"Live R{p['round']} {p['event']}: pick {r['model_pick']} ({'won' if r['pick_won'] else 'lost'}), "
              f"winner {r['winner']} given {r['p_winner']:.1%}, podium {r['podium_hits']}/3 in model top 3")

    # ---- export for the website
    site = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_version": MODEL_VERSION,
        "next_event": next_event(int(done["round"].max())),
        "prediction": prediction,
        "live": live,
        "backtest": {"summary": bt_summary, "races": bt_races.to_dict(orient="records"), "sigma": sigma},
    }
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "site.json").write_text(json.dumps(site, indent=2))
    # Same data as a script, so docs/index.html also works when opened straight from disk.
    (SITE_DATA / "site.js").write_text(f"window.SITE = {json.dumps(site)};\n")
    print(f"\nWrote docs/data/site.json (+ .js)")


if __name__ == "__main__":
    main()
