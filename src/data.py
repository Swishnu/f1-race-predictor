"""Pull every 2026 race weekend from FastF1 into one tidy table.

One row per (round, driver) with:
    round, event, driver, name, team,
    quali_pos, quali_gap_pct   - qualifying position and % gap to pole
    grid                       - actual starting slot (after penalties); for the
                                 upcoming race this is quali_pos until the grid is final
    pace_gap_pct               - long-run practice pace, % slower than the best driver
    finish_pos, points, dnf    - race outcome (NaN for the upcoming race)

Run:  python -m src.data            -> writes data/weekends.csv
"""
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

SEASON = 2026
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = ROOT / "data" / "weekends.csv"

fastf1.set_log_level("ERROR")
CACHE.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE))

MIN_STINT_LAPS = 4     # a "long run" needs at least this many consistent laps
RUN_WINDOW = 1.04      # laps within 4% of the stint's best lap count as "race pace"
MIN_CONSISTENT = 0.75  # ...and they must be 75%+ of the stint (rejects push/cool-down quali runs)
NEUTRALISED = "4567"   # TrackStatus codes: 4 SC, 5 red flag, 6/7 VSC. Yellows (2) are fine.


def _load(round_no, identifier, laps=True):
    """Load a session, or return None if it doesn't exist / has no data yet."""
    try:
        s = fastf1.get_session(SEASON, round_no, identifier)
        s.load(laps=laps, telemetry=False, weather=False, messages=False)
        if s.results is None or s.results.empty:
            return None
        return s
    except Exception:
        return None


def long_run_pace(sessions):
    """Median lap time of each driver's best long run across the practice sessions.

    A stint counts as a long run if most of its laps sit within RUN_WINDOW of
    its best lap (a quali run alternates push / cool-down laps, so it fails).
    Each driver's fastest long-run median is returned as % gap to the best driver.
    """
    runs = []
    for s in sessions:
        laps = s.laps
        status = laps["TrackStatus"].astype(str)
        laps = laps[
            laps["LapTime"].notna()
            & laps["PitInTime"].isna()
            & laps["PitOutTime"].isna()
            & ~status.apply(lambda st: any(c in st for c in NEUTRALISED))
            & (laps["Deleted"] != True)  # noqa: E712 (column can hold NaN)
        ].copy()
        laps["t"] = laps["LapTime"].dt.total_seconds()
        for (drv, _stint), g in laps.groupby(["Driver", "Stint"]):
            pace = g[g["t"] < g["t"].min() * RUN_WINDOW]
            if len(pace) >= MIN_STINT_LAPS and len(pace) / len(g) >= MIN_CONSISTENT:
                runs.append((drv, pace["t"].median()))
    if not runs:
        return pd.Series(dtype=float)
    best = pd.DataFrame(runs, columns=["driver", "t"]).groupby("driver")["t"].min()
    return (best / best.min() - 1) * 100


def quali_table(q):
    r = q.results.copy()
    best = r[["Q1", "Q2", "Q3"]].min(axis=1).dt.total_seconds()
    return pd.DataFrame({
        "driver": r["Abbreviation"].values,
        "name": r["FullName"].values,
        "team": r["TeamName"].values,
        "quali_pos": r["Position"].values,
        "quali_gap_pct": ((best / best.min() - 1) * 100).values,
    })


def weekend(round_no, event_name):
    practice = [s for s in (_load(round_no, p) for p in ("FP1", "FP2", "FP3")) if s]
    q = _load(round_no, "Q", laps=False)
    r = _load(round_no, "R", laps=False)
    if q is None and r is None and not practice:
        return None

    if q is not None:
        df = quali_table(q)
    else:  # quali not run yet: start from the practice entry list
        res = practice[-1].results
        df = pd.DataFrame({"driver": res["Abbreviation"].values, "name": res["FullName"].values,
                           "team": res["TeamName"].values,
                           "quali_pos": np.nan, "quali_gap_pct": np.nan})

    df["pace_gap_pct"] = df["driver"].map(long_run_pace(practice))

    if r is not None:
        res = r.results.set_index("Abbreviation")
        grid = res["GridPosition"].replace(0, len(res))  # 0 = pit-lane start
        df["grid"] = df["driver"].map(grid)
        df["finish_pos"] = df["driver"].map(res["Position"])
        df["points"] = df["driver"].map(res["Points"])
        status = res["Status"].fillna("")
        df["dnf"] = df["driver"].map(~status.isin(["Finished", "Lapped"]) & ~status.str.startswith("+"))
    else:
        df["grid"] = df["quali_pos"]
        df["finish_pos"] = np.nan
        df["points"] = np.nan
        df["dnf"] = np.nan

    df.insert(0, "event", event_name)
    df.insert(0, "round", round_no)
    return df


def build(up_to_round=None):
    sched = fastf1.get_event_schedule(SEASON, include_testing=False)
    if up_to_round is not None:
        sched = sched[sched["RoundNumber"] <= up_to_round]
    frames = []
    for _, ev in sched.iterrows():
        print(f"Round {ev.RoundNumber:>2}  {ev.EventName} ...", flush=True)
        df = weekend(int(ev.RoundNumber), ev.EventName)
        if df is None:
            print("    no data yet, stopping")
            break
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    build()
