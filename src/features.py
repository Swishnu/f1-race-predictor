"""Turn data/weekends.csv into model features.

No leakage: a feature for round N only uses race results from rounds < N,
plus this weekend's practice and qualifying. Each row sees only what came
before it via groupby + shift(1).

Run:  python -m src.features      -> prints a sanity check
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "weekends.csv"

FEATURES = [
    "grid",
    "quali_gap_pct",
    "pace_gap_pct",
    "points_before",
    "out_of_position",   # v2
]

# Built and backtested but not used directly: recent_finish made podium
# log-loss worse (1.031 -> 1.099), and team_strength alone added nothing on
# top of points_before (it feeds out_of_position instead).
EXTRA_FEATURES = ["recent_finish", "team_strength"]

MIDFIELD = 11  # neutral finishing position for a 22-car grid when there's no history


def load():
    df = pd.read_csv(DATA).sort_values(["round", "driver"]).reset_index(drop=True)
    df["won"] = (df["finish_pos"] == 1).astype(int)
    return df


def add_points_before(df):
    """Championship points the driver had before this round."""
    df["points_before"] = (
        df.groupby("driver")["points"]
          .transform(lambda s: s.shift(1).fillna(0).cumsum())
    )
    return df


def add_recent_finish(df, window=3):
    """Driver's average finishing position over their previous `window` races."""
    df["recent_finish"] = (
        df.groupby("driver")["finish_pos"]
          .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
          .fillna(MIDFIELD)
    )
    return df


def add_team_strength(df):
    """Team's average points per car per race so far this season, before this round."""
    per_race = df.groupby(["team", "round"])["points"].mean().reset_index()
    per_race["team_strength"] = (
        per_race.groupby("team")["points"]
                .transform(lambda s: s.shift(1).expanding().mean())
                .fillna(0)
    )
    return df.merge(per_race[["team", "round", "team_strength"]], on=["team", "round"], how="left")


def add_out_of_position(df):
    """How many places behind its car's usual slot a driver starts (0 if not behind).

    Teams are ranked by team_strength; the best team's cars "should" fill
    P1-P2, the next team P3-P4, and so on. A Mercedes starting P16 scores ~14.
    Backtested against "fast in practice but starting back": practice pace is
    too noisy in the midfield, while strong cars out of position really do come
    through (podium log-loss 1.048 -> 1.030 over 11 races).
    """
    team_rank = df.groupby("round")["team_strength"].rank(ascending=False, method="dense")
    df["out_of_position"] = (df["grid"] - (2 * team_rank - 0.5)).clip(lower=0)
    return df


def fill_missing(df):
    """Drivers with no usable long run (crashes, sprint weekends with only FP1)
    fall back to their qualifying gap."""
    df["pace_gap_pct"] = df["pace_gap_pct"].fillna(df["quali_gap_pct"])
    df["pace_gap_pct"] = df["pace_gap_pct"].fillna(df["pace_gap_pct"].median())
    df["quali_gap_pct"] = df["quali_gap_pct"].fillna(df["quali_gap_pct"].median())
    return df


def build():
    df = load()
    df = add_points_before(df)
    df = add_recent_finish(df)
    df = add_team_strength(df)
    df = add_out_of_position(df)
    df = fill_missing(df)
    return df


if __name__ == "__main__":
    df = build()
    pd.set_option("display.width", 200)
    last = df["round"][df["finish_pos"].notna()].max()
    print(f"Round {last} (latest completed race), top 10 by grid:\n")
    print(df[df["round"] == last].sort_values("grid")[["driver", "team", "finish_pos"] + FEATURES].head(10).to_string(index=False))
    missing = df[FEATURES].isna().sum()
    print("\nMissing values per feature:\n" + missing.to_string())
