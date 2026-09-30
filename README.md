# F1 Race Predictor

Win and podium probabilities for the next Formula 1 race, from a model backtested on every 2026 race so far and 20,000 Monte Carlo race simulations.

**Live site:** https://swishnu.github.io/f1-race-predictor/

## Live record

Every prediction is published before lights out, frozen in [`docs/data/races/`](docs/data/races/), and scored after the race.

| Round | Race | Model's pick | Winner | Winner's odds | Podium in model's top 3 |
|---|---|---|---|---|---|
| 15 | Azerbaijan GP | Russell (42%) ✅ | Russell | 42% | 1 / 3 |

**Baku:** Russell won from pole as predicted. Verstappen (P8 → P2) and Antonelli (P16 → P5) came through the field. That's the blind spot flagged before the race: the model doesn't expect fast cars starting out of position to move forward.

## Backtest

Walk-forward over rounds 5–15: each race is predicted using only the races before it.

| | Model | Baseline: "pole sitter wins" |
|---|---|---|
| Top pick won | **8 / 11** | 6 / 11 |
| Winner in model's top 3 | **10 / 11** | – |
| Avg. probability given to the actual winner | 34% | – |

## How it works

1. **Data** ([src/data.py](src/data.py)): pulls every 2026 session with [FastF1](https://docs.fastf1.dev/): qualifying gap to pole, grid slot, race result and **long-run practice pace**. A stint counts as a long run if 4+ laps (at least 75% of the stint) sit within 4% of its best lap. That rejects push/cool-down qualifying runs. Safety-car and red-flag laps are removed.
2. **Features** ([src/features.py](src/features.py)): a race-N feature only uses results from races before N (`groupby` + `shift(1)`), so nothing leaks from the future.
3. **Model** ([src/model.py](src/model.py)): ridge regression predicting **finishing position** rather than "won". 2026 has 14 winners but ~300 finishing positions to learn from.
4. **Simulation** ([src/simulate.py](src/simulate.py)): each of 20,000 races adds random noise to the predicted positions, applies random DNFs (2026 average rate), and on street circuits a likely safety car (60%) that adds extra shuffle. Win % = share of simulations each driver wins.
5. **Calibration** ([src/predict.py](src/predict.py)): the noise level is chosen by minimising podium log-loss on the backtest.

## Things I learned building it

- **My first calibration was overconfident.** Calibrating the noise on winners alone kept favouring less noise, all the way down to zero. 2026 has been unusually predictable (13 of 14 winners started P1 or P2), and with one outcome per race the calibration overfit to that. Calibrating on podiums (3× the evidence, and misses are penalised too) gives a clear optimum.
- **More features didn't help.** Recent form (average finish over the last 3 races) made predictions worse, and team strength added nothing beyond championship points. Both are still in `features.py` but excluded from the model.
- **Practice pace is noisy.** Teams run different fuel loads, and Baku FP2 was full of yellow flags. The first version of the long-run filter picked the wrong drivers as fastest.
- **Known blind spot:** the model is linear in grid position, so it doesn't understand that a fast car starting at the back will overtake. It gave Antonelli 0.1% at Monza from P19, and Antonelli won. At Baku it gave Antonelli (P16, #2 long-run pace) a 3% podium chance, and Antonelli finished 5th.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.data       # download / refresh data  -> data/weekends.csv
.venv/bin/python -m src.predict    # backtest, prediction, scoring -> docs/data/site.json
python3 -m http.server -d docs     # view the site at http://localhost:8000
```

Re-run both after each session (FP3, qualifying) to update the prediction, and after the race to score it. Once a race has a result, its prediction in `docs/data/races/round-N.json` is never overwritten.

## Ideas for next

- Model grid position non-linearly (e.g. gradient boosting or an overtaking term per track), so fast cars out of position aren't written off
- Track-specific safety-car rates from historical data instead of an assumption
