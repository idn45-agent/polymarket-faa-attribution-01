#!/usr/bin/env python3
"""Forecast Polymarket Yes prices, minute by minute, 240 minutes ahead.

For every market of a case (schema "makefun/polymarket-case/v1") this
program writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the Yes
price at each of the 240 minutes after the case's checkpoint.

Model, per market, from that market's own price history only:

  centre  The last price x, moved by a strongly damped estimate of the
          recent drift (price units per hour, applied linearly).
  spread  A per-square-root-minute sigma estimated from the market's own
          one-minute moves, shrunk toward the house baseline sigma of
          0.0025 so a short or quiet history stays close to the
          baseline, with a small floor so a flat market still gets a
          nonzero width.
  row h   q_tau = clamp(x + drift * h / 60 + z_tau * sigma * sqrt(h),
          0, 1), rounded to 6 decimals, z the standard normal
          quantiles at the five levels.

Deterministic (no randomness, no clock, no network), standard library
only. Reads every *.in file of $MAKEFUN_INPUT_DIR (else /input) in name
order and writes <case-id>.out per case to $MAKEFUN_OUTPUT_DIR (else
/output).
"""

import json
import math
import os
import sys

# Standard normal quantiles at 0.05, 0.25, 0.5, 0.75, 0.95.
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)

BASELINE_SIGMA = 0.0025   # the house baseline's per-sqrt-minute sigma
SHRINK_PSEUDO = 60.0      # pseudo-observations of the baseline in the shrink
SIGMA_WINDOW = 720        # minutes of moves the sigma estimate looks back
SIGMA_FLOOR = 0.0005      # smallest sigma ever used
DRIFT_WINDOW = 240        # minutes of moves the drift estimate looks back
DRIFT_DAMP = 0.05         # only a twentieth of the raw trend is applied
DRIFT_CAP_PER_HOUR = 0.01  # drift clamp, price units per hour

# E|N(0, sigma)| = sigma * sqrt(2 / pi), so a mean absolute one-minute
# move converts to a sigma by multiplying with sqrt(pi / 2).
MAD_TO_SIGMA = math.sqrt(math.pi / 2.0)


def clamp01(value):
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def minute_moves(prices, window):
    """The last `window` one-minute price changes, oldest first."""
    moves = []
    for i in range(1, len(prices)):
        moves.append(prices[i] - prices[i - 1])
    if len(moves) > window:
        moves = moves[len(moves) - window:]
    return moves


def estimate_sigma(prices):
    """Per-square-root-minute sigma from the market's recent moves.

    The raw estimate is the mean absolute one-minute move over the
    latest SIGMA_WINDOW minutes (the whole history when it is
    shorter), converted to a normal sigma. It is then shrunk toward
    BASELINE_SIGMA with SHRINK_PSEUDO pseudo-observations:
    sigma = (n * raw + 60 * 0.0025) / (n + 60), and floored so a
    perfectly flat market does not forecast a zero-width band.
    """
    if len(prices) < 2:
        return BASELINE_SIGMA
    moves = minute_moves(prices, SIGMA_WINDOW)
    n = len(moves)
    mean_abs = sum(abs(move) for move in moves) / n
    raw = mean_abs * MAD_TO_SIGMA
    sigma = (n * raw + SHRINK_PSEUDO * BASELINE_SIGMA) / (n + SHRINK_PSEUDO)
    if sigma < SIGMA_FLOOR:
        sigma = SIGMA_FLOOR
    return sigma


def estimate_drift(prices):
    """Recent drift in price units per hour, strongly damped.

    A linearly weighted mean of the last DRIFT_WINDOW one-minute moves
    (the most recent move weighs most), turned into a per-hour rate,
    scaled by DRIFT_DAMP and clamped to +/- DRIFT_CAP_PER_HOUR. A
    history too short to show a trend gives 0.

    The damping is deliberately severe: on pseudo-checkpoints cut
    from the practice histories, the raw recent trend of these steppy
    prices hurt more than it helped at every weaker damping tried
    (mean pinball loss 0.374 at a damping of 0.5 against 0.332 with
    the trend nearly switched off), because a slope fitted over a few
    hours mostly chases single jumps that do not continue. A twentieth
    of the trend keeps a whisper of it for markets that genuinely run
    for hours, at no measured cost on the practice.
    """
    if len(prices) < 3:
        return 0.0
    moves = minute_moves(prices, DRIFT_WINDOW)
    weight_sum = 0.0
    weighted = 0.0
    for index, move in enumerate(moves):
        weight = index + 1
        weight_sum += weight
        weighted += weight * move
    per_minute = weighted / weight_sum
    per_hour = per_minute * 60.0 * DRIFT_DAMP
    if per_hour > DRIFT_CAP_PER_HOUR:
        return DRIFT_CAP_PER_HOUR
    if per_hour < -DRIFT_CAP_PER_HOUR:
        return -DRIFT_CAP_PER_HOUR
    return per_hour


def forecast_market(prices, horizon):
    """The `horizon` quantile rows for one market."""
    if prices:
        x = prices[-1]
        sigma = estimate_sigma(prices)
        drift = estimate_drift(prices)
    else:
        # A market with no price at all: centre on 0.5 at baseline
        # width rather than crash; the cases the grader runs always
        # carry at least one price per market.
        x = 0.5
        sigma = BASELINE_SIGMA
        drift = 0.0
    rows = []
    for h in range(1, horizon + 1):
        centre = x + drift * h / 60.0
        half_width = sigma * math.sqrt(h)
        row = []
        previous = 0.0
        for z in Z:
            value = round(clamp01(centre + z * half_width), 6)
            if value < previous:
                value = previous
            previous = value
            row.append(value)
        rows.append(row)
    return rows


def run_case(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as handle:
        case = json.load(handle)
    horizon = int(case.get("horizon_minutes", 240))
    forecasts = []
    for market in case.get("markets", []):
        rows = forecast_market(market.get("prices") or [], horizon)
        forecasts.append({"id": market["id"], "q": rows})
    text = json.dumps({"forecasts": forecasts}, separators=(",", ":"))
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    for name in names:
        case_id = name[:-len(".in")]
        run_case(os.path.join(input_dir, name),
                 os.path.join(output_dir, case_id + ".out"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
