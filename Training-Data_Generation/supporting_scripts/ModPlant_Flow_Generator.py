#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Union, Callable

import json

from supporting_scripts.ModPlant_Render_Tools import print_flowchart, print_schedule_table


# --------------------------- Configuration ---------------------------

@dataclass(frozen=True)
class Config:
    dose_defaults: Dict[str, float]
    sep_defaults: Dict[str, float]
    usage_s: int = 3600
    settling_s: int = 300


DEFAULT_CFG = Config(
    dose_defaults={"A": 0.30, "B": 0.20, "C": 0.15},
    sep_defaults={"A": 0.25, "B": 0.20, "C": 0.25},
    usage_s=3600,
    settling_s=300,
)

# --------------------------- Small utilities ---------------------------

def parse_order_strict_list(order: List[Union[str, dict]]) -> List[dict]:
    """
    Accepts:
      - Ingredient string: "A" | "B" | "C" | "D" | ... (auto uppercased)
      - Mix dict: {"mix": {"rpm": int, "duration": int}}
    Returns a normalized op list with items like:
      {"type":"dose","ingr":"A"} or {"type":"mix","rpm":150,"duration":10}
    """
    ops: List[dict] = []
    for item in order:
        if isinstance(item, str):
            token = item.strip().upper()
            if not token:
                raise ValueError("Empty ingredient token in 'order'.")
            ops.append({"type": "dose", "ingr": token})
        elif isinstance(item, dict):
            m = item.get("mix")
            if not isinstance(m, dict):
                raise ValueError("Mix must be exactly {'mix': {'rpm': <int>, 'duration': <int_seconds>}}.")
            rpm = m.get("rpm")
            dur = m.get("duration")
            if not isinstance(rpm, int) or not isinstance(dur, int):
                raise ValueError("Mix.rpm and Mix.duration must be integers.")
            ops.append({"type": "mix", "rpm": rpm, "duration": dur})
        else:
            raise ValueError(f"Unsupported order item: {item!r}")
    return ops


def get_usage_settling(order_json: Dict, cfg: Config) -> tuple[int, int]:
    """
    Read per-order Usage and Settling durations.
    Accepts "usage_and_settling": [usage_seconds, settling_seconds]; falls back to cfg.
    """
    uas = order_json.get("usage_and_settling")
    if uas is None:
        return cfg.usage_s, cfg.settling_s
    if (not isinstance(uas, list)) or len(uas) != 2:
        raise ValueError("'usage_and_settling' must be a 2-element list: [usage_seconds, settling_seconds].")
    usage, settling = uas
    if not (isinstance(usage, int) and isinstance(settling, int)):
        raise ValueError("'usage_and_settling' values must be integers.")
    if usage < 0 or settling < 0:
        raise ValueError("'usage_and_settling' values must be non-negative.")
    return usage, settling


def get_rate_getters(order_json: Dict, cfg: Config) -> tuple[Callable[[str], float], Callable[[str], float]]:
    """
    Return two closures: (dose_rate, sep_rate).
    They resolve an ingredient's rate by preferring user overrides, otherwise config defaults.
    """
    user_rates = order_json.get("rates") or {}
    user_dose = user_rates.get("dose") or {}
    user_sep = user_rates.get("sep") or {}

    def dose_rate(ingr: str) -> float:
        if ingr in user_dose:
            val = user_dose[ingr]
        else:
            val = cfg.dose_defaults.get(ingr)
        if not isinstance(val, (int, float)):
            raise ValueError(f"No dosing rate available for ingredient {ingr!r}. Provide it under 'rates.dose' or extend defaults.")
        return float(val)

    def sep_rate(ingr: str) -> float:
        if ingr in user_sep:
            val = user_sep[ingr]
        else:
            val = cfg.sep_defaults.get(ingr)
        if not isinstance(val, (int, float)):
            raise ValueError(f"No separation rate available for ingredient {ingr!r}. Provide it under 'rates.sep' or extend defaults.")
        return float(val)

    return dose_rate, sep_rate


# --------------------------- Dose plan (count + validate + allocate) ---------------------------

def make_dose_plan(order_list: List[Union[str, dict]], ratio: Dict[str, List[Union[int, float]]], total_volume: float) -> List[dict]:
    """
    Build the per-occurrence dosing plan (ONE pass API):
      - Extract dose sequence from order_list (skip 'mix' steps).
      - For each present ingredient X, ratio[X] must be a list whose length equals the
        number of times X appears in the order.
      - Collect weights in dose order, normalize globally, multiply by total_volume.
    Returns a list aligned with the dose occurrences order. Each item:
      {
        "ingr": "A",
        "portion_L": 0.8,
        "occurrence": 1,
        "occurrences": 3,
        "total_L": 3.0
      }
    """
    dose_seq = [x for x in order_list if isinstance(x, str)]
    if not dose_seq:
        raise ValueError("Order contains no dosing steps; nothing to produce.")

    # Count occurrences
    counts: Dict[str, int] = {}
    for g in dose_seq:
        counts[g] = counts.get(g, 0) + 1

    # Validate ratio lists and gather weights per ingredient
    weights_per_ingr: Dict[str, List[float]] = {}
    for ingr, cnt in counts.items():
        if ingr not in ratio or not isinstance(ratio[ingr], list):
            raise ValueError(f"ratio[{ingr}] must be a list with {cnt} weights (missing or non-list).")
        lst = ratio[ingr]
        if len(lst) != cnt:
            raise ValueError(f"ratio[{ingr}] length mismatch: expected {cnt}, got {len(lst)}.")
        try:
            weights_per_ingr[ingr] = [float(v) for v in lst]
        except Exception:
            raise ValueError(f"ratio[{ingr}] must contain numeric weights.")

    # Stitch global weights following dose order
    idx = {k: 0 for k in counts}
    weights: List[float] = []
    for g in dose_seq:
        i = idx[g]
        weights.append(weights_per_ingr[g][i])
        idx[g] = i + 1

    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("Sum of per-occurrence weights must be positive.")
    portions = [total_volume * (w / total_w) for w in weights]

    # Build totals and plan entries
    totals: Dict[str, float] = {k: 0.0 for k in counts}
    for g, p in zip(dose_seq, portions):
        totals[g] += p

    seen = {k: 0 for k in counts}
    plan: List[dict] = []
    for g, p in zip(dose_seq, portions):
        seen[g] += 1
        plan.append({
            "ingr": g,
            "portion_L": p,
            "occurrence": seen[g],
            "occurrences": counts[g],
            "total_L": totals[g],
        })
    return plan

# --------------------------- Core builder & API ---------------------------

def _resolve_collecting_config(
    order: Dict,
    totals: Dict[str, float],
    default_total_volume: float,
) -> dict:
    """
    Decide Collecting parameters:
      - volume_L: use order["collecting"]["volume"] if present; else default_total_volume.
      - ratio:
          * If order["collecting"]["ratio"] is given -> use those scalars as-is.
          * Else -> **sum** the per-occurrence weights from order["ratio"] for each ingredient.
    Returns: {"volume_L": float, "ratio": {ingr: raw_sum}}
    """
    # Volume
    col_cfg = order.get("collecting") or {}
    if "volume" in col_cfg:
        try:
            volume_L = float(col_cfg["volume"])
        except Exception:
            raise ValueError("collecting.volume must be a number (litres).")
        if volume_L < 0:
            raise ValueError("collecting.volume must be non-negative.")
    else:
        volume_L = float(default_total_volume)

    # Ratio: raw sums (no normalization)
    if "ratio" in col_cfg and isinstance(col_cfg["ratio"], dict):
        # Use user-provided numbers as-is
        raw = {}
        for ingr, w in col_cfg["ratio"].items():
            try:
                raw[ingr.upper()] = float(w)
            except Exception:
                raise ValueError("collecting.ratio weights must be numeric.")
        # Check ingredient coverage matches produced ingredients
        if set(raw.keys()) != set(totals.keys()):
            missing = [k for k in totals.keys() if k not in raw]
            extra = [k for k in raw.keys() if k not in totals]
            if missing:
                raise ValueError(f"collecting.ratio missing ingredients: {missing}")
            if extra:
                raise ValueError(f"collecting.ratio contains unknown ingredients: {extra}")
        ratio_raw = {k: raw[k] for k in totals.keys()}  # keep stable order
    else:
        # Derive from order["ratio"] by summing per-occurrence weights for each ingredient
        order_ratio = order.get("ratio", {})
        ratio_raw: Dict[str, float] = {}
        for ingr in totals.keys():
            lst = order_ratio.get(ingr, [])
            if not isinstance(lst, list) or len(lst) == 0:
                raise ValueError(f"ratio[{ingr}] must be a non-empty list to derive collecting ratios.")
            s = 0.0
            for v in lst:
                s += float(v)
            ratio_raw[ingr] = s

    return {"volume_L": volume_L, "ratio": ratio_raw}


def build_steps(order: Dict, cfg: Config = DEFAULT_CFG) -> List[dict]:
    """
    Build internal step list (no 'start_s'). Steps contain:
      - type: "dose" | "mix" | "usage" | "collecting" | "settling" | "sep"
      - stage: human-readable title
      - duration_s: int
      - params: step-specific
    """
    if not isinstance(order.get("order"), list):
        raise TypeError("'order' must be a list; MIX must be {'mix': {'rpm':..., 'duration':...}}.")
    ops = parse_order_strict_list(order["order"])

    # Dose plan (per-occurrence lists only)
    plan = make_dose_plan(order["order"], order.get("ratio", {}), float(order["volume"]))

    # Rate getters
    dose_rate, sep_rate = get_rate_getters(order, cfg)

    # Build steps from ops (mix interleaves with dose occurrences)
    steps: List[dict] = []
    di = 0
    for op in ops:
        if op["type"] == "dose":
            dp = plan[di]; di += 1
            ingr = dp["ingr"]
            portion = dp["portion_L"]
            rate = dose_rate(ingr)
            dur_i = int(round(portion / rate))
            steps.append({
                "type": "dose",
                "stage": f"Dosing {ingr}",
                "duration_s": dur_i,
                "params": {
                    "ingredient": ingr,
                    "portion_L": portion,
                    "total_L": dp["total_L"],
                    "occurrence": dp["occurrence"],
                    "occurrences": dp["occurrences"],
                    "rate_Lps": rate,
                },
            })
        else:  # mix
            rpm, dur = op["rpm"], int(op["duration"])
            steps.append({
                "type": "mix",
                "stage": f"Mix {rpm} rpm",
                "duration_s": dur,
                "params": {"rpm": rpm},
            })

    # Usage
    usage_s, settling_s = get_usage_settling(order, cfg)
    steps.append({"type": "usage", "stage": "Usage", "duration_s": usage_s, "params": {}})

    # Totals (needed for Collecting + Separation)
    totals: Dict[str, float] = {}
    for d in plan:
        totals[d["ingr"]] = totals.get(d["ingr"], 0.0) + d["portion_L"]

    # # Collecting (0s duration, carries volume + raw ratio sums)
    # collecting_cfg = _resolve_collecting_config(order, totals, float(order["volume"]))
    # steps.append({
    #     "type": "collecting",
    #     "stage": "Collecting",
    #     "duration_s": 0,  # change if a time model is desired
    #     "params": {"volume_L": collecting_cfg["volume_L"], "ratio": collecting_cfg["ratio"]},
    # })

    # Settling
    steps.append({"type": "settling", "stage": "Settling", "duration_s": settling_s, "params": {}})

    # Separation order
    dose_seq = [d["ingr"] for d in plan]
    if "separation_order" in order and isinstance(order["separation_order"], list):
        sep_order_raw = [str(x).upper() for x in order["separation_order"]]
        sep_order = [x for x in sep_order_raw if x in totals]
    else:
        sep_order, seen = [], set()
        for x in reversed(dose_seq):
            if x not in seen:
                seen.add(x)
                sep_order.append(x)

    for ingr in sep_order:
        vol = totals[ingr]
        rate = sep_rate(ingr)
        dur_i = int(round(vol / rate))
        steps.append({
            "type": "sep",
            "stage": f"Separation {ingr}",
            "duration_s": dur_i,
            "params": {"ingredient": ingr, "volume_L": vol, "rate_Lps": rate},
        })

    return steps


def build_schedule(order: Dict, *, render: bool = False, include_params: bool = True, cfg: Config = DEFAULT_CFG) -> List[dict]:
    """
    Build the final schedule.
    Returns entries WITHOUT 'end_s' to keep payload small:
      {"stage": ..., "type": ..., "start_s": int, "duration_s": int, "params": {...}}
    If render=True, prints ASCII flowchart and table before returning.
    """
    steps = build_steps(order, cfg=cfg)

    # Accumulate time -> add 'start_s'
    t = 0
    schedule: List[dict] = []
    for s in steps:
        d = int(s["duration_s"])
        schedule.append({
            "type": s["type"],
            "stage": s["stage"],
            "start_s": t,
            "duration_s": d,
            "params": s.get("params", {}),
        })
        t += d

    # if render:
    #     print("\n=== ASCII FLOWCHART ===\n")
    #     print_flowchart(schedule)
    
        #print_schedule_table(schedule, include_params=include_params)

    return schedule
