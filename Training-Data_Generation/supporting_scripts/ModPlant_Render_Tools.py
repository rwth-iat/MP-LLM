#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waben_JSON_Render
-----------------
Rendering helpers for the 4-Waben flow:
- print_flowchart(schedule): ASCII flowchart
- print_schedule_table(schedule, include_params=True): timing table with params

Also supports rendering directly from:
- schedule JSON list
- GeneralRecipe-parsed JSON dict (with ProcessElements / DirectedLinks)
via:
- render_flowchart_from_json(json_input)
- render_schedule_table_from_json(json_input, include_params=True)
"""

from __future__ import annotations
import os
import re
import json
from typing import List, Dict, Any, Union, Optional


# --------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------

def fmt_time(seconds: int) -> str:
    """Format seconds into a compact '1h02m03s' or '10m05s' or '12s' string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fmt_ratio_value(x: float) -> str:
    """Print integers without decimals; otherwise a compact float."""
    try:
        xv = float(x)
    except Exception:
        return str(x)
    return str(int(xv)) if xv.is_integer() else f"{xv:.6g}"


# --------------------------------------------------------------------
# ASCII flowchart (box rendering)
# --------------------------------------------------------------------

def make_box(title: str, lines: List[str]) -> List[str]:
    """Build a single ASCII box with a title + lines."""
    content = [title] + lines
    width = max(len(x) for x in content) + 2
    top = "┌" + "─" * width + "┐"
    bot = "└" + "─" * width + "┘"
    body = ["│ " + x.ljust(width - 2) + " │" for x in content]
    return [top] + body + [bot]


def _render_box_from_entry(entry: Dict) -> List[str]:
    """Create a box on the fly from one schedule entry."""
    t = entry["type"]
    title = entry["stage"]
    p = entry.get("params", {})
    dur = int(entry.get("duration_s", 0))

    if t == "dose":
        lines = [
            f"Portion: {p.get('portion_L', 0.0):.3f} L",
            #f"Rate: {p.get('rate_Lps', 0.0)} L/s",
            #f"Dose {p.get('occurrence', 1)}/{p.get('occurrences', 1)} (total {p.get('total_L', p.get('portion_L', 0.0)):.3f} L)",
            #f"Duration: {fmt_time(dur)}",
        ]
    elif t == "mix":
        lines = [f"RPM: {p.get('rpm', 0)}",
                 f"Duration: {fmt_time(dur)}"
                 ]
    elif t == "usage":
        lines = [f"Duration: {fmt_time(dur)}"]
    elif t == "collecting":
        ratio = p.get("ratio", {})
        ratio_str = ", ".join(f"{k}: {_fmt_ratio_value(v)}" for k, v in ratio.items())
        #lines = [f"Volume: {p.get('volume_L', 0.0):.3f} L", f"Ratio: {ratio_str}"]
        lines = [f"Volume: {p.get('volume_L', 0.0):.3f} L"]
    elif t == "settling":
        lines = [f"Duration: {fmt_time(dur)}"]
    elif t == "sep":
        lines = [
            f"Volume: {p.get('volume_L', 0.0):.3f} L",
            #f"Rate: {p.get('rate_Lps', 0.0)} L/s",
            #f"Duration: {fmt_time(dur)}",
        ]
    else:
        lines = [
            #f"Duration: {fmt_time(dur)}"
        ]

    return make_box(title, lines)


def print_flowchart(schedule: List[Dict]) -> None:
    """Render a vertical ASCII flowchart directly from schedule entries."""
    out: List[str] = []
    for i, e in enumerate(schedule):
        out.extend(_render_box_from_entry(e))
        if i < len(schedule) - 1:
            out.append(" " * 4 + "↓")
    print("\n".join(out))

def generate_ascii_flowchart(schedule: List[Dict]) -> str:
    """
    Generates the vertical ASCII flowchart string from schedule entries.
    Returns the string instead of printing it.
    """
    out: List[str] = []
    for i, e in enumerate(schedule):
        out.extend(_render_box_from_entry(e))
        if i < len(schedule) - 1:
            out.append(" " * 4 + "↓")
    
    return "\n".join(out)


# --------------------------------------------------------------------
# Table rendering
# --------------------------------------------------------------------

def _format_params(entry: Dict) -> str:
    """Compact params string per step type for table printing."""
    t = entry["type"]
    p = entry.get("params", {})
    if t == "dose":
        return (
            f"ingredient={p.get('ingredient','')}, portion_L={p.get('portion_L',0.0):.3f}, "
            f"rate_Lps={p.get('rate_Lps',0.0)}, occ={p.get('occurrence',1)}/{p.get('occurrences',1)}"
        )
    if t == "mix":
        return f"rpm={p.get('rpm',0)}"
    if t == "collecting":
        ratio = p.get("ratio", {})
        ratio_str = ", ".join(f"{k}:{_fmt_ratio_value(v)}" for k, v in ratio.items())
        return f"volume_L={p.get('volume_L',0.0):.3f}, ratio={{{ {ratio_str} }}}"
    if t == "sep":
        return (
            f"ingredient={p.get('ingredient','')}, "
            f"volume_L={p.get('volume_L',0.0):.3f}, rate_Lps={p.get('rate_Lps',0.0)}"
        )
    return ""  # usage/settling have no extra params


def print_schedule_table(schedule: List[Dict], include_params: bool = True) -> None:
    """
    Pretty-print schedule as a table. End(s) is computed on the fly.
    Columns: Stage | Start(s) | End(s) | Duration | (Params)
    """
    cols = ["Stage", "Start(s)", "End(s)", "Duration"]
    if include_params:
        cols.append("Params")

    # Build rows and widths
    rows: List[List[str]] = []
    t = 0
    for e in schedule:
        d = int(e.get("duration_s", 0))
        row = [e.get("stage", ""), str(t), str(t + d), fmt_time(d)]
        if include_params:
            row.append(_format_params(e))
        rows.append(row)
        t += d

    colw = [max(len(c), max(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    sep = "+" + "+".join("-" * (w + 2) for w in colw) + "+"

    print("\nSchedule:")
    print(sep)
    print("| " + " | ".join(cols[i].ljust(colw[i]) for i in range(len(cols))) + " |")
    print(sep)
    for r in rows:
        print("| " + " | ".join(r[i].ljust(colw[i]) for i in range(len(cols))) + " |")
    print(sep)

    total_s = sum(int(x.get("duration_s", 0)) for x in schedule)
    print(f"Total time: {fmt_time(total_s)}")


# --------------------------------------------------------------------
# JSON adapter: accept schedule list OR GeneralRecipe JSON dict
# --------------------------------------------------------------------

Schedule = List[Dict[str, Any]]
JsonInput = Union[str, Schedule, Dict[str, Any]]

def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default

def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _parse_ratio_text(s: str) -> Dict[str, float]:
    """
    Parse 'A: 1, B: 2, C: 3' -> {'A':1.0, 'B':2.0, 'C':3.0}
    Robust to spaces; ignores malformed items.
    """
    out: Dict[str, float] = {}
    if not isinstance(s, str):
        return out
    for part in s.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except Exception:
            # keep as-is if not numeric? we keep numeric only for renderer
            continue
    return out

def _param_by_id_prefix(params: List[Dict[str, Any]], prefix: str) -> Optional[Dict[str, Any]]:
    """Find first param whose ID starts with a given prefix."""
    for p in params or []:
        pid = str(p.get("ID", ""))
        if pid.startswith(prefix):
            return p
    return None

def _digits_at_end(s: str) -> int:
    """Helper to sort IDs with numeric suffix, e.g., Mix...001 < Mix...010."""
    m = re.search(r"(\d+)(?!.*\d)", s or "")
    return int(m.group(1)) if m else 0

def _convert_gr_json_to_schedule(gr: Dict[str, Any]) -> Schedule:
    """
    Convert a GeneralRecipe-like JSON dict (as in your uploaded file) into
    a schedule list for rendering.
    """
    pes: List[Dict[str, Any]] = list(gr.get("ProcessElements") or [])
    links: List[Dict[str, Any]] = list(gr.get("DirectedLinks") or [])

    # Build maps for quick lookup
    id_to_pe: Dict[str, Dict[str, Any]] = {str(pe.get("ID")): pe for pe in pes}
    desc_to_ids: Dict[str, List[str]] = {}
    for pid, pe in id_to_pe.items():
        desc = str(pe.get("Description", ""))
        desc_to_ids.setdefault(desc, []).append(pid)

    def _first_id_by_desc(desc: str) -> Optional[str]:
        ids = desc_to_ids.get(desc)
        return ids[0] if ids else None

    # Identify groups
    dosing_ids = [pid for pid, pe in id_to_pe.items()
                  if str(pe.get("Description","")).startswith("Dosing ")]
    mixing_ids = [pid for pid, pe in id_to_pe.items()
                  if str(pe.get("Description","")) == "Mixing_of_Liquids"]
    usage_id  = _first_id_by_desc("Usage")
    collecting_id = _first_id_by_desc("Collecting")
    settling_id   = _first_id_by_desc("Settling")
    separation_ids_all = [pid for pid, pe in id_to_pe.items()
                          if str(pe.get("Description","")).startswith("Separation of ")]

    # Derive dosing order (A,B,C,...) by ingredient letter
    def _ing_from_dosing_desc(desc: str) -> str:
        # "Dosing A" -> "A"
        parts = desc.split()
        return parts[-1] if parts else ""

    dosing_ids_sorted = sorted(
        dosing_ids,
        key=lambda pid: _ing_from_dosing_desc(str(id_to_pe[pid].get("Description","")))
    )

    # Derive mixing order by numeric suffix of ID
    mixing_ids_sorted = sorted(mixing_ids, key=_digits_at_end)

    # Derive separation order by following DirectedLinks from Settling -> Separation_* chain
    sep_order_ids: List[str] = []
    if settling_id:
        # Map from FromID -> list of ToIDs
        edges: Dict[str, List[str]] = {}
        for e in links:
            edges.setdefault(str(e.get("FromID")), []).append(str(e.get("ToID")))
        # Follow chain
        cur = settling_id
        visited = set()
        while True:
            nxts = [nid for nid in edges.get(cur, []) if nid in separation_ids_all]
            if not nxts:
                break
            # choose first deterministically
            nxt = nxts[0]
            if nxt in visited:
                break
            visited.add(nxt)
            sep_order_ids.append(nxt)
            cur = nxt

    # Build schedule entries
    schedule: Schedule = []
    t = 0  # running start time

    # Dosing steps
    for pid in dosing_ids_sorted:
        pe = id_to_pe[pid]
        desc = str(pe.get("Description",""))            # e.g., "Dosing A"
        ingredient = _ing_from_dosing_desc(desc) or "X"
        params = pe.get("Parameters") or []
        p_amt = _param_by_id_prefix(params, "Dosing_Amount")
        vol_str = (p_amt or {}).get("ValueString")
        vol = _coerce_float(vol_str, 0.0)

        entry = {
            "type": "dose",
            "stage": desc,
            "start_s": t,
            "duration_s": 0,  # GR has no dosing duration; keep 0 for rendering
            "params": {
                "ingredient": ingredient,
                "portion_L": vol,
                "total_L": vol,
                "occurrence": 1,
                "occurrences": 1,
                "rate_Lps": 0.0,  # unknown in GR
            },
        }
        schedule.append(entry)
        t += entry["duration_s"]

    # Mixing steps
    for i, pid in enumerate(mixing_ids_sorted, start=1):
        pe = id_to_pe[pid]
        params = pe.get("Parameters") or []
        p_rpm = _param_by_id_prefix(params, "Revolutions_per_minute")
        p_dur = _param_by_id_prefix(params, "Mixing_Duration")
        rpm = _coerce_int((p_rpm or {}).get("ValueString"), 0)
        dur = _coerce_int((p_dur or {}).get("ValueString"), 0)

        entry = {
            "type": "mix",
            "stage": f"Mix {rpm} rpm",
            "start_s": t,
            "duration_s": dur,
            "params": {"rpm": rpm},
        }
        schedule.append(entry)
        t += entry["duration_s"]

    # Usage
    if usage_id:
        pe = id_to_pe[usage_id]
        params = pe.get("Parameters") or []
        p_dur = _param_by_id_prefix(params, "Usage_Duration")
        dur = _coerce_int((p_dur or {}).get("ValueString"), 0)
        entry = {
            "type": "usage",
            "stage": "Usage",
            "start_s": t,
            "duration_s": dur,
            "params": {},
        }
        schedule.append(entry)
        t += entry["duration_s"]

    # Collecting
    if collecting_id:
        pe = id_to_pe[collecting_id]
        params = pe.get("Parameters") or []
        p_vol = _param_by_id_prefix(params, "Collecting_Volume")
        p_rat = _param_by_id_prefix(params, "Collecting_Ratio")
        vol = _coerce_float((p_vol or {}).get("ValueString"), 0.0)
        ratio_text = (p_rat or {}).get("ValueString") or ""
        ratio = _parse_ratio_text(ratio_text)

        entry = {
            "type": "collecting",
            "stage": "Collecting",
            "start_s": t,
            "duration_s": 0,  # collecting modeled as instantaneous in GR
            "params": {"volume_L": vol, "ratio": ratio},
        }
        schedule.append(entry)
        t += entry["duration_s"]

    # Settling
    if settling_id:
        pe = id_to_pe[settling_id]
        params = pe.get("Parameters") or []
        p_dur = _param_by_id_prefix(params, "Settling_Duration")
        dur = _coerce_int((p_dur or {}).get("ValueString"), 0)
        entry = {
            "type": "settling",
            "stage": "Settling",
            "start_s": t,
            "duration_s": dur,
            "params": {},
        }
        schedule.append(entry)
        t += entry["duration_s"]

    # Separation (ordered)
    for pid in sep_order_ids:
        pe = id_to_pe[pid]
        desc = str(pe.get("Description",""))  # e.g., "Separation of C"
        m = re.search(r"Separation of\s+([A-Za-z])", desc)
        ingredient = m.group(1) if m else "X"
        params = pe.get("Parameters") or []
        p_vol = _param_by_id_prefix(params, "Separation_Volume")
        p_dur = _param_by_id_prefix(params, "Separation_Duration")
        vol = _coerce_float((p_vol or {}).get("ValueString"), 0.0)
        dur = _coerce_int((p_dur or {}).get("ValueString"), 0)

        entry = {
            "type": "sep",
            "stage": f"Separation {ingredient}",
            "start_s": t,
            "duration_s": dur,
            "params": {
                "ingredient": ingredient,
                "volume_L": vol,
                "rate_Lps": 0.0,  # not present in GR; keep 0 for renderer
            },
        }
        schedule.append(entry)
        t += entry["duration_s"]

    return schedule


def _normalize_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make an entry safe for renderer:
      - required keys: type, stage, start_s, duration_s, params
      - coerce numeric fields
      - provide sensible defaults for params per type
    """
    etype = str(e.get("type", "")).strip().lower()
    stage = str(e.get("stage", ""))

    start_s = _coerce_int(e.get("start_s", 0), 0)
    duration_s = _coerce_int(e.get("duration_s", 0), 0)

    params = dict(e.get("params", {}) or {})

    if etype == "dose":
        params.setdefault("ingredient", "X")
        params["portion_L"]   = _coerce_float(params.get("portion_L", 0.0), 0.0)
        params["total_L"]     = _coerce_float(params.get("total_L",   params["portion_L"]), 0.0)
        params["occurrence"]  = _coerce_int(params.get("occurrence", 1), 1)
        params["occurrences"] = _coerce_int(params.get("occurrences", 1), 1)
        params["rate_Lps"]    = _coerce_float(params.get("rate_Lps", 0.0), 0.0)

    elif etype == "mix":
        params["rpm"] = _coerce_int(params.get("rpm", 0), 0)

    elif etype == "collecting":
        params["volume_L"] = _coerce_float(params.get("volume_L", 0.0), 0.0)
        ratio = params.get("ratio") or {}
        params["ratio"] = {str(k): _coerce_float(v, 0.0) for k, v in ratio.items()}

    elif etype == "sep":
        params.setdefault("ingredient", "X")
        params["volume_L"] = _coerce_float(params.get("volume_L", 0.0), 0.0)
        params["rate_Lps"] = _coerce_float(params.get("rate_Lps", 0.0), 0.0)

    # usage/settling have empty params by design
    return {
        "type": etype,
        "stage": stage,
        "start_s": start_s,
        "duration_s": duration_s,
        "params": params,
    }

def _is_general_recipe_json(data: Any) -> bool:
    """Heuristic: a dict with ProcessElements(list) and DirectedLinks(list)."""
    return isinstance(data, dict) and isinstance(data.get("ProcessElements"), list)

def _load_json_input(json_input: JsonInput) -> Schedule:
    """
    Accepts:
      - a Python list (already loaded schedule)
      - a JSON string
      - a file path ending with .json
      - a GeneralRecipe JSON dict (with ProcessElements)
    Returns a normalized schedule list, sorted by start_s if present or built deterministically.
    """
    data: Any

    if isinstance(json_input, list):
        data = json_input
    elif isinstance(json_input, dict):
        # GeneralRecipe dict -> schedule
        return _convert_gr_json_to_schedule(json_input)
    elif isinstance(json_input, str):
        # File path?
        if os.path.isfile(json_input) and json_input.lower().endswith(".json"):
            with open(json_input, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Try to parse as JSON string
            data = json.loads(json_input)
    else:
        raise TypeError("json_input must be a list, a JSON string, a .json file path, or a GeneralRecipe dict.")

    # If it's a GeneralRecipe dict parsed from file/string, convert it
    if _is_general_recipe_json(data):
        schedule = _convert_gr_json_to_schedule(data)
        return schedule

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a schedule list or a GeneralRecipe dict.")

    norm = [_normalize_entry(e) for e in data]
    if all(isinstance(e.get("start_s"), int) for e in norm):
        norm = sorted(norm, key=lambda x: x.get("start_s", 0))
    return norm


# Public APIs (call these)
def render_flowchart_from_json(json_input: JsonInput) -> None:
    """Normalize the provided JSON (schedule list or GeneralRecipe dict) and print an ASCII flowchart."""
    schedule = _load_json_input(json_input)
    print_flowchart(schedule)

def render_schedule_table_from_json(json_input: JsonInput, include_params: bool = True) -> None:
    """Normalize the provided JSON (schedule list or GeneralRecipe dict) and print a schedule table."""
    schedule = _load_json_input(json_input)
    print_schedule_table(schedule, include_params=include_params)


__all__ = [
    "print_flowchart",
    "print_schedule_table",
    "render_flowchart_from_json",
    "render_schedule_table_from_json",
]
