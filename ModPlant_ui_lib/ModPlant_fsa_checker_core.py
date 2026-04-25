#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any, Callable

STEP_PATTERN = re.compile(
    r"^\s*Step\s*(?P<idx>\d+)\s*\|\s*Op\s*:\s*(?P<op>.*?)\s*\|\s*Cost\s*:\s*(?P<cost>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\|\s*Dur\s*:\s*(?P<dur>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
    re.IGNORECASE,
)
COST_WARN_RE = re.compile(
    r"^Step\s+(?P<idx>\d+):\s+Cost mismatch\.\s+expected=(?P<exp>[-+]?(?:\d+(?:\.\d*)?|\.\d+)),\s+got=(?P<got>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\.$"
)
DUR_WARN_RE = re.compile(
    r"^Step\s+(?P<idx>\d+):\s+Duration mismatch\.\s+expected=(?P<exp>[-+]?(?:\d+(?:\.\d*)?|\.\d+)),\s+got=(?P<got>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\.$"
)
CONNECT_OP_RE = re.compile(
    r"^\s*Connect\(\s*(?P<out>[A-Za-z0-9_]+)\s*->\s*(?P<inp>[A-Za-z0-9_]+)\s*\)\s*for\s*(?P<material>.+?)\s*$",
    re.IGNORECASE,
)
TRANSFER_OP_RE = re.compile(
    r"^\s*(?P<kind>Dosing|Separation)\s*:\s*Open\s+Valve\s+of\s+(?P<out>[A-Za-z0-9_]+)\s+only\s*,\s*Draining\((?P<src>[A-Za-z0-9_]+)\)\s*,\s*Filling\((?P<dst>[A-Za-z0-9_]+)\)\s*,\s*\((?P<mats>.*)\)\s*$",
    re.IGNORECASE,
)
STIRRING_OP_RE = re.compile(
    r"^\s*Stirring\s*\(\s*(?P<wb>[A-Za-z0-9_]+)\s*\)\s*,\s*(?P<rpm>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*rpm\s*for\s*(?P<dur>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*s\s*$",
    re.IGNORECASE,
)
USAGE_OP_RE = re.compile(
    r"^\s*Usage\s*\(\s*(?P<wb>[A-Za-z0-9_]+)\s*\)\s*,\s*(?P<dur>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*s\s*:\s*None\s*$",
    re.IGNORECASE,
)
SETTLING_OP_RE = re.compile(
    r"^\s*Settling\s*\(\s*(?P<wb>[A-Za-z0-9_]+)\s*\)\s*,\s*(?P<dur>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*s\s*:\s*Settling\s*$",
    re.IGNORECASE,
)

_LAST_FLOW_TRACE: list[str] = []


def get_last_flow_trace() -> list[str]:
    return list(_LAST_FLOW_TRACE)


def normalize_op(op: str) -> str:
    return " ".join(op.strip().lower().split())


def should_relax_port_match(op_norm: str) -> bool:
    return op_norm.startswith(("connect", "dosing", "separation"))


def normalize_port_index(op_norm: str) -> str:
    return re.sub(r"\b([a-z]+\d+)_(in|out)\d+\b", r"\1_\2", op_norm)


def is_connect_op(op_norm: str) -> bool:
    return op_norm.startswith("connect")


def parse_steps(text: str) -> list[dict[str, Any]] | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    step_lines = [ln for ln in lines if ln.lower().startswith("step")]
    if not step_lines:
        return None

    out: list[dict[str, Any]] = []
    for ln in step_lines:
        m = STEP_PATTERN.match(ln)
        if not m:
            return None
        op_raw = m.group("op").strip()
        op_norm = normalize_op(op_raw)
        op_match = normalize_port_index(op_norm) if should_relax_port_match(op_norm) else op_norm
        out.append(
            {
                "step_idx": int(m.group("idx")),
                "op_raw": op_raw,
                "op_norm": op_norm,
                "op_match": op_match,
                "cost": float(m.group("cost")),
                "dur": float(m.group("dur")),
                "is_connect": is_connect_op(op_norm),
                "line": ln,
            }
        )
    return out


def extract_first_step_line(text: str) -> str | None:
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if STEP_PATTERN.match(ln):
            return ln
    return None


def detect_op_type(op_str: str) -> str:
    op = normalize_op(op_str)
    if op.startswith("connect("):
        return "connect"
    if op.startswith("dosing"):
        return "dosing"
    if op.startswith("stirring"):
        return "stirring"
    if op.startswith("usage"):
        return "usage"
    if op.startswith("settling"):
        return "settling"
    if op.startswith("separation"):
        return "separation"
    if op == "end" or op.startswith("end "):
        return "end"
    return "unknown"


def _extract_material_amounts(op_str: str) -> dict[str, float]:
    mats: dict[str, float] = {}
    for name, qty in re.findall(
        r"([A-Za-z0-9_]+)\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*litre",
        op_str,
        re.IGNORECASE,
    ):
        try:
            mats[name.strip()] = float(qty)
        except Exception:
            continue
    return mats


def _parse_material_string(material_str: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for entry in (material_str or "").split(","):
        if ":" not in entry:
            continue
        name, qty = entry.split(":", 1)
        name = name.strip()
        try:
            val = float(qty.strip().split()[0])
            result[name] = val
        except Exception:
            continue
    return result


def _materials_equal(d1: dict[str, float], d2: dict[str, float], tol: float = 1e-4) -> bool:
    if set(d1.keys()) != set(d2.keys()):
        return False
    for k, v in d1.items():
        tgt = d2.get(k, 0.0)
        if abs(tgt) < 1e-9:
            continue
        if abs(v - tgt) > tol:
            return False
    return True


def _sum_volume(mats: dict[str, float]) -> float:
    return sum(float(v) for v in mats.values())


def _normalize_rule_type(raw: str) -> str:
    t = (raw or "").strip().lower()
    if t == "mix":
        return "stirring"
    return t


def _op_entry(ModPlant_ops: dict[str, Any], wb: str, name: str, param: str | None = None) -> list[Any] | None:
    for op in ModPlant_ops.get(wb, []):
        if len(op) < 3:
            continue
        op_name = str(op[0])
        if op_name != name:
            continue
        if param is not None and name == "Stirring" and str(op[1]) != str(param):
            continue
        return op
    return None


def _op_cost(ModPlant_ops: dict[str, Any], wb: str, name: str, param: str | None = None) -> float:
    op = _op_entry(ModPlant_ops, wb, name, param)
    if not op:
        return 0.0
    try:
        return float(op[2])
    except Exception:
        return 0.0


def _op_rate(ModPlant_ops: dict[str, Any], wb: str, name: str) -> float:
    op = _op_entry(ModPlant_ops, wb, name)
    if not op:
        return 0.0
    try:
        return float(op[1])
    except Exception:
        return 0.0


def _warn_cost_dur_mismatch(
    warnings: list[str],
    step_no: int,
    got_cost: float,
    got_dur: float,
    exp_cost: float,
    exp_dur: float,
) -> None:
    if abs(got_cost - exp_cost) > 1e-3:
        warnings.append(f"Step {step_no}: Cost mismatch. expected={exp_cost:.3f}, got={got_cost:.3f}.")
    if abs(got_dur - exp_dur) > 1e-3:
        warnings.append(f"Step {step_no}: Duration mismatch. expected={exp_dur:.3f}, got={got_dur:.3f}.")


def _rule_result_materials(rule: dict[str, Any], total_volume: float | None = None) -> dict[str, float]:
    result_text = str(rule.get("result_text", "")).strip()
    if result_text.lower() == "end":
        return {"End": 0.0}
    parsed = _parse_material_string(result_text)
    if not parsed:
        return {}
    if total_volume is not None and len(parsed) == 1:
        key = next(iter(parsed.keys()))
        return {key: float(total_volume)}
    return parsed


def _parse_rule_param_rpm(param_text: str) -> str | None:
    txt = (param_text or "").strip()
    if not txt:
        return None
    left = txt.split("/", 1)[0].strip()
    parts = left.split()
    return parts[0] if parts else None


def check_prediction_against_rules(context_obj: dict[str, Any], prediction_text: str) -> dict[str, Any]:
    global _LAST_FLOW_TRACE
    _LAST_FLOW_TRACE = []

    def _trace(msg: str) -> None:
        _LAST_FLOW_TRACE.append(msg)

    def _fmt_mats(mats: dict[str, float]) -> str:
        if not mats:
            return "{}"
        parts = [f"{k}:{v:.3f}" for k, v in sorted(mats.items())]
        return "{ " + ", ".join(parts) + " }"

    def _fmt_conn(conn: tuple[str, str, str] | None) -> str:
        if conn is None:
            return "None"
        return f"{conn[0]}:{conn[1]} ({conn[2]})"

    def _trace_state_delta(
        step_no: int,
        prev_contents: dict[str, dict[str, float]],
        prev_connections: dict[str, tuple[str, str, str] | None],
        new_contents: dict[str, dict[str, float]],
        new_connections: dict[str, tuple[str, str, str] | None],
    ) -> None:
        changed = False
        for wb in sorted(set(prev_contents.keys()) | set(new_contents.keys())):
            p = prev_contents.get(wb, {})
            n = new_contents.get(wb, {})
            if p != n:
                changed = True
                _trace(f"  [Step {step_no}] Content {wb}: {_fmt_mats(p)} -> {_fmt_mats(n)}")
        for out_port in sorted(set(prev_connections.keys()) | set(new_connections.keys())):
            p = prev_connections.get(out_port)
            n = new_connections.get(out_port)
            if p != n:
                changed = True
                _trace(f"  [Step {step_no}] Connection {out_port}: {_fmt_conn(p)} -> {_fmt_conn(n)}")
        if not changed:
            _trace(f"  [Step {step_no}] State unchanged")

    rules = list(context_obj.get("reaction_rules") or [])
    ModPlant_ops = context_obj.get("ModPlant_ops") or {}
    ModPlant_interfaces = context_obj.get("ModPlant_interfaces") or {}
    ModPlant_resources = context_obj.get("ModPlant_resources") or {}
    ModPlant_max_volume = context_obj.get("ModPlant_maximum_volume") or {}
    parsed = parse_steps(prediction_text)
    if not parsed:
        _trace("[FlowCheck] Failed: LLM output format invalid; cannot parse step lines.")
        return {
            "ok": False,
            "error_step": 1,
            "reason": "LLM output format invalid; cannot parse step lines.",
            "warnings": [],
        }

    rule_idx = 0
    reached_end = False
    warnings: list[str] = []

    wb_names = set(ModPlant_ops.keys()) | set(ModPlant_interfaces.keys()) | set(ModPlant_max_volume.keys()) | set(ModPlant_resources.keys())
    output_owner: dict[str, str] = {}
    input_owner: dict[str, str] = {}
    connections: dict[str, tuple[str, str, str] | None] = {}
    contents: dict[str, dict[str, float]] = {}
    max_vol: dict[str, float] = {}

    for wb in wb_names:
        contents[wb] = {}
        try:
            max_vol[wb] = float((ModPlant_max_volume.get(wb) or [0.0])[0])
        except Exception:
            max_vol[wb] = 0.0

        for port_type, port_name in (ModPlant_interfaces.get(wb, []) or []):
            ptype = str(port_type).strip().lower()
            pnm = str(port_name).strip()
            if ptype == "output":
                output_owner[pnm] = wb
                connections[pnm] = None
            elif ptype == "input":
                input_owner[pnm] = wb

    for wb, res in (ModPlant_resources or {}).items():
        if wb not in contents:
            contents[wb] = {}
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            mat = str(res[0]).strip()
            try:
                qty = float(res[1])
            except Exception:
                qty = 0.0
            if mat:
                contents[wb][mat] = contents[wb].get(mat, 0.0) + qty

    _trace("[FlowCheck] Initial contents:")
    for wb in sorted(contents.keys()):
        _trace(f"  [Init] {wb}: {_fmt_mats(contents[wb])}")
    _trace("[FlowCheck] Initial active connections: none")

    parsed_rules: list[dict[str, Any]] = []
    for r in rules:
        parsed_rules.append(
            {
                "type": _normalize_rule_type(str(r.get("Reaction Type", ""))),
                "inputs": _parse_material_string(str(r.get("Inputs", ""))),
                "param_text": str(r.get("Reaction Param", "")),
                "param_mats": _parse_material_string(str(r.get("Reaction Param", ""))),
                "result_text": str(r.get("Result", "")),
            }
        )

    def _find_matching_rule(op_type: str, predicate: Callable[[dict[str, Any]], bool]) -> int | None:
        for i in range(rule_idx, len(parsed_rules)):
            rr = parsed_rules[i]
            if rr["type"] != op_type:
                continue
            if predicate(rr):
                return i
        return None

    def _has_end_material() -> bool:
        for mats in contents.values():
            if "End" in mats:
                return True
        return False

    for idx, step in enumerate(parsed, start=1):
        op = str(step["op_raw"])
        op_type = detect_op_type(op)
        prev_contents = {wb: dict(mats) for wb, mats in contents.items()}
        prev_connections = dict(connections)
        _trace(f"[FlowCheck] Step {idx}: {op}")

        if op_type == "unknown":
            _trace(f"[FlowCheck] Failed at step {idx}: Unsupported operation format.")
            return {
                "ok": False,
                "error_step": idx,
                "reason": f"Unsupported operation format: {op}",
                "warnings": warnings,
            }

        if op_type == "connect":
            m = CONNECT_OP_RE.match(op)
            if not m:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Cannot parse Connect operation: {op}",
                    "warnings": warnings,
                }
            out_port = m.group("out").strip()
            in_port = m.group("inp").strip()
            material = m.group("material").strip()

            src_wb = output_owner.get(out_port)
            dst_wb = input_owner.get(in_port)
            if not src_wb or not dst_wb:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Invalid port mapping in Connect: {out_port} -> {in_port}.",
                    "warnings": warnings,
                }
            if not _op_entry(ModPlant_ops, src_wb, "Connect") or not _op_entry(ModPlant_ops, dst_wb, "Connect"):
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Connect capability missing on {src_wb} or {dst_wb}.",
                    "warnings": warnings,
                }

            cur_conn = connections.get(out_port)
            if cur_conn is not None and (cur_conn[0] != dst_wb or cur_conn[1] != in_port):
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Output port already connected: {out_port} -> {cur_conn[1]} ({cur_conn[0]}).",
                    "warnings": warnings,
                }
            for other_out, conn in connections.items():
                if other_out == out_port or conn is None:
                    continue
                if conn[1] == in_port:
                    return {
                        "ok": False,
                        "error_step": idx,
                        "reason": f"Input port already occupied: {in_port}.",
                        "warnings": warnings,
                    }

            connections[out_port] = (dst_wb, in_port, material)
            exp_cost = _op_cost(ModPlant_ops, src_wb, "Connect") + _op_cost(ModPlant_ops, dst_wb, "Connect")
            _warn_cost_dur_mismatch(warnings, idx, step["cost"], step["dur"], exp_cost, 3.0)
            _trace_state_delta(idx, prev_contents, prev_connections, contents, connections)
            continue

        if op_type == "end":
            reached_end = True
            if not _has_end_material() and rule_idx < len(parsed_rules):
                _trace(f"[FlowCheck] Failed at step {idx}: End reached before terminal state.")
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Reached End too early. Remaining reaction rules: {len(parsed_rules) - rule_idx}.",
                    "warnings": warnings,
                }
            _warn_cost_dur_mismatch(warnings, idx, step["cost"], step["dur"], 0.0, 0.0)
            _trace_state_delta(idx, prev_contents, prev_connections, contents, connections)
            break

        if op_type in {"dosing", "separation"}:
            m = TRANSFER_OP_RE.match(op)
            if not m:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Cannot parse transfer operation: {op}",
                    "warnings": warnings,
                }

            out_port = m.group("out").strip()
            src = m.group("src").strip()
            dst = m.group("dst").strip()
            mats = _extract_material_amounts(m.group("mats"))
            if not mats:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": "Transfer operation has no parsable materials.",
                    "warnings": warnings,
                }

            if output_owner.get(out_port) != src:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Output port {out_port} does not belong to {src}.",
                    "warnings": warnings,
                }
            conn = connections.get(out_port)
            if conn is None:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"No active connection for output port {out_port}.",
                    "warnings": warnings,
                }
            if conn[0] != dst:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Connection mismatch for {out_port}. expected target={conn[0]}, got={dst}.",
                    "warnings": warnings,
                }
            if not _op_entry(ModPlant_ops, src, "Draining") or not _op_entry(ModPlant_ops, dst, "Filling"):
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Draining/Filling capability missing for transfer {src}->{dst}.",
                    "warnings": warnings,
                }

            src_before = dict(contents.get(src, {}))
            dst_before = dict(contents.get(dst, {}))
            dst_pre_for_dose = dict(dst_before)

            added_vol = _sum_volume(mats)
            if _sum_volume(dst_before) + added_vol > max_vol.get(dst, 0.0) + 1e-9:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Capacity exceeded in {dst}. limit={max_vol.get(dst, 0.0)}, attempted={_sum_volume(dst_before) + added_vol:.3f}.",
                    "warnings": warnings,
                }

            matched_sep_rule: dict[str, Any] | None = None
            matched_sep_rule_idx: int | None = None
            pre_src_for_sep = dict(src_before)
            if op_type == "separation":
                for i in range(rule_idx, len(parsed_rules)):
                    rr = parsed_rules[i]
                    if rr["type"] != "separation":
                        continue
                    param_mat = rr["param_text"].strip()
                    if not param_mat or param_mat not in mats:
                        continue
                    if not _materials_equal(rr["inputs"], pre_src_for_sep):
                        continue
                    target_after_sep = _rule_result_materials(rr)
                    src_vol = _sum_volume(pre_src_for_sep)
                    remain_vol = 0.0 if rr["result_text"].strip().lower() == "end" else _sum_volume(target_after_sep)
                    expected_transfer = max(0.0, src_vol - remain_vol)
                    if abs(mats.get(param_mat, 0.0) - expected_transfer) > 1e-3:
                        continue
                    matched_sep_rule_idx = i
                    matched_sep_rule = rr
                    break
                if matched_sep_rule_idx is None:
                    return {
                        "ok": False,
                        "error_step": idx,
                        "reason": "Separation does not match any valid reaction-rule state transition.",
                        "warnings": warnings,
                    }
                if _sum_volume(pre_src_for_sep) + 1e-9 < added_vol:
                    return {
                        "ok": False,
                        "error_step": idx,
                        "reason": f"Insufficient source volume in {src}. needed={added_vol:.3f}, available={_sum_volume(pre_src_for_sep):.3f}.",
                        "warnings": warnings,
                    }
            else:
                for mat, qty in mats.items():
                    if src_before.get(mat, 0.0) + 1e-9 < qty:
                        return {
                            "ok": False,
                            "error_step": idx,
                            "reason": f"Insufficient source material in {src}: {mat}. needed={qty}, available={src_before.get(mat, 0.0)}.",
                            "warnings": warnings,
                        }

            if op_type == "dosing":
                for mat, qty in mats.items():
                    src_before[mat] = src_before.get(mat, 0.0) - qty
                    if src_before[mat] <= 1e-9:
                        src_before.pop(mat, None)
                    dst_before[mat] = dst_before.get(mat, 0.0) + qty
                contents[src] = src_before
                contents[dst] = dst_before

                matched_rule = _find_matching_rule(
                    "dosing",
                    lambda rr: _materials_equal(rr["param_mats"], mats) and _materials_equal(rr["inputs"], dst_pre_for_dose),
                )
                if matched_rule is not None:
                    rule_idx = matched_rule + 1
                speed = min(_op_rate(ModPlant_ops, src, "Draining"), _op_rate(ModPlant_ops, dst, "Filling"))
                speed = speed if speed > 1e-9 else 1.0
                exp_cost = (_op_cost(ModPlant_ops, src, "Draining") + _op_cost(ModPlant_ops, dst, "Filling")) * added_vol
                exp_dur = added_vol / speed
                _warn_cost_dur_mismatch(warnings, idx, step["cost"], step["dur"], exp_cost, exp_dur)
                _trace_state_delta(idx, prev_contents, prev_connections, contents, connections)
                continue

            for mat, qty in mats.items():
                dst_before[mat] = dst_before.get(mat, 0.0) + qty
            contents[dst] = dst_before
            assert matched_sep_rule is not None
            assert matched_sep_rule_idx is not None
            contents[src] = _rule_result_materials(matched_sep_rule)
            rule_idx = matched_sep_rule_idx + 1
            exp_cost = _op_cost(ModPlant_ops, src, "Draining") + _op_cost(ModPlant_ops, dst, "Filling")
            _warn_cost_dur_mismatch(warnings, idx, step["cost"], step["dur"], exp_cost, 0.0)
            _trace_state_delta(idx, prev_contents, prev_connections, contents, connections)
            continue

        if op_type == "stirring":
            m = STIRRING_OP_RE.match(op)
            if not m:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Cannot parse Stirring operation: {op}",
                    "warnings": warnings,
                }
            wb = m.group("wb").strip()
            rpm = m.group("rpm").strip()
            duration = float(m.group("dur"))
            if not _op_entry(ModPlant_ops, wb, "Stirring", rpm):
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Stirring capability missing on {wb} for rpm={rpm}.",
                    "warnings": warnings,
                }
            wb_before = dict(contents.get(wb, {}))
            matched_rule_idx = _find_matching_rule(
                "stirring",
                lambda rr: _materials_equal(rr["inputs"], wb_before) and _parse_rule_param_rpm(rr["param_text"]) == rpm,
            )
            if matched_rule_idx is None:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Stirring preconditions not met on {wb} for recipe rule.",
                    "warnings": warnings,
                }
            rr = parsed_rules[matched_rule_idx]
            contents[wb] = _rule_result_materials(rr, total_volume=_sum_volume(wb_before))
            rule_idx = matched_rule_idx + 1
            exp_cost = _op_cost(ModPlant_ops, wb, "Stirring", rpm)
            _warn_cost_dur_mismatch(warnings, idx, step["cost"], step["dur"], exp_cost, duration)
            _trace_state_delta(idx, prev_contents, prev_connections, contents, connections)
            continue

        if op_type == "usage":
            m = USAGE_OP_RE.match(op)
            if not m:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Cannot parse Usage operation: {op}",
                    "warnings": warnings,
                }
            wb = m.group("wb").strip()
            if not _op_entry(ModPlant_ops, wb, "None"):
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Usage prerequisite missing: operation None not available on {wb}.",
                    "warnings": warnings,
                }
            wb_before = dict(contents.get(wb, {}))
            matched_rule_idx = _find_matching_rule("usage", lambda rr: _materials_equal(rr["inputs"], wb_before))
            if matched_rule_idx is None:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Usage preconditions not met on {wb} for recipe rule.",
                    "warnings": warnings,
                }
            rr = parsed_rules[matched_rule_idx]
            contents[wb] = _rule_result_materials(rr, total_volume=_sum_volume(wb_before))
            rule_idx = matched_rule_idx + 1
            _warn_cost_dur_mismatch(warnings, idx, step["cost"], step["dur"], 0.0, 0.0)
            _trace_state_delta(idx, prev_contents, prev_connections, contents, connections)
            continue

        if op_type == "settling":
            m = SETTLING_OP_RE.match(op)
            if not m:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Cannot parse Settling operation: {op}",
                    "warnings": warnings,
                }
            wb = m.group("wb").strip()
            if not _op_entry(ModPlant_ops, wb, "Settling"):
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Settling capability missing on {wb}.",
                    "warnings": warnings,
                }
            wb_before = dict(contents.get(wb, {}))
            matched_rule_idx = _find_matching_rule("settling", lambda rr: _materials_equal(rr["inputs"], wb_before))
            if matched_rule_idx is None:
                return {
                    "ok": False,
                    "error_step": idx,
                    "reason": f"Settling preconditions not met on {wb} for recipe rule.",
                    "warnings": warnings,
                }
            rr = parsed_rules[matched_rule_idx]
            contents[wb] = _rule_result_materials(rr)
            rule_idx = matched_rule_idx + 1
            exp_cost = _op_cost(ModPlant_ops, wb, "Settling")
            _warn_cost_dur_mismatch(warnings, idx, step["cost"], step["dur"], exp_cost, 0.0)
            _trace_state_delta(idx, prev_contents, prev_connections, contents, connections)
            continue

        _trace(f"[FlowCheck] Failed at step {idx}: Unsupported operation type {op_type}.")
        return {
            "ok": False,
            "error_step": idx,
            "reason": f"Unsupported operation type: {op_type}.",
            "warnings": warnings,
        }

    if not reached_end:
        _trace("[FlowCheck] Failed: No End step produced.")
        return {
            "ok": False,
            "error_step": len(parsed),
            "reason": "No End step produced.",
            "warnings": warnings,
        }

    if rule_idx < len(parsed_rules) and not _has_end_material():
        _trace("[FlowCheck] Failed: Process ended before reaching terminal reaction state.")
        return {
            "ok": False,
            "error_step": len(parsed),
            "reason": f"Process ended before reaching terminal reaction state. Remaining rules: {len(parsed_rules) - rule_idx}.",
            "warnings": warnings,
        }

    _trace("[FlowCheck] PASS: Process path is executable.")
    return {
        "ok": True,
        "error_step": None,
        "reason": "Process path is executable under DYNA3-style state-transition simulation.",
        "warnings": warnings,
    }


def compare_prediction_to_reference(prediction_text: str, ref_lines: list[str]) -> dict[str, Any]:
    ref_text = "\n".join(ref_lines)
    ref_steps = parse_steps(ref_text)
    pred_steps = parse_steps(prediction_text)
    if ref_steps is None:
        return {"ok": False, "reason": "Reference steps cannot be parsed.", "step": 1}
    if pred_steps is None:
        return {"ok": False, "reason": "Prediction steps cannot be parsed.", "step": 1}

    n = min(len(ref_steps), len(pred_steps))
    warnings: list[str] = []
    for i in range(n):
        r = ref_steps[i]
        p = pred_steps[i]
        if r["op_match"] != p["op_match"]:
            return {
                "ok": False,
                "step": i + 1,
                "reason": f"Operation mismatch. expected={r['op_raw']}, got={p['op_raw']}.",
            }
        if abs(r["cost"] - p["cost"]) > 1e-3:
            warnings.append(f"Step {i + 1}: Cost mismatch. expected={r['cost']:.3f}, got={p['cost']:.3f}.")
        if abs(r["dur"] - p["dur"]) > 1e-3:
            warnings.append(f"Step {i + 1}: Duration mismatch. expected={r['dur']:.3f}, got={p['dur']:.3f}.")

    if len(pred_steps) < len(ref_steps):
        return {
            "ok": False,
            "step": len(pred_steps) + 1,
            "reason": "Prediction ended early.",
        }
    if len(pred_steps) > len(ref_steps):
        return {
            "ok": False,
            "step": len(ref_steps) + 1,
            "reason": "Prediction has extra steps after reference path.",
        }
    if warnings:
        return {
            "ok": True,
            "step": None,
            "reason": "Process path is executable (operation sequence matched).",
            "warnings": warnings,
        }
    return {
        "ok": True,
        "step": None,
        "reason": "Prediction matches FSA+solver reference path.",
        "warnings": [],
    }


def _format_step_line(step_idx: int, op: str, cost: float, dur: float) -> str:
    return f"Step {step_idx} | Op: {op} | Cost: {cost:.3f} | Dur: {dur:.3f}"


def apply_cost_duration_corrections(step_lines: list[str], warnings: list[str]) -> tuple[list[str], list[str], bool]:
    corrections: dict[int, dict[str, float]] = {}
    for w in warnings or []:
        txt = str(w).strip()
        m_cost = COST_WARN_RE.match(txt)
        if m_cost:
            idx = int(m_cost.group("idx"))
            corrections.setdefault(idx, {})["cost"] = float(m_cost.group("exp"))
            continue
        m_dur = DUR_WARN_RE.match(txt)
        if m_dur:
            idx = int(m_dur.group("idx"))
            corrections.setdefault(idx, {})["dur"] = float(m_dur.group("exp"))
            continue

    changed = False
    notes: list[str] = []
    out = list(step_lines)
    for idx in sorted(corrections.keys()):
        if idx < 1 or idx > len(out):
            continue
        m = STEP_PATTERN.match(out[idx - 1].strip())
        if not m:
            continue
        op = m.group("op").strip()
        old_cost = float(m.group("cost"))
        old_dur = float(m.group("dur"))
        new_cost = corrections[idx].get("cost", old_cost)
        new_dur = corrections[idx].get("dur", old_dur)
        if abs(new_cost - old_cost) <= 1e-9 and abs(new_dur - old_dur) <= 1e-9:
            continue
        out[idx - 1] = _format_step_line(idx, op, new_cost, new_dur)
        changed = True
        cost_note = f"cost {old_cost:.3f}->{new_cost:.3f}" if abs(new_cost - old_cost) > 1e-9 else "cost unchanged"
        dur_note = f"dur {old_dur:.3f}->{new_dur:.3f}" if abs(new_dur - old_dur) > 1e-9 else "dur unchanged"
        notes.append(f"Step {idx}: corrected {cost_note}, {dur_note}.")

    return out, notes, changed


__all__ = [
    "STEP_PATTERN",
    "COST_WARN_RE",
    "DUR_WARN_RE",
    "CONNECT_OP_RE",
    "TRANSFER_OP_RE",
    "STIRRING_OP_RE",
    "USAGE_OP_RE",
    "SETTLING_OP_RE",
    "get_last_flow_trace",
    "normalize_op",
    "should_relax_port_match",
    "normalize_port_index",
    "is_connect_op",
    "parse_steps",
    "extract_first_step_line",
    "detect_op_type",
    "check_prediction_against_rules",
    "compare_prediction_to_reference",
    "apply_cost_duration_corrections",
]
