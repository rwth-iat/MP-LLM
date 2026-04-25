from pathlib import Path
import json
from collections import OrderedDict, defaultdict
import pandas as pd
import re
import heapq

def _load_general_recipe_json(path_or_obj):
    if isinstance(path_or_obj, (str, Path)):
        p = Path(path_or_obj)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    elif isinstance(path_or_obj, dict):
        return path_or_obj
    else:
        raise TypeError("Expected a file path or dict")

def _get_param(pe: dict, id_prefix: str | None = None, description_contains: str | None = None):
    for prm in pe.get("Parameters", []):
        pid = prm.get("ID") or ""
        desc = prm.get("Description") or ""
        if id_prefix and pid.startswith(id_prefix):
            return prm
        if description_contains and description_contains in desc:
            return prm
    return None

def generate_reaction_rules_from_general_recipe_json(path_or_obj):
    data = _load_general_recipe_json(path_or_obj)

    # Get all process elements by ID
    process_elements = {pe['ID']: pe for pe in data.get("ProcessElements", [])}

    process_ids = list(process_elements.keys())

    # Build graph
    successors = defaultdict(list)
    predecessors_count = {pid: 0 for pid in process_ids}

    # First, find outputs of each process
    outputs = {}
    for link in data.get("DirectedLinks", []):
        from_id = link['FromID']
        to_id = link['ToID']
        if from_id in process_ids and to_id not in process_ids:
            outputs[from_id] = to_id

    # Now, build predecessors and successors
    for link in data.get("DirectedLinks", []):
        from_id = link['FromID']
        to_id = link['ToID']
        if to_id in process_ids:
            pred_process = None
            if from_id in process_ids:
                pred_process = from_id
            elif from_id in outputs.values():
                for p, out in outputs.items():
                    if out == from_id:
                        pred_process = p
                        break
            if pred_process:
                successors[pred_process].append(to_id)
                predecessors_count[to_id] += 1

    # Topological sort with priority
    def _key(pid):
        m = re.search(r"(\d{3})", pid)
        return int(m.group(1)) if m else 0

    ready = []
    for pid in process_ids:
        if predecessors_count[pid] == 0:
            heapq.heappush(ready, (_key(pid), pid))

    topo_order = []
    while ready:
        _, pid = heapq.heappop(ready)
        topo_order.append(pid)
        for succ in successors[pid]:
            predecessors_count[succ] -= 1
            if predecessors_count[succ] == 0:
                heapq.heappush(ready, (_key(succ), succ))

    if len(topo_order) != len(process_ids):
        raise ValueError("Graph has cycle or disconnected")

    # Now, generate rules in order
    rules = OrderedDict()
    current_state = []  # list of (name, vol, unit, breakdown: OrderedDict ing:vol)

    for pid in topo_order:
        pe = process_elements[pid]
        desc = pe.get("Description", "")
        sem = pe.get("SemanticDescription", "")

        inputs_list = [f"{name} : {vol} {unit}" for name, vol, unit, _ in current_state]
        inputs_tuple = tuple(inputs_list)

        if "Dosing" in sem:
            ingr = desc.split(" ")[1]
            prm = _get_param(pe, description_contains="Amount of Dosing")
            val = float(prm.get("ValueString", "0"))
            unit = prm.get("UnitOfMeasure", "").split("/")[-1]
            rparam = f"{ingr}: {val} {unit}"
            new_comp_str = f"{ingr}: {val} {unit}"
            result = ", ".join(inputs_list + [new_comp_str]) if inputs_list else new_comp_str
            rules[(inputs_tuple, "Dosing", rparam)] = (result,)
            new_bd = OrderedDict({ingr: val})
            current_state.append((ingr, val, unit, new_bd))

        elif "MixingOfLiquids" in sem:
            rpm_prm = _get_param(pe, id_prefix="Revolutions_per_minute")
            rpm_val = rpm_prm.get("ValueString", "")
            dur_prm = _get_param(pe, description_contains="Duration of the process step mixing")
            dur_val = dur_prm.get("ValueString", "")
            dur_unit = dur_prm.get("UnitOfMeasure", "").split("/")[-1]
            rparam = f"{rpm_val} rpm / {dur_val} {dur_unit}"
            all_bd = OrderedDict()
            for _, _, _, bd in current_state:
                all_bd.update(bd)
            all_ings = sorted(all_bd.keys())
            name = "_".join(all_ings) + "_mixed"
            vol = sum(all_bd.values())
            unit = current_state[0][2] if current_state else "litre"
            result = f"{name} : {vol} {unit}"
            rules[(inputs_tuple, "Mix", rparam)] = (result,)
            current_state = [(name, vol, unit, all_bd)]

        elif "Usage" in sem:
            assert len(current_state) == 1
            name, vol, unit, bd = current_state[0]
            all_ings = sorted(bd.keys())
            new_name = "_".join(all_ings) + "_used"
            dur_prm = _get_param(pe, id_prefix="Usage_Duration")
            dur_val = dur_prm.get("ValueString", "")
            dur_unit = dur_prm.get("UnitOfMeasure", "").split("/")[-1]
            rparam = f"{dur_val} {dur_unit}"
            inputs_tuple = (f"{name} : {vol} {unit}",)
            result = f"{new_name} : {vol} {unit}"
            rules[(inputs_tuple, "Usage", rparam)] = (result,)
            current_state = [(new_name, vol, unit, bd)]

        elif "Collecting" in sem:
            assert len(current_state) == 1
            name, vol, unit, bd = current_state[0]
            vol_prm = _get_param(pe, id_prefix="Collecting_Volume")
            collect_vol = float(vol_prm.get("ValueString", "0"))
            collect_unit = vol_prm.get("UnitOfMeasure", "").split("/")[-1]
            ratio_prm = _get_param(pe, description_contains="Ingredient ratio")
            ratio_str = ratio_prm.get("ValueString", "") if ratio_prm else ""
            rparam = f"{collect_vol} {collect_unit}"
            if ratio_str:
                ratio_dict = OrderedDict()
                for pair in ratio_str.split(","):
                    k, v = pair.split(":")
                    ratio_dict[k.strip()] = float(v.strip())
                total_parts = sum(ratio_dict.values())
                new_bd = OrderedDict((k, collect_vol * p / total_parts) for k, p in ratio_dict.items())
                all_ings = list(ratio_dict.keys())
                bd = new_bd
            else:
                all_ings = sorted(bd.keys())
            new_name = "_".join(all_ings) + "_collected"
            result = f"{new_name} : {collect_vol} {collect_unit}"
            inputs_tuple = (f"{name} : {vol} {unit}",)
            rules[(inputs_tuple, "Collect", rparam)] = (result,)
            current_state = [(new_name, collect_vol, collect_unit, bd)]

        elif "Settling" in sem:
            assert len(current_state) == 1
            name, vol, unit, bd = current_state[0]
            all_ings = list(bd.keys())  # preserve order
            new_name = "_".join(all_ings) + "_settled"
            dur_prm = _get_param(pe, id_prefix="Settling_Duration")
            dur_val = dur_prm.get("ValueString", "")
            dur_unit = dur_prm.get("UnitOfMeasure", "").split("/")[-1]
            rparam = f"{dur_val} {dur_unit}"
            inputs_tuple = (f"{name} : {vol} {unit}",)
            result = f"{new_name} : {vol} {unit}"
            rules[(inputs_tuple, "Settling", rparam)] = (result,)
            current_state = [(new_name, vol, unit, bd)]

        elif "Separation" in sem:
            assert len(current_state) == 1
            name, vol, unit, bd = current_state[0]
            sep_ingr = desc.split(" of ")[-1]
            sep_vol_prm = _get_param(pe, description_contains="Volume to separate")
            sep_vol = float(sep_vol_prm.get("ValueString", "0"))
            rparam = sep_ingr
            new_bd = bd.copy()
            if sep_ingr in new_bd:
                del new_bd[sep_ingr]
            new_vol = vol - sep_vol
            new_ings = list(new_bd.keys())  # preserve order
            new_name = "_".join(new_ings) + "_settled" if new_ings else "End"
            inputs_tuple = (f"{name} : {vol} {unit}",)
            result = "End" if not new_ings else f"{new_name} : {new_vol} {unit}"
            rules[(inputs_tuple, "Separation", rparam)] = (result,)
            if new_ings:
                current_state = [(new_name, new_vol, unit, new_bd)]
            else:
                current_state = []

    return rules

def rules_to_dataframe(rules: OrderedDict):
    rows = []
    for key, (res,) in rules.items():
        inputs, rtype, rparam = key
        input_list = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        row = {
            "Inputs": ", ".join(input_list),
            "Reaction Type": rtype,
            "Reaction Param": rparam,
            "Result": res,
        }
        rows.append(row)
    return pd.DataFrame(rows)