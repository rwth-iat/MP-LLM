# =========================
# Stage 2: GeneralRecipe (B2MML) generator
# =========================
# Fix: Last mix produces Product, then Product -> Usage. No Separation -> Product link.
# Also: Intermediates only between mixes (Mix_j -> InterMix_j -> Mix_{j+1}) and after-usage.

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List

# Namespaces
NS_B2MML = "http://www.mesa.org/xml/B2MML"
NS_XSI   = "http://www.w3.org/2001/XMLSchema-instance"
ET.register_namespace("b2mml", NS_B2MML)
ET.register_namespace("xsi", NS_XSI)

def _b(tag: str) -> str:
    return f"{{{NS_B2MML}}}{tag}"

def _slug(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:maxlen] or "Flow"

def _ts_suffix() -> str:
    """Run-wide element-ID suffix: '/MM.DD/HH:MM:SS.mm/' style (2-digit millis)."""
    return "/" + datetime.now().strftime("%m.%d/%H:%M:%S.%f")[:-5] + "/"

def _sum_weights_ratio(order: Dict, totals_keys: List[str]) -> Dict[str, float]:
    """
    Derive Collecting ratio:
      - If order["collecting"]["ratio"] exists -> use that as-is (numbers per ingredient).
      - Else -> sum per-occurrence weights in order["ratio"][ingr].
    Returns a dict {INGR: raw_sum}.
    """
    col = order.get("collecting") or {}
    if "ratio" in col and isinstance(col["ratio"], dict):
        raw = {str(k).upper(): float(v) for k, v in col["ratio"].items()}
        need = set(totals_keys)
        got  = set(raw.keys())
        if got != need:
            missing = list(need - got)
            extra   = list(got - need)
            if missing:
                raise ValueError(f"collecting.ratio missing ingredients: {missing}")
            if extra:
                raise ValueError(f"collecting.ratio unknown ingredients: {extra}")
        return {k: raw[k] for k in totals_keys}
    # Fallback: sum order["ratio"] lists
    order_ratio = order.get("ratio", {})
    out: Dict[str, float] = {}
    for ingr in totals_keys:
        lst = order_ratio.get(ingr, [])
        if not isinstance(lst, list) or not lst:
            raise ValueError(f"ratio[{ingr}] must be a non-empty list to derive collecting ratios.")
        s = 0.0
        for v in lst:
            s += float(v)
        out[ingr] = s
    return out

def _fmt_ratio_text(ratio: Dict[str, float]) -> str:
    """Make a compact, human-readable ratio string: 'A: 3, B: 2, C: 3'."""
    def f(x: float) -> str:
        xi = int(x)
        return str(xi) if abs(x - xi) < 1e-9 else f"{x:.6g}"
    return ", ".join(f"{k}: {f(v)}" for k, v in ratio.items())

def _collect_totals(schedule: List[Dict]) -> Dict[str, float]:
    """Sum total volume per ingredient from dosing occurrences."""
    totals: Dict[str, float] = {}
    for e in schedule:
        if e["type"] == "dose":
            ingr = e["params"]["ingredient"]
            portion = float(e["params"]["portion_L"])
            totals[ingr] = totals.get(ingr, 0.0) + portion
    return totals

def _total_volume(schedule: List[Dict]) -> float:
    """Final mixture volume equals sum of all dosing portions."""
    return sum(float(e["params"]["portion_L"]) for e in schedule if e["type"] == "dose")

def _grecipe_id_from_flow(schedule: List[Dict]) -> str:
    """Readable flow summary (used in file name)."""
    tokens = []
    for e in schedule:
        t = e["type"]
        if t == "dose":
            tokens.append(f"D{e['params']['ingredient']}")
        elif t == "mix":
            tokens.append(f"M{e['params']['rpm']}")
        elif t == "usage":
            tokens.append("U")
        # elif t == "collecting":
        #     tokens.append("C")
        elif t == "settling":
            tokens.append("S")
        elif t == "sep":
            tokens.append(f"Sep{e['params']['ingredient']}")
    base = _slug("-".join(tokens), 60)
    return f"{base}"

def _materials_block(parent, block_id: str, desc: str, mtype: str, material_ids: List[str]):
    """Create a <b2mml:Materials> block with child <Material><ID> references (no amounts)."""
    mats = ET.SubElement(parent, _b("Materials"))
    ET.SubElement(mats, _b("ID")).text = block_id
    ET.SubElement(mats, _b("Description")).text = desc
    ET.SubElement(mats, _b("MaterialsType")).text = mtype
    for mid in material_ids:
        m = ET.SubElement(mats, _b("Material"))
        ET.SubElement(m, _b("ID")).text = mid

def _amount_block(parent, qty: float):
    """Append Amount block (QuantityString/DataType/UnitOfMeasure/Key)."""
    amt = ET.SubElement(parent, _b("Amount"))
    ET.SubElement(amt, _b("QuantityString")).text = f"{qty:.6g}"
    ET.SubElement(amt, _b("DataType")).text = "double"
    ET.SubElement(amt, _b("UnitOfMeasure")).text = "http://si-digital-framework.org/SI/units/litre"
    ET.SubElement(amt, _b("Key")).text = "http://qudt.org/vocab/quantitykind/LiquidVolume"

def _duration_param(pe: ET.Element, pid: str, desc: str, seconds: int, param_sfx: str):
    """Append a ProcessElementParameter with duration in seconds (with param ID suffix)."""
    p = ET.SubElement(pe, _b("ProcessElementParameter"))
    ET.SubElement(p, _b("ID")).text = pid + param_sfx
    ET.SubElement(p, _b("Description")).text = desc
    v = ET.SubElement(p, _b("Value"))
    ET.SubElement(v, _b("ValueString")).text = str(int(seconds))
    ET.SubElement(v, _b("DataType")).text = "int"
    ET.SubElement(v, _b("UnitOfMeasure")).text = "http://si-digital-framework.org/SI/units/second"
    ET.SubElement(v, _b("Key")).text = "http://www.w3.org/2006/time#Duration"

def _rpm_param(pe: ET.Element, rpm: int, idx: int, param_sfx: str):
    """RPM parameter (with param ID suffix)."""
    p = ET.SubElement(pe, _b("ProcessElementParameter"))
    ET.SubElement(p, _b("ID")).text = f"Revolutions_per_minute{idx:03d}" + param_sfx
    ET.SubElement(p, _b("Description")).text = "Revolutions per minute"
    v = ET.SubElement(p, _b("Value"))
    ET.SubElement(v, _b("ValueString")).text = str(int(rpm))
    ET.SubElement(v, _b("DataType")).text = "int"
    ET.SubElement(v, _b("UnitOfMeasure")).text = "http://qudt.org/vocab/unit/REV-PER-MIN"
    ET.SubElement(v, _b("Key")).text = "http://qudt.org/vocab/quantitykind/RotationalVelocity"

def _string_param(pe: ET.Element, pid: str, desc: str, value: str, param_sfx: str, unit_of_measure: str | None = None, key: str | None = None):
    """Generic string parameter (with param ID suffix)."""
    p = ET.SubElement(pe, _b("ProcessElementParameter"))
    ET.SubElement(p, _b("ID")).text = pid + param_sfx
    ET.SubElement(p, _b("Description")).text = desc
    v = ET.SubElement(p, _b("Value"))
    ET.SubElement(v, _b("ValueString")).text = value
    ET.SubElement(v, _b("DataType")).text = "string"
    if unit_of_measure:
        ET.SubElement(v, _b("UnitOfMeasure")).text = unit_of_measure
    if key:
        ET.SubElement(v, _b("Key")).text = key

def _other_info(pe: ET.Element, key: str, uri: str):
    """Append an OtherInformation block."""
    oi = ET.SubElement(pe, _b("OtherInformation"))
    ET.SubElement(oi, _b("OtherInfoID")).text = "SemanticDescription"
    ET.SubElement(oi, _b("Description")).text = "URI referencing the Ontology Class definition"
    ov = ET.SubElement(oi, _b("OtherValue"))
    ET.SubElement(ov, _b("ValueString")).text = uri
    ET.SubElement(ov, _b("DataType")).text = "uriReference"
    ET.SubElement(ov, _b("Key")).text = key

def _directed_link(pproc: ET.Element, idx: int, from_id: str, to_id: str, link_sfx: str):
    """Create DirectedLink with ID appended by millisecond suffix."""
    dl = ET.SubElement(pproc, _b("DirectedLink"))
    ET.SubElement(dl, _b("ID")).text = f"{idx}" + link_sfx
    ET.SubElement(dl, _b("FromID")).text = from_id
    ET.SubElement(dl, _b("ToID")).text = to_id

def apply_stylistic_blank_lines(xml_str: str) -> str:
    """Insert blank lines to mimic the reference layout."""
    lines = xml_str.splitlines()
    out: list[str] = []

    def ensure_blank_before():
        if out and out[-1].strip() != "":
            out.append("")

    def next_nonempty(idx: int) -> str | None:
        j = idx + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        return lines[j].strip() if j < len(lines) else None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped in ("<b2mml:ProcessInputs>", "<b2mml:ProcessOutputs>", "<b2mml:ProcessIntermediates>"):
            ensure_blank_before()

        out.append(line)

        if stripped.startswith("<b2mml:LifeCycleState>") and stripped.endswith("</b2mml:LifeCycleState>"):
            out.append("")

        if stripped == "<b2mml:MaterialsType>Input</b2mml:MaterialsType>":
            out.append("")

        if stripped == "</b2mml:Material>":
            nxt = next_nonempty(i)
            if nxt and nxt.startswith("<b2mml:Material>"):
                out.append("")

        if stripped == "</b2mml:Materials>":
            out.append("")

        if stripped in ("</b2mml:ProcessInputs>", "</b2mml:ProcessOutputs>", "</b2mml:ProcessIntermediates>"):
            out.append("")
        if stripped == "</b2mml:Formula>":
            out.append("")

        if stripped == "</b2mml:DirectedLink>":
            nxt = next_nonempty(i)
            if nxt and nxt.startswith("<b2mml:ProcessElement>"):
                out.append("")

        if stripped == "</b2mml:ProcessElementType>":
            out.append("")
        if stripped == "</b2mml:ProcessElementParameter>":
            out.append("")
        if stripped == "</b2mml:ProcessElement>":
            out.append("")
        if stripped == "</b2mml:ProcessProcedure>":
            out.append("")

        i += 1

    result = "\n".join(out)
    if not result.endswith("\n"):
        result += "\n"
    return result


def build_general_recipe_tree(schedule: List[Dict], order: Dict) -> ET.ElementTree:
    """
    Build B2MML General Recipe:
      - Dosing links to the FIRST following Mix.
      - For multiple mixes: Mix_j -> IntermediateAfterMix_j -> Mix_{j+1} (j = 1..N-1).
      - LAST Mix links to Product.
      - Product -> Usage -> IntermediateAfterUsage -> Collecting -> Settling -> Separation chain.
      - No Separation -> Product link.
    """
    # Suffixes used in IDs
    ts_sfx = _ts_suffix()     # element IDs
    pl_sfx = _ts_suffix()     # parameter & DirectedLink IDs

    # Flow summary for file/recipe ID
    grecipe_id = _grecipe_id_from_flow(schedule)

    # Ingredient totals & first-appearance order
    totals = _collect_totals(schedule)
    ingredient_order: List[str] = []
    seen = set()
    for e in schedule:
        if e["type"] == "dose":
            g = e["params"]["ingredient"]
            if g not in seen:
                seen.add(g)
                ingredient_order.append(g)

    total_mix_volume = _total_volume(schedule)

    # Mix indices (in schedule) and count
    mix_indices_in_schedule: List[int] = [i for i, e in enumerate(schedule) if e["type"] == "mix"]
    mix_count = len(mix_indices_in_schedule)

    # Intermediates between mixes: only for j=1..mix_count-1
    inter_mix_ids: List[str] = []
    for j in range(1, max(mix_count, 1)):  # if 0 or 1 mixes => no inter-mix intermediates
        inter_mix_ids.append(f"IntermediateAfterMix{j:03d}" + ts_sfx)

    # Fixed IDs
    product_id       = "Product001" + ts_sfx
    inter_usage_id   = "IntermediateAfterUsage001" + ts_sfx
    usage_id         = "Usage001" + ts_sfx
    #collecting_id    = "Collecting001" + ts_sfx
    settling_id      = "Settling001" + ts_sfx

    # Collecting config (Volume + Ratio text)
    #collecting_volume = float(order.get("collecting", {}).get("volume", order.get("volume", total_mix_volume)))
    ratio_raw = _sum_weights_ratio(order, ingredient_order)
    ratio_text = _fmt_ratio_text(ratio_raw)

    # ----- Root -----
    root = ET.Element(_b("GRecipe"), {
        f"{{{NS_XSI}}}schemaLocation": "http://www.mesa.org/xml/B2MML "
    })
    ET.SubElement(root, _b("ID")).text = f"GeneralRecipe_{grecipe_id}"
    ET.SubElement(root, _b("Description")).text = "General Recipe auto-generated from schedule"
    ET.SubElement(root, _b("GRecipeType")).text = "General"
    ET.SubElement(root, _b("LifeCycleState")).text = "Draft"

    # ----- Formula -----
    formula = ET.SubElement(root, _b("Formula"))
    ET.SubElement(formula, _b("Description")).text = "The formula defines the Inputs, Intermediates and Outputs of the Procedure"

    # ProcessInputs
    pin = ET.SubElement(formula, _b("ProcessInputs"))
    ET.SubElement(pin, _b("ID")).text = "InputListID001" + ts_sfx
    ET.SubElement(pin, _b("Description")).text = "List of Process Inputs"
    ET.SubElement(pin, _b("MaterialsType")).text = "Input"

    educt_ids: List[str] = []
    for idx_i, ingr in enumerate(ingredient_order, start=1):
        mid = (f"Educt_{ingr}{idx_i:03d}") + ts_sfx
        educt_ids.append(mid)
        mat = ET.SubElement(pin, _b("Material"))
        ET.SubElement(mat, _b("ID")).text = mid
        ET.SubElement(mat, _b("Description")).text = f"Ingredient {ingr}"
        matid = ET.SubElement(mat, _b("MaterialID"), {"schemeAgencyName": "GESTIS", "schemeName": "CAS"})
        matid.text = "7732-18-5"
        ET.SubElement(mat, _b("Order")).text = str(idx_i)
        _amount_block(mat, totals[ingr])

    # ProcessOutputs (Product is produced after last Mix)
    pout = ET.SubElement(formula, _b("ProcessOutputs"))
    ET.SubElement(pout, _b("ID")).text = "OutputListID001" + ts_sfx
    ET.SubElement(pout, _b("Description")).text = "List of Process Outputs"
    ET.SubElement(pout, _b("MaterialsType")).text = "Output"
    m_out = ET.SubElement(pout, _b("Material"))
    ET.SubElement(m_out, _b("ID")).text = product_id
    ET.SubElement(m_out, _b("Description")).text = "Mixed product"
    mo_id = ET.SubElement(m_out, _b("MaterialID"), {"schemeAgencyName": "GESTIS", "schemeName": "CAS"})
    mo_id.text = "7732-18-5"
    ET.SubElement(m_out, _b("Order")).text = "1"
    _amount_block(m_out, total_mix_volume)

    # ProcessIntermediates = {InterMix_1..InterMix_{N-1}} + {IntermediateAfterUsage001}
    pinter = ET.SubElement(formula, _b("ProcessIntermediates"))
    ET.SubElement(pinter, _b("ID")).text = "IntermediateListID001" + ts_sfx
    ET.SubElement(pinter, _b("Description")).text = "List of Process Intermediates"
    ET.SubElement(pinter, _b("MaterialsType")).text = "Intermediate"

    for j, im_id in enumerate(inter_mix_ids, start=1):
        m_ij = ET.SubElement(pinter, _b("Material"))
        ET.SubElement(m_ij, _b("ID")).text = im_id
        ET.SubElement(m_ij, _b("Description")).text = f"Mixture after mixing #{j}"
        mi_id = ET.SubElement(m_ij, _b("MaterialID"), {"schemeAgencyName": "GESTIS", "schemeName": "CAS"})
        mi_id.text = "7732-18-5"
        ET.SubElement(m_ij, _b("Order")).text = str(j)
        _amount_block(m_ij, total_mix_volume)

    m_i_usage = ET.SubElement(pinter, _b("Material"))
    ET.SubElement(m_i_usage, _b("ID")).text = inter_usage_id
    ET.SubElement(m_i_usage, _b("Description")).text = "Mixture after usage"
    miu_id = ET.SubElement(m_i_usage, _b("MaterialID"), {"schemeAgencyName": "GESTIS", "schemeName": "CAS"})
    miu_id.text = "7732-18-5"
    ET.SubElement(m_i_usage, _b("Order")).text = str(len(inter_mix_ids) + 1)
    _amount_block(m_i_usage, total_mix_volume)

    # ----- ProcessProcedure -----
    pproc = ET.SubElement(root, _b("ProcessProcedure"))
    ET.SubElement(pproc, _b("ID")).text = "ProcessProcedureID001" + ts_sfx
    ET.SubElement(pproc, _b("Description")).text = "Top level ProcessElement"
    ET.SubElement(pproc, _b("ProcessElementType")).text = "Process"
    ET.SubElement(pproc, _b("LifeCycleState")).text = "Draft"

    # Materials lists
    _materials_block(pproc, "ProcedureInputMaterials" + ts_sfx, "Input Materials of Procedure", "Input", educt_ids)
    _materials_block(pproc, "ProcedureIntermediateMaterials" + ts_sfx, "Intermediate Materials of Procedure", "Intermediate", inter_mix_ids + [inter_usage_id])
    _materials_block(pproc, "ProcedureOutputMaterials" + ts_sfx, "Output Materials of Procedure", "Output", [product_id])

    # ----- ProcessElement IDs -----
    dosing_ids: List[str] = [(f"Dosing_{g}{i:03d}") + ts_sfx for i, g in enumerate(ingredient_order, start=1)]
    mix_ids: List[str] = []
    mix_counter = 0
    for e in schedule:
        if e["type"] == "mix":
            mix_counter += 1
            mix_ids.append((f"Mixing_of_Liquids{mix_counter:03d}") + ts_sfx)

    # Separation order
    if "separation_order" in order and isinstance(order["separation_order"], list):
        sep_order = [str(x).upper() for x in order["separation_order"] if x in totals]
    else:
        sep_order, sset = [], set()
        for e in reversed(schedule):
            if e["type"] == "dose":
                g = e["params"]["ingredient"]
                if g not in sset:
                    sset.add(g)
                    sep_order.append(g)
    sep_ids = [(f"Separation_{g}{i+1:03d}") + ts_sfx for i, g in enumerate(sep_order)]

    # ----- Directed Links -----
    link_idx = 0

    # Inputs -> Dosing (one-to-one)
    for mid, did in zip(educt_ids, dosing_ids):
        _directed_link(pproc, link_idx, mid, did, pl_sfx); link_idx += 1

    if mix_count == 0:
        # No mix: link Dosing -> Product directly (fallback)
        for did in dosing_ids:
            _directed_link(pproc, link_idx, did, product_id, pl_sfx); link_idx += 1
    else:
        # Map Dosing to the FIRST following mix in schedule
        first_dose_idx: Dict[str, int] = {}
        for idx, e in enumerate(schedule):
            if e["type"] == "dose":
                g = e["params"]["ingredient"]
                if g not in first_dose_idx:
                    first_dose_idx[g] = idx

        dose_to_mix: Dict[str, int] = {}  # ingredient -> mix_number (1-based)
        for ingr in ingredient_order:
            di = first_dose_idx.get(ingr, -1)
            target_mix_num = None
            for j, mix_idx in enumerate(mix_indices_in_schedule, start=1):
                if mix_idx > di:
                    target_mix_num = j
                    break
            if target_mix_num is None:
                target_mix_num = mix_count
            dose_to_mix[ingr] = target_mix_num

        # Dosing -> target Mix
        for ingr, did in zip(ingredient_order, dosing_ids):
            j = dose_to_mix[ingr]
            _directed_link(pproc, link_idx, did, mix_ids[j-1], pl_sfx); link_idx += 1

        # Bridge mixes with intermix intermediates (only between mixes)
        if mix_count >= 2:
            # Mix_1 -> InterMix_1
            _directed_link(pproc, link_idx, mix_ids[0], inter_mix_ids[0], pl_sfx); link_idx += 1
            # For j = 2 .. mix_count-1:
            for j in range(2, mix_count):
                _directed_link(pproc, link_idx, inter_mix_ids[j-2], mix_ids[j-1], pl_sfx); link_idx += 1
                _directed_link(pproc, link_idx, mix_ids[j-1], inter_mix_ids[j-1], pl_sfx); link_idx += 1
            # After the (mix_count-1)-th intermix, connect to last mix
            _directed_link(pproc, link_idx, inter_mix_ids[-1], mix_ids[-1], pl_sfx); link_idx += 1

        # LAST Mix -> Product
        _directed_link(pproc, link_idx, mix_ids[-1], product_id, pl_sfx); link_idx += 1

    # Product -> Usage -> IntermediateAfterUsage -> Collecting -> Settling
    _directed_link(pproc, link_idx, product_id, usage_id, pl_sfx); link_idx += 1
    _directed_link(pproc, link_idx, usage_id, inter_usage_id, pl_sfx); link_idx += 1
    # _directed_link(pproc, link_idx, inter_usage_id, collecting_id, pl_sfx); link_idx += 1
    # _directed_link(pproc, link_idx, collecting_id, settling_id, pl_sfx); link_idx += 1
    _directed_link(pproc, link_idx, inter_usage_id, settling_id, pl_sfx); link_idx += 1


    # Settling -> separations (NO link back to Product)
    prev = settling_id
    for sid in sep_ids:
        _directed_link(pproc, link_idx, prev, sid, pl_sfx); link_idx += 1
        prev = sid
    # (End of chain here; no Product link)

    # ----- Process Elements -----
    # Dosing elements (with Amount parameter)
    for i, ingr in enumerate(ingredient_order, start=1):
        pe = ET.SubElement(pproc, _b("ProcessElement"))
        pe_id = (f"Dosing_{ingr}{i:03d}") + ts_sfx
        ET.SubElement(pe, _b("ID")).text = pe_id
        ET.SubElement(pe, _b("Description")).text = f"Dosing {ingr}"
        ET.SubElement(pe, _b("ProcessElementType")).text = "Process"

        _materials_block(pe, f"{pe_id}InputMaterials", f"Input Materials of {pe_id}", "Input", [])
        _materials_block(pe, f"{pe_id}IntermediateMaterials", f"Intermediate Materials of {pe_id}", "Intermediate", [])
        _materials_block(pe, f"{pe_id}OutputMaterials", f"Output Materials of {pe_id}", "Output", [])

        p = ET.SubElement(pe, _b("ProcessElementParameter"))
        ET.SubElement(p, _b("ID")).text = f"Dosing_Amount{i:03d}" + pl_sfx
        ET.SubElement(p, _b("Description")).text = "Amount of Dosing"
        v = ET.SubElement(p, _b("Value"))
        ET.SubElement(v, _b("ValueString")).text = f"{totals[ingr]:.6g}"
        ET.SubElement(v, _b("DataType")).text = "double"
        ET.SubElement(v, _b("UnitOfMeasure")).text = "http://si-digital-framework.org/SI/units/litre"
        ET.SubElement(v, _b("Key")).text = "http://qudt.org/vocab/quantitykind/LiquidVolume"

        _other_info(pe, "Capability_with_Query.Dosing", "http://www.iat.rwth-aachen.de/capability-ontology#Dosing")

    # Mixing elements (RPM + duration)
    mix_idx = 0
    for e in schedule:
        if e["type"] != "mix":
            continue
        mix_idx += 1
        pe = ET.SubElement(pproc, _b("ProcessElement"))
        pe_id = (f"Mixing_of_Liquids{mix_idx:03d}") + ts_sfx
        ET.SubElement(pe, _b("ID")).text = pe_id
        ET.SubElement(pe, _b("Description")).text = "Mixing_of_Liquids"
        ET.SubElement(pe, _b("ProcessElementType")).text = "Process"

        _materials_block(pe, f"{pe_id}InputMaterials", f"Input Materials of {pe_id}", "Input", [])
        _materials_block(pe, f"{pe_id}IntermediateMaterials", f"Intermediate Materials of {pe_id}", "Intermediate", [])
        _materials_block(pe, f"{pe_id}OutputMaterials", f"Output Materials of {pe_id}", "Output", [])

        _rpm_param(pe, int(e["params"]["rpm"]), mix_idx, pl_sfx)
        _duration_param(pe, f"Mixing_Duration{mix_idx:03d}", "Duration of the process step mixing", int(e["duration_s"]), pl_sfx)
        _other_info(pe, "Capability_with_Query.Mixing_of_Liquids", "http://www.iat.rwth-aachen.de/capability-ontology#MixingOfLiquids")

    # Usage element
    usage_seconds = next((e["duration_s"] for e in schedule if e["type"] == "usage"), 0)
    pe = ET.SubElement(pproc, _b("ProcessElement"))
    usage_pe_id = usage_id
    ET.SubElement(pe, _b("ID")).text = usage_pe_id
    ET.SubElement(pe, _b("Description")).text = "Usage"
    ET.SubElement(pe, _b("ProcessElementType")).text = "Process"
    _materials_block(pe, f"{usage_pe_id}InputMaterials", f"Input Materials of {usage_pe_id}", "Input", [])
    _materials_block(pe, f"{usage_pe_id}IntermediateMaterials", f"Intermediate Materials of {usage_pe_id}", "Intermediate", [])
    _materials_block(pe, f"{usage_pe_id}OutputMaterials", f"Output Materials of {usage_pe_id}", "Output", [])
    _duration_param(pe, "Usage_Duration001", "Duration of the process step usage", int(usage_seconds), pl_sfx)
    _other_info(pe, "Capability_with_Query.Usage", "http://www.iat.rwth-aachen.de/capability-ontology#Usage")

    # # Collecting element
    # pe = ET.SubElement(pproc, _b("ProcessElement"))
    # collecting_pe_id = collecting_id
    # ET.SubElement(pe, _b("ID")).text = collecting_pe_id
    # ET.SubElement(pe, _b("Description")).text = "Collecting"
    # ET.SubElement(pe, _b("ProcessElementType")).text = "Process"
    # _materials_block(pe, f"{collecting_pe_id}InputMaterials", f"Input Materials of {collecting_pe_id}", "Input", [])
    # _materials_block(pe, f"{collecting_pe_id}IntermediateMaterials", f"Intermediate Materials of {collecting_pe_id}", "Intermediate", [])
    # _materials_block(pe, f"{collecting_pe_id}OutputMaterials", f"Output Materials of {collecting_pe_id}", "Output", [])
    # # Volume param
    # p = ET.SubElement(pe, _b("ProcessElementParameter"))
    # ET.SubElement(p, _b("ID")).text = "Collecting_Volume001" + pl_sfx
    # ET.SubElement(p, _b("Description")).text = "Volume to collect"
    # v = ET.SubElement(p, _b("Value"))
    # ET.SubElement(v, _b("ValueString")).text = f"{collecting_volume:.6g}"
    # ET.SubElement(v, _b("DataType")).text = "double"
    # ET.SubElement(v, _b("UnitOfMeasure")).text = "http://si-digital-framework.org/SI/units/litre"
    # ET.SubElement(v, _b("Key")).text = "http://qudt.org/vocab/quantitykind/LiquidVolume"
    # # Ratio param (string; UnitOfMeasure "Ratio")
    # _string_param(
    #     pe,
    #     "Collecting_Ratio001",
    #     "Ingredient ratio (sum of weights)",
    #     ratio_text,
    #     param_sfx=pl_sfx,
    #     unit_of_measure="Ratio",
    #     key="http://qudt.org/vocab/quantitykind/Dimensionless",
    # )
    # _other_info(pe, "Capability_with_Query.Collecting", "http://www.iat.rwth-aachen.de/capability-ontology#Collecting")

    # Settling element
    settling_seconds = next((e["duration_s"] for e in schedule if e["type"] == "settling"), 0)
    pe = ET.SubElement(pproc, _b("ProcessElement"))
    settling_pe_id = settling_id
    ET.SubElement(pe, _b("ID")).text = settling_pe_id
    ET.SubElement(pe, _b("Description")).text = "Settling"
    ET.SubElement(pe, _b("ProcessElementType")).text = "Process"
    _materials_block(pe, f"{settling_pe_id}InputMaterials", f"Input Materials of {settling_pe_id}", "Input", [])
    _materials_block(pe, f"{settling_pe_id}IntermediateMaterials", f"Intermediate Materials of {settling_pe_id}", "Intermediate", [])
    _materials_block(pe, f"{settling_pe_id}OutputMaterials", f"Output Materials of {settling_pe_id}", "Output", [])
    _duration_param(pe, "Settling_Duration001", "Duration of the process step settling", int(settling_seconds), pl_sfx)
    _other_info(pe, "Capability_with_Query.Settling", "http://www.iat.rwth-aachen.de/capability-ontology#Settling")

    # Separation elements (no final link to Product)
    sep_map = {}
    for e in schedule:
        if e["type"] == "sep":
            ing = e["params"]["ingredient"]
            sep_map[ing] = totals.get(ing, 0.0)

    for i, ingr in enumerate(sep_order, start=1):
        pe = ET.SubElement(pproc, _b("ProcessElement"))
        pe_id = (f"Separation_{ingr}{i:03d}") + ts_sfx
        ET.SubElement(pe, _b("ID")).text = pe_id
        ET.SubElement(pe, _b("Description")).text = f"Separation of {ingr}"
        ET.SubElement(pe, _b("ProcessElementType")).text = "Process"
        _materials_block(pe, f"{pe_id}InputMaterials", f"Input Materials of {pe_id}", "Input", [])
        _materials_block(pe, f"{pe_id}IntermediateMaterials", f"Intermediate Materials of {pe_id}", "Intermediate", [])
        _materials_block(pe, f"{pe_id}OutputMaterials", f"Output Materials of {pe_id}", "Output", [])
        # Volume parameter
        p = ET.SubElement(pe, _b("ProcessElementParameter"))
        ET.SubElement(p, _b("ID")).text = f"Separation_Volume{i:03d}" + pl_sfx
        ET.SubElement(p, _b("Description")).text = "Volume to separate"
        v = ET.SubElement(p, _b("Value"))
        ET.SubElement(v, _b("ValueString")).text = f"{sep_map.get(ingr, 0.0):.6g}"
        ET.SubElement(v, _b("DataType")).text = "double"
        ET.SubElement(v, _b("UnitOfMeasure")).text = "http://si-digital-framework.org/SI/units/litre"
        ET.SubElement(v, _b("Key")).text = "http://qudt.org/vocab/quantitykind/LiquidVolume"
        # Duration parameter
        dur_s = next((int(e["duration_s"]) for e in schedule if e["type"] == "sep" and e["params"]["ingredient"] == ingr), 0)
        _duration_param(pe, f"Separation_Duration{i:03d}", "Duration of the process step separation", dur_s, pl_sfx)
        _other_info(pe, "Capability_with_Query.Separation_of_Liquids", "http://www.iat.rwth-aachen.de/capability-ontology#Separation")

    return ET.ElementTree(root)

def _inject_xml_model_pi(xml_str: str) -> str:
    """Insert xml-model PI after XML declaration."""
    pi = '<?xml-model href="Schema/AllSchemas.xsd" ?>\n'
    if xml_str.startswith('<?xml'):
        first_nl = xml_str.find('?>') + 2
        return xml_str[:first_nl] + "\n" + pi + xml_str[first_nl:]
    return pi + xml_str

def save_general_recipe_xml(
    schedule: List[Dict],
    order: Dict,
    out_dir: str | None = None,
) -> str:
    """
    Save pretty-printed General Recipe XML.

    Args:
        schedule: Generated process schedule.
        order: Source order definition used to build the General Recipe.
        out_dir: Optional override. If None, defaults to <ScriptDir>/Recipe/XML.
    """
    tree = build_general_recipe_tree(schedule, order)

    def _grecipe_id_from_flow_local(schedule: List[Dict]) -> str:
        tokens = []
        for e in schedule:
            t = e.get("type")
            if t == "dose":
                tokens.append(f"{e['params']['ingredient']}")
            elif t == "mix":
                tokens.append(f"Mix{e['params']['rpm']}")
            elif t == "usage":
                tokens.append("U")
            # elif t == "collecting":
            #     tokens.append("C")
            elif t == "settling":
                tokens.append("S")
            elif t == "sep":
                tokens.append(f"Sep{e['params']['ingredient']}")
        base = re.sub(r"[^A-Za-z0-9]+", "_", "-".join(tokens)).strip("_")[:60] or "Flow"
        return f"{base}"

    flow_id = _grecipe_id_from_flow_local(schedule)
    if out_dir:
        xml_output_dir = out_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_output_dir = os.path.join(script_dir, "Recipe", "XML")
    
    os.makedirs(xml_output_dir, exist_ok=True)
    
    
    fname = f"GRecipe_{flow_id}.xml"
    fpath = os.path.join(xml_output_dir, fname)

    # Pretty-print
    try:
        ET.indent(tree, space="  ", level=0)  # Python 3.9+
        xml_bytes = ET.tostring(
            tree.getroot(),
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=False
        )
        xml_str = xml_bytes.decode("utf-8")
    except Exception:
        import xml.dom.minidom as minidom
        rough_bytes = ET.tostring(
            tree.getroot(),
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=False
        )
        rough = rough_bytes.decode("utf-8")
        if rough.startswith("<?xml"):
            pos = rough.find("?>")
            inner = rough[pos+2:].lstrip("\n")
            decl = rough[:pos+2]
        else:
            inner = rough
            decl = "<?xml version='1.0' encoding='utf-8'?>"
        pretty = minidom.parseString(inner).toprettyxml(indent="  ")
        if pretty.startswith("<?xml"):
            pretty = "\n".join(pretty.splitlines()[1:])
        xml_str = decl + "\n" + pretty

    xml_str = _inject_xml_model_pi(xml_str)
    xml_str = apply_stylistic_blank_lines(xml_str)

    with open(fpath, "w", encoding="utf-8", newline="\n") as f:
        f.write(xml_str)

    print(f"[GeneralRecipe] Saved: {fpath}")
    return fpath
