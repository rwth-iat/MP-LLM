from __future__ import annotations

import os, sys, json, traceback, types
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
stub_mod = types.SimpleNamespace()
stub_mod.display_dataframe_to_user = lambda *args, **kwargs: None
sys.modules.setdefault('ace_tools_open', stub_mod)
class _ToolsStub: display_dataframe_to_user = staticmethod(lambda *a, **k: None)
tools = _ToolsStub()
import sys
import os
# Use the current Jupyter kernel's Python to install the packages
#!{sys.executable} -m pip install pyomo pandas ace_tools_open typing_extensions packaging pyvis gurobipy
current_dir = os.path.dirname(os.path.abspath('__file__')) 
sys.path.append(os.path.abspath('..'))

from pyomo.environ import *
from itertools import permutations
import pandas as pd
import ace_tools_open as tools;
from collections import deque
from typing import List, Tuple, Dict
import sys
import sysconfig

# Print full Python version information
print("Python version:", sys.version)

# Check compile-time flag: was this Python built with GIL disabled?
gil_flag = sysconfig.get_config_var("Py_GIL_DISABLED")
print("Py_GIL_DISABLED config var:", gil_flag)

# Check at runtime whether the GIL is currently enabled (Python 3.13+)
if hasattr(sys, "_is_gil_enabled"):
    print("GIL enabled at runtime:", sys._is_gil_enabled())
else:
    print("sys._is_gil_enabled() is not available on this Python version")
    

# ---- Original definitions (kept for reproducibility) ----
ModPlant_ops = {
    'HC10': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Settling', '', 1), ('Stirring', '100', 3), ('Stirring', '200', 3),('Connect', '', 1), ('Disconnect', '', 0),  ('None', '', 0)],
    'HC20': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Settling', '', 1), ('Stirring', '150', 3), ('Stirring', '300', 3), ('Connect', '', 1), ('Disconnect', '', 0), ('None', '', 0)],
    'HC30': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Settling', '', 1), ('Stirring', '100', 3), ('Stirring', '150', 3),('Connect', '', 1), ('Disconnect', '', 0),  ('None', '', 0)],
    'HC40': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Settling', '', 1), ('Connect', '', 1), ('Disconnect', '', 0), ('None', '', 0)]
}

ModPlant_interfaces = {
    'HC10': [('Input', 'HC10_In1'), ('Input', 'HC10_In2'), ('Input', 'HC10_In3'), ('Output', 'HC10_Out1'), ('Output', 'HC10_Out2'), ('Output', 'HC10_Out3')],
    'HC20': [('Input', 'HC20_In1'), ('Input', 'HC20_In2'), ('Input', 'HC20_In3'), ('Output', 'HC20_Out1'), ('Output', 'HC20_Out2'), ('Output', 'HC20_Out3')],
    'HC30': [('Input', 'HC30_In1'), ('Input', 'HC30_In2'), ('Input', 'HC30_In3'), ('Input', 'HC30_In4'), ('Output', 'HC30_Out1')],
    'HC40': [('Input', 'HC40_In1'), ('Output', 'HC40_Out1'), ('Output', 'HC40_Out2')],
}



ModPlant_maximum_volume = {
    'HC10': [10],
    'HC20': [15],
    'HC30': [10],
    'HC40': [30]
}

ModPlant_resources = {
    'HC10': ['A', 10],
    'HC20': ['B', 10],
    'HC30': ['C', 10],
}


import random
import os

# Toggle between original fixed ModPlants and random generation
use_original_ModPlants = False
# Default multiprocessing tuning for Python 3.11 BFS runs.
os.environ.setdefault("WABEN_BFS_WORKERS", "11")
os.environ.setdefault("WABEN_BFS_BATCH_SIZE", "1024")
env_seed = os.getenv('WABEN_SEED')
random_seed = int(env_seed) if env_seed is not None else random.randint(1, 10000)  # You can set this to a fixed value for reproducibility
min_ModPlant_count = 4
max_ModPlant_count = 4

# ---- Random generator under given constraints ----
rate_choices = [0.1, 0.2, 0.3]
rpm_choices = list(range(50, 301, 50))


def generate_random_ModPlants(seed: int, min_n: int, max_n: int):
    random.seed(seed)
    n = random.randint(max(4, min_n), min(6, max_n))
    names = [f"HC{(i+1)*10}" for i in range(n)]

    w_ops = {}
    w_ifaces = {}
    w_cap = {}

    # Ensure at least one Settling and one Stirring holder
    settling_assigned = False
    stirring_assigned = False

    for idx, name in enumerate(names):
        # interfaces
        num_inputs = random.randint(1, 4)
        num_outputs = random.randint(1, 4)
        w_ifaces[name] = [("Input", f"{name}_In{i+1}") for i in range(num_inputs)] +                           [("Output", f"{name}_Out{i+1}") for i in range(num_outputs)]

        # capacity
        max_vol = random.choice(range(10, 31, 5))
        w_cap[name] = [max_vol]

        # base ops
        drain_rate = random.choice(rate_choices)
        fill_rate = random.choice(rate_choices)
        ops = [
            ("Draining", drain_rate, 3),
            ("Filling", fill_rate, 0),
            ("Connect", "", 1),
            ("Disconnect", "", 0),
            ("None", "", 0),
        ]

        # Settling (ensure at least one)
        if not settling_assigned or random.random() < 0.5:
            ops.append(("Settling", "", random.randint(1, 3)))
            settling_assigned = True

        # Stirring 1-2 params (ensure at least one overall)
        stir_count = random.randint(1, 2)
        rpms = random.sample(rpm_choices, stir_count)
        for rpm in rpms:
            ops.append(("Stirring", str(rpm), random.randint(1, 4)))
        stirring_assigned = stirring_assigned or bool(rpms)

        w_ops[name] = ops

    # Guarantee constraints if randomness skipped them
    if not settling_assigned:
        name = names[0]
        w_ops[name].append(("Settling", "", random.randint(1, 3)))
    if not stirring_assigned:
        name = names[0]
        rpm = random.choice(rpm_choices)
        w_ops[name].append(("Stirring", str(rpm), random.randint(1, 4)))

    # resources: assign A,B,C to three distinct ModPlants
    resources = {}
    chosen = random.sample(names, k=min(3, len(names)))
    for mat, wb in zip(['A', 'B', 'C'], chosen):
        cap = w_cap[wb][0]
        qty = min(random.randint(5, 15), cap)
        resources[wb] = [mat, qty]

    return w_ops, w_ifaces, w_cap, resources

# ---- Choose config ----
if not use_original_ModPlants:
    ModPlant_ops, ModPlant_interfaces, ModPlant_maximum_volume, ModPlant_resources = generate_random_ModPlants(
        seed=random_seed,
        min_n=min_ModPlant_count,
        max_n=max_ModPlant_count,
    )

# === Waben summary table after generation ===
# Expanded per-operation rows for structured display
summary_rows = []
for wb in sorted(ModPlant_ops.keys()):
    ifaces = ModPlant_interfaces.get(wb, [])
    num_inputs = sum(1 for t, _ in ifaces if t == 'Input')
    num_outputs = sum(1 for t, _ in ifaces if t == 'Output')
    max_vols = ModPlant_maximum_volume.get(wb, [])
    max_vol = max_vols[0] if max_vols else None
    res = ModPlant_resources.get(wb)
    res_str = f"{res[0]}:{res[1]}" if res else ''

    ops = ModPlant_ops.get(wb, [])
    if not ops:
        summary_rows.append({
            'Waben': wb,
            'Inputs': num_inputs,
            'Outputs': num_outputs,
            'MaxVolume': max_vol,
            'Resources': res_str,
            'Operation Type': '',
            'Operation Param': '',
            'Operation Cost': '',
        })
    else:
        for op_type, op_param, op_cost in ops:
            summary_rows.append({
                'Waben': wb,
                'Inputs': num_inputs,
                'Outputs': num_outputs,
                'MaxVolume': max_vol,
                'Resources': res_str,
                'Operation Type': op_type,
                'Operation Param': op_param,
                'Operation Cost': op_cost,
            })

try:
    import pandas as pd
    df_ModPlant_summary = pd.DataFrame(summary_rows)
    # Display in notebook UI if available
    tools.display_dataframe_to_user(name='Waben Summary (Expanded)', dataframe=df_ModPlant_summary)
    # Also print ASCII table for batch_runner visibility
    print('Waben Summary (Expanded):')
    try:
        print(df_ModPlant_summary.to_string(index=False))
    except Exception:
        for row in summary_rows:
            print(row)
    # cleanup dataframe to reduce per-seed memory
    try:
        import gc
        del df_ModPlant_summary
        gc.collect()
    except Exception:
        pass
except Exception:
    print('Waben Summary (Expanded):')
    for row in summary_rows:
        print(row)




# # Create DataFrame from ModPlant_ops
df_ops = pd.DataFrame([
    {"Waben": wb, "Operation Type": op[0], "Operation Param": op[1], "Cost": op[2]}
    for wb, op_list in ModPlant_ops.items()
    for op in op_list
])

# Display DataFrame
tools.display_dataframe_to_user(name="Waben Operations", dataframe=df_ops)

# cleanup dataframe to reduce per-seed memory
try:
    import gc
    del df_ops
    gc.collect()
except Exception:
    pass


# # Input data: (HC_name, num_inputs, num_outputs)


# #hc_data = [('HC10', 5, 5), ('HC20', 5, 5), ('HC30', 5, 5), ('HC40', 5, 5)]


# # Generate ModPlant_interfaces dictionary
# hc_data = [('HC10', 3, 3), ('HC20', 3, 3), ('HC30', 4, 1), ('HC40', 1, 2)]
# ModPlant_interfaces = {}
# for hc_name, num_inputs, num_outputs in hc_data:
#     interfaces = []
#     # Add input interfaces
#     for i in range(1, num_inputs + 1):
#         interfaces.append(('Input', f'{hc_name}_In{i}'))
#     # Add output interfaces
#     for i in range(1, num_outputs + 1):
#         interfaces.append(('Output', f'{hc_name}_Out{i}'))
#     ModPlant_interfaces[hc_name] = interfaces

# # # Create DataFrame from ModPlant_interfaces
df_interfaces = pd.DataFrame([
    {"Waben": wb, "Interface Type": iface[0], "Interface Name": iface[1]}
    for wb, iface_list in ModPlant_interfaces.items()
    for iface in iface_list
])

# Display DataFrame
tools.display_dataframe_to_user(name="Waben Interfaces", dataframe=df_interfaces)

# cleanup dataframe to reduce per-seed memory
try:
    import gc
    del df_interfaces
    gc.collect()
except Exception:
    pass


df_maximum_volume = pd.DataFrame([
    {"Waben": wb, "Maximum Volume": vol}
    for wb, vols in ModPlant_maximum_volume.items()
    for vol in vols
])

tools.display_dataframe_to_user(name="Waben Maximum Volumes", dataframe=df_maximum_volume)


# cleanup dataframe to reduce per-seed memory
try:
    import gc
    del df_maximum_volume
    gc.collect()
except Exception:
    pass


df_resources = pd.DataFrame([
    {"Waben": wb, "Material": res[0], "Quantity": res[1]}
    for wb, res in ModPlant_resources.items()
])

tools.display_dataframe_to_user(name="Waben Resources", dataframe=df_resources)


# cleanup dataframe to reduce per-seed memory
try:
    import gc
    del df_resources
    gc.collect()
except Exception:
    pass


def res_to_tuple(v):
    """Convert resource list to sorted tuple of strings"""
    if isinstance(v, dict):
        return (f"{v.get('material','')} : {float(v.get('quantity',0)):.1f} {v.get('unit','litre')}",)
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
        return tuple(sorted(
            f"{d.get('material','')} : {float(d.get('quantity',0)):.1f} {d.get('unit','litre')}"
            for d in v
        ))
    if isinstance(v, (list, tuple)) and len(v) >= 2 and isinstance(v[0], str):
        return (f"{v[0]} : {float(v[1]):.1f} litre",)
    return ()

def build_initial_state(ModPlant_resources, ModPlant_interfaces):
    state = tuple(sorted(
        (
            wb,
            ("content", res_to_tuple(ModPlant_resources.get(wb, []))),
            ("outputs", tuple(sorted(
                (name, "", "") for t, name in ifaces if t == "Output"  # Modified: from (name, "") to (name, "", "")
            )))
        )
        for wb, ifaces in ModPlant_interfaces.items()
    ))
    return state

def display_state(state_tuple):
    """Display two DataFrames from the immutable state tuple."""

    df_state_contents = pd.DataFrame([
        {
            "Waben": wb,
            "Content": ", ".join(sorted(content_val))
        }
        for wb, (content_key, content_val), _ in state_tuple
    ])

    df_state_ports = pd.DataFrame([
        {
            "Waben": wb,
            "Port Type": "Output",
            "Port Name": port_name,
            "Connected To": target or "",
            "Material": material or ""  # Added Material column
        }
        for wb, _, (out_key, out_ports) in state_tuple
        for port_name, target, material in out_ports  # Modified: from port_name, target to port_name, target, material
    ])

    tools.display_dataframe_to_user(name="Initial Waben Contents", dataframe=df_state_contents)
    tools.display_dataframe_to_user(name="Initial Waben Ports (Connections)", dataframe=df_state_ports)

start_state = build_initial_state(ModPlant_resources, ModPlant_interfaces)
display_state(start_state)

def build_end_state_from_start(start_state_tuple):
    """Create an end state where all Waben have 'End' and outputs are empty but identical."""
    end_state = tuple(sorted(
        (
            wb,
            ("content", ("End",)),
            ("outputs", tuple(sorted((port_name, "", "") for port_name, _, _ in outputs)))  # Modified here
        )
        for wb, (_, _), (_, outputs) in start_state_tuple
    ))
    return end_state

end_state = build_end_state_from_start(start_state)
display_state(end_state)

from FSA.Waben_Flow_Generator import build_schedule
# from FSA.Waben_Render_Tools import print_flowchart
from FSA.Waben_Flow_To_General_Recipe import save_general_recipe_xml
import sys
import io
import random

# Force UTF-8 output encoding
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Toggle to use the original fixed recipe
use_original_order = use_original_ModPlants

# Original fixed recipe
order = {
  "volume": 6.0,
  "order": [
    "A",
    "B",
    "C",
    {"mix": {"rpm": 150, "duration": 30}}
  ],
  "ratio": {
    "A": [1],
    "B": [2],
    "C": [3]
  },
  "usage_and_settling": [3600, 300]
}



if not use_original_order:
    # Randomized order builder controlled by random_seed
    random.seed(random_seed)
    # volume: fixed 10L for randomized orders
    order_volume = 10

    # letters A,B,C in random order
    letters = ['A', 'B', 'C']
    random.shuffle(letters)

    # helper to create a mix step
    rpm_choices = list(range(50, 301, 50))
    def make_mix():
        return {"mix": {"rpm": random.choice(rpm_choices), "duration": random.randrange(100, 1001, 100)}}

    # Build order list with at most one mix (placed at the end)
    order_list = letters.copy()
    # Append a single mix step to keep behavior consistent with original flow
    order_list.append(make_mix())

    # ratio integers summing to 10
    parts = [random.randint(1, 8) for _ in letters]
    scale = 10 / sum(parts)
    ratio_vals = [max(1, round(v * scale)) for v in parts]
    adj = 10 - sum(ratio_vals)
    ratio_vals[0] += adj  # fix sum exactly to 10
    ratio = {k: [v] for k, v in zip(letters, ratio_vals)}

    # usage and settling durations: multiples of 100
    usage_and_settling = [random.randrange(100, 5001, 100), random.randrange(100, 1001, 100)]

    order = {
      "volume": float(order_volume),
      "order": order_list,
      "ratio": ratio,
      "usage_and_settling": usage_and_settling
    }

# Profit factor for the order (a * volume , b)
# From the time the order is placed, every second of delay in delivery will result in a loss of b.
order_profit_factor = (50 * order["volume"], -0.5)

# Generate the schedule from the order
print("Generating schedule from order.")
schedule = build_schedule(order)


from typing import List, Dict

# === Helper functions for rendering ===

def fmt_time(seconds: float) -> str:
    """Helper to format seconds into a readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds // 60:.0f}m {seconds % 60:.1f}s"

def _fmt_ratio_value(val) -> str:
    """Helper to format ratio values (handles lists or single numbers)."""
    if isinstance(val, list):
        return str(val[0])
    return str(val)

def make_box(title: str, lines: List[str]) -> List[str]:
    """Build a single ASCII box with a title + lines."""
    content = [title] + lines
    # Calculate width based on the longest line
    width = max((len(x) for x in content), default=0) + 2
    top = "┌" + "─" * width + "┐"
    bot = "└" + "─" * width + "┘"
    body = ["│ " + x.ljust(width - 2) + " │" for x in content]
    return [top] + body + [bot]

def _render_box_from_entry(entry: Dict) -> List[str]:
    """Create a box on the fly from one schedule entry."""
    t = entry["type"]
    title = entry["stage"]
    p = entry.get("params", {})
    dur = float(entry.get("duration_s", 0))

    lines = []
    if t == "dose":
        lines = [f"Portion: {p.get('portion_L', 0.0):.3f} L"]
    elif t == "mix":
        lines = [f"RPM: {p.get('rpm', 0)}", f"Duration: {fmt_time(dur)}"]
    elif t == "usage":
        lines = [f"Duration: {fmt_time(dur)}"]
    elif t == "collecting":
        lines = [f"Volume: {p.get('volume_L', 0.0):.3f} L"]
    elif t == "settling":
        lines = [f"Duration: {fmt_time(dur)}"]
    elif t == "sep":
        lines = [f"Volume: {p.get('volume_L', 0.0):.3f} L"]
    
    return make_box(title, lines)

def get_flowchart_text(schedule: List[Dict]) -> str:
    """
    Render a vertical ASCII flowchart and RETURN it as a string.
    This avoids output stream issues in multiprocess environments.
    """
    out: List[str] = []
    for i, e in enumerate(schedule):
        out.extend(_render_box_from_entry(e))
        if i < len(schedule) - 1:
            # Add a centered arrow
            out.append(" " * 4 + "↓")
    return "\n".join(out)

# === Usage ===

print("\n=== ASCII FLOWCHART ===\n")
# Capture the text first
flowchart_output = get_flowchart_text(schedule)
# Print it in one go to ensure the block stays together in the logs
print(flowchart_output, flush=True)

# Convert schedule to GeneralRecipe XML

from FSA.Waben_General_Recipe_To_Json import save_parsed_recipe_json_by_id
from FSA.Waben_Reaction_Rules import generate_reaction_rules_from_general_recipe_json, rules_to_dataframe

# # Use input() to ask for file path manually
# xml_path = input("Please enter the path to the General Recipe XML file: ").strip()
# if not os.path.exists(xml_path):
#     raise FileNotFoundError(f"File not found: {xml_path}")

persist_recipe_files = os.getenv("WABEN_PERSIST_RECIPE_FILES", "1").strip() == "1"
if persist_recipe_files:
    xml_path = save_general_recipe_xml(schedule, order)
    json_path = save_parsed_recipe_json_by_id(xml_path)
    reaction_rules = generate_reaction_rules_from_general_recipe_json(json_path)
else:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ModPlant_worker_recipe_") as tmp_dir:
        xml_path = save_general_recipe_xml(schedule, order, out_dir=tmp_dir)
        json_path = save_parsed_recipe_json_by_id(xml_path, out_dir=tmp_dir)
        reaction_rules = generate_reaction_rules_from_general_recipe_json(json_path)
    xml_path = "(disabled: temporary only)"
    json_path = "(disabled: temporary only)"
    print("Skipped persistent recipe XML/JSON export because WABEN_PERSIST_RECIPE_FILES=0")

reaction_rules_df = rules_to_dataframe(reaction_rules)

tools.display_dataframe_to_user(name="Semantic Reaction Rules", dataframe=reaction_rules_df)

from typing import List, Tuple, Set, Dict, Union, Optional
from collections import deque
import concurrent.futures
import multiprocessing as mp
import pandas as pd
# ========== Typedefs ==========

WabenName = str
MaterialStr = str
ContentTuple = Tuple[str, Tuple[MaterialStr, ...]]
# outputs: (key, ((port_name, connected_to, material), ...))
OutputsTuple = Tuple[str, Tuple[Tuple[str, str, str], ...]]
WabenStateTuple = Tuple[WabenName, ContentTuple, OutputsTuple]
FullState = Tuple[WabenStateTuple, ...]
# Transition in the final graph (with IDs)
Transition = Tuple[int, FullState, str, int, FullState, float, float]

# Pure-function stage candidate transition: no IDs, only from_state to to_state
CandidateTransition = Tuple[
    FullState,  # from_state
    FullState,  # to_state
    str,        # operation string
    float,      # cost
    float,      # duration
]

SymbolicVars = Tuple[Tuple[str, float, float], ...]

# Global objects expected to exist outside this module:
# ModPlant_ops, reaction_rules_df, ModPlant_maximum_volume, end_state

_BFS_WORKER_WABEN_INTERFACES = None
_BFS_WORKER_WABEN_OPS = None
_BFS_WORKER_REACTION_RULES_DF = None


def _init_bfs_worker(
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
    reaction_rules_df: pd.DataFrame,
    ModPlant_maximum_volume: Dict[str, List[int]],
    end_state_value: FullState,
) -> None:
    global _BFS_WORKER_WABEN_INTERFACES, _BFS_WORKER_WABEN_OPS, _BFS_WORKER_REACTION_RULES_DF
    _BFS_WORKER_WABEN_INTERFACES = ModPlant_interfaces
    _BFS_WORKER_WABEN_OPS = ModPlant_ops
    _BFS_WORKER_REACTION_RULES_DF = reaction_rules_df
    globals()["ModPlant_ops"] = ModPlant_ops
    globals()["reaction_rules_df"] = reaction_rules_df
    globals()["ModPlant_maximum_volume"] = ModPlant_maximum_volume
    globals()["end_state"] = end_state_value


def _expand_state_worker(state: FullState) -> List[CandidateTransition]:
    if (
        _BFS_WORKER_WABEN_INTERFACES is None
        or _BFS_WORKER_WABEN_OPS is None
        or _BFS_WORKER_REACTION_RULES_DF is None
    ):
        raise RuntimeError("BFS worker was not initialized with shared state")
    return expand_state_pure(
        state,
        _BFS_WORKER_WABEN_INTERFACES,
        _BFS_WORKER_WABEN_OPS,
        _BFS_WORKER_REACTION_RULES_DF,
    )


def _expand_state_batch_worker(states: List[FullState]) -> List[List[CandidateTransition]]:
    return [_expand_state_worker(state) for state in states]


def _get_bfs_mp_context():
    try:
        methods = mp.get_all_start_methods()
    except Exception:
        return None
    if sys.platform.startswith("linux") and "fork" in methods:
        return mp.get_context("fork")
    if sys.platform == "darwin" and "fork" in methods:
        # Fork keeps notebook-defined top-level functions importable on Unix.
        return mp.get_context("fork")
    return None

# Helper to get vars from state
def get_symbolic_vars(state: FullState) -> List[Tuple[str, float, float]]:
    if state and state[-1][0] == "__VARS__":
        # stored in content: ("vars", ("x:0:10", ...))
        content = state[-1][1][1]
        vars_list = []
        for s in content:
            # format "x:min:max"
            parts = s.split(":")
            vars_list.append((parts[0], float(parts[1]), float(parts[2])))
        return vars_list
    return []

def remove_symbolic_vars(state: FullState) -> FullState:
    if state and state[-1][0] == "__VARS__":
        return state[:-1]
    return state

# Helper to parse symbolic material string
# Format: "Material : 10.0 + -1.0 x litre" OR "Material : 10.0 litre"
def parse_material_string_symbolic(material_str: str) -> Dict[str, Any]:
    """
    Returns dict: { 'Material': {'base': 10.0, 'vars': {'x': -1.0}} }
    """
    result = {}
    entries = material_str.split(',')
    for entry in entries:
        if ':' not in entry: continue
        name, qty_part = entry.split(':', 1)
        name = name.strip()
        qty_part = qty_part.strip()
        
        # Check for symbolic marker "x" (simplification)
        # Regex to parse "10.0 + -1.0 x litre" or "10.0 litre"
        # We assume format: "{base} + {coeff} {var} {unit}" or "{base} {unit}"
        
        base_val = 0.0
        vars_dict = {}
        
        # Simple parsing logic
        tokens = qty_part.split() # ['10.0', '+', '-1.0', 'x', 'litre'] or ['10.0', 'litre']
        
        try:
            if len(tokens) >= 5 and tokens[1] == '+' and tokens[3] in ['x', 'y', 'z']:
                base_val = float(tokens[0])
                coeff = float(tokens[2])
                var_name = tokens[3]
                vars_dict[var_name] = coeff
            else:
                base_val = float(tokens[0])
        except:
            continue
            
        result[name] = {'base': base_val, 'vars': vars_dict}
    return result

def get_symbolic_coeffs(state: FullState) -> Tuple[float, float]:
    """Extract the cost and duration coefficients (multipliers of x) from a symbolic state."""
    cost_c = 0.0
    dur_c = 0.0
    if state and state[-1][0] == "__VARS__":
        # state[-1] structure: ('__VARS__', ('vars', ...), ('coeffs', ('cost:6.0', 'dur:1.0')))
        try:
            # Find the tuple named 'coeffs'
            coeffs_tuple = None
            for item in state[-1]:
                if isinstance(item, tuple) and item[0] == "coeffs":
                    coeffs_tuple = item[1]
                    break
            
            if coeffs_tuple:
                for s in coeffs_tuple:
                    if s.startswith("cost:"):
                        cost_c = float(s.split(":")[1])
                    elif s.startswith("dur:"):
                        dur_c = float(s.split(":")[1])
        except:
            pass
    return cost_c, dur_c

# ========== General Utility Functions ==========
import math
from typing import Dict

def get_operation_cost(
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
    wb_name: str,
    operation: str
) -> int:
    """Extract cost (3rd value) of a given operation for a specific Waben."""
    for op_name, _, cost in ModPlant_ops.get(wb_name, []):
        if op_name == operation:
            return cost
    return 0


def parse_material_string(material_str: str) -> Dict[str, float]:
    """Parse 'A: 1.0 litre' -> {'A': 1.0}. Returns empty if symbolic."""
    # if "x" in material_str: return {} # Skip symbolic for standard parsing
    # ... (Original code) ...
    result: Dict[str, float] = {}
    entries = material_str.split(',')
    for entry in entries:
        if ':' not in entry: continue
        name, qty = entry.split(':', 1)
        name = name.strip()
        try:
            amount = float(qty.strip().split()[0])
            result[name] = amount
        except ValueError:
            continue
    return result


def get_material_dict(contents: Tuple[str, ...]) -> Dict[str, float]:
    """Convert tuple of material strings to a material->amount dict."""
    return parse_material_string(', '.join(contents))


def parse_total_volume(content: Tuple[str, Tuple[str, ...]]) -> float:
    """Sum up all numeric quantities in a material content tuple."""
    total = 0.0
    for entry in content[1]:
        try:
            qty_str = entry.split(":")[1].strip().split(" ")[0]
            total += float(qty_str)
        except Exception:
            continue
    return total

def are_materials_equal(d1: Dict[str, float], d2: Dict[str, float], tol=1e-4) -> bool:
    """Performs a fuzzy match between two material dictionaries."""
    if set(d1.keys()) != set(d2.keys()):
        return False
        
    for k, v in d1.items():
        # If the target value is 0 (e.g., volume-less definition from 'Usage'), 
        # ignore numerical comparison; matching the name is sufficient.
        if abs(d2[k]) < 1e-9: 
            continue
            
        if not math.isclose(v, d2[k], abs_tol=tol):
            return False
            
    return True

# ========== Connect Helper Functions ==========

def is_input_free(state: FullState, target_wb: str, target_port: str) -> bool:
    """Check if a given input port on target_wb is free (not referenced by any output)."""
    for _, _, (_, outputs) in state:
        for _, connected_to, _ in outputs:
            if connected_to == target_port:
                return False
    return True


def get_free_outputs(state: FullState, wb_name: str) -> List[str]:
    """Return list of free (unconnected) output ports of a Waben."""
    for wb, _, (_, outputs) in state:
        if wb == wb_name:
            return [port_name for port_name, connected, material in outputs if not connected]
    return []


def get_all_input_ports(
    wb_name: str,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]]
) -> List[str]:
    """Return all input port names for a given Waben based on interface definitions."""
    return [
        port_name
        for port_type, port_name in ModPlant_interfaces.get(wb_name, [])
        if port_type == "Input"
    ]


def get_connected_port_pair(
    state: FullState,
    from_wb: str,
    to_wb: str,
    material: str = ""
) -> Tuple[str, str]:
    """
    Find an existing connection from 'from_wb' to 'to_wb'.
    Returns (out_port, in_port), optionally filtered by material.
    """
    for wb, _, (_, outputs) in state:
        if wb == from_wb:
            for port_name, conn, mat in outputs:
                if conn.startswith(to_wb + ":") and (not material or mat == material):
                    return port_name, conn.split(":")[1]
    return "", ""


def get_free_output_port(state: FullState, wb: str) -> str:
    """Return a single free output port of 'wb', or empty string if none is free."""
    for wb_name, _, (_, outputs) in state:
        if wb_name == wb:
            for port_name, conn, material in outputs:
                if not conn:
                    return port_name
    return ""


def get_free_input_port(
    wb_name: str,
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]]
) -> str:
    """Return a single free input port of 'wb_name', or empty string if none is free."""
    used_ports = set()
    for _, _, (_, outputs) in state:
        for _, conn, material in outputs:
            if conn.startswith(wb_name + ":"):
                used_ports.add(conn.split(":")[1])
    return next(
        (
            port
            for t, port in ModPlant_interfaces[wb_name]
            if t == "Input" and port not in used_ports
        ),
        ""
    )


def update_connection(
    state: FullState,
    wb_from: str,
    out_port: str,
    wb_to: str,
    in_port: str,
    material: str
) -> FullState:
    """Return a new state where wb_from:out_port is connected to wb_to:in_port for material."""
    new_state: List[WabenStateTuple] = []
    target_str = wb_to + ":" + in_port
    for wb, (content_key, content_val), (out_key, outputs) in state:
        if wb != wb_from:
            new_state.append((wb, (content_key, content_val), (out_key, outputs)))
        else:
            new_outputs = tuple(
                (name, target_str, material) if name == out_port else (name, conn, mat)
                for name, conn, mat in outputs
            )
            new_state.append((wb, (content_key, content_val), (out_key, new_outputs)))
    return tuple(sorted(new_state))

# ========== Pure Connect Operation ==========

def pure_set_connection(
    state: FullState,
    wb_from: str,
    out_port: str,
    wb_to: str,
    in_port: str,
    material: str,
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
) -> Optional[Tuple[FullState, CandidateTransition]]:
    """
    Pure version of 'set_connection':
    - does NOT touch visited / queue / transition_list
    - returns (new_state, candidate_transition) or None if no connection possible
    """
    if not is_input_free(state, wb_to, in_port):
        return None

    new_state = update_connection(state, wb_from, out_port, wb_to, in_port, material)
    op_str = f"Connect({out_port} -> {in_port}) for {material}"

    cost1 = get_operation_cost(ModPlant_ops, wb_from, "Connect")
    cost2 = get_operation_cost(ModPlant_ops, wb_to, "Connect")
    cost = float(cost1 + cost2)
    duration = 3.0

    cand: CandidateTransition = (state, new_state, op_str, cost, duration)
    return new_state, cand


# ========== Content Update Helpers (already pure) ==========

def apply_draining(state: FullState, wb: str) -> FullState:
    """Return a new state where all material is removed from Waben 'wb'."""
    new_state: List[WabenStateTuple] = []
    for w, (content_key, content_val), (out_key, outputs) in state:
        if w == wb:
            new_state.append((w, (content_key, ()), (out_key, outputs)))
        else:
            new_state.append((w, (content_key, content_val), (out_key, outputs)))
    return tuple(sorted(new_state))


def apply_filling(state: FullState, wb: str, new_content: Tuple[str, ...]) -> FullState:
    """Return a new state where Waben 'wb' content is replaced by 'new_content'."""
    new_state: List[WabenStateTuple] = []
    for w, (content_key, content_val), (out_key, outputs) in state:
        if w == wb:
            new_state.append((w, (content_key, new_content), (out_key, outputs)))
        else:
            new_state.append((w, (content_key, content_val), (out_key, outputs)))
    return tuple(sorted(new_state))


def apply_partial_draining(
    state: FullState,
    wb: str,
    to_drain: Dict[str, float]
) -> FullState:
    """Return a new state where specified material quantities are removed from Waben 'wb'."""
    new_state: List[WabenStateTuple] = []
    for w, (content_key, content_val), (out_key, outputs) in state:
        if w != wb:
            new_state.append((w, (content_key, content_val), (out_key, outputs)))
            continue

        current_dict = get_material_dict(content_val)
        for k, v in to_drain.items():
            if k in current_dict:
                current_dict[k] -= v
                if current_dict[k] <= 0:
                    del current_dict[k]
        new_content = tuple(f"{k}: {v} litre" for k, v in current_dict.items())
        new_state.append((w, (content_key, new_content), (out_key, outputs)))
    return tuple(sorted(new_state))


def apply_partial_filling(
    state: FullState,
    wb: str,
    to_fill: Dict[str, float]
) -> FullState:
    """Return a new state where specified material quantities are added to Waben 'wb'."""
    new_state: List[WabenStateTuple] = []
    for w, (content_key, content_val), (out_key, outputs) in state:
        if w != wb:
            new_state.append((w, (content_key, content_val), (out_key, outputs)))
            continue

        current_dict = get_material_dict(content_val)
        for k, v in to_fill.items():
            current_dict[k] = current_dict.get(k, 0) + v
        new_content = tuple(f"{k}: {v} litre" for k, v in current_dict.items())
        new_state.append((w, (content_key, new_content), (out_key, outputs)))
    return tuple(sorted(new_state))


# ========== Pure Transfer Operation Candidates ==========

def check_transfer_candidates(
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
) -> List[CandidateTransition]:
    """
    Pure version of 'check_transfer':
    - does not modify visited / state_list / queue / transition_list
    - returns a list of (from_state, to_state, op_str, cost, duration)
    - semantically matches "Connect + Draining + Filling"
    """
    results: List[CandidateTransition] = []

    for wb1, (content_key1, contents1), (_, _) in state:
        vol1 = parse_total_volume((content_key1, contents1))
        if vol1 <= 0:
            continue
        if not any(op[0] == 'Draining' for op in ModPlant_ops.get(wb1, [])):
            continue

        # Material name for connection (first material entry)
        material_name = ""
        if contents1:
            first_material = contents1[0].split(':')[0].strip()
            material_name = first_material

        for wb2, (content_key2, contents2), (_, _) in state:
            if wb1 == wb2:
                continue
            vol2 = parse_total_volume((content_key2, contents2))
            if vol2 > 0:
                continue
            max_capacity = ModPlant_maximum_volume.get(wb2, [0])[0]
            if max_capacity < vol1:
                continue
            if not any(op[0] == 'Filling' for op in ModPlant_ops.get(wb2, [])):
                continue

            # Check if connection already exists
            out_port, in_port = get_connected_port_pair(state, wb1, wb2, material_name)

            local_state = state
            local_cands: List[CandidateTransition] = []

            if not out_port or not in_port:
                out_port = get_free_output_port(state, wb1)
                in_port = get_free_input_port(wb2, state, ModPlant_interfaces)
                if not out_port or not in_port:
                    continue

                conn_result = pure_set_connection(
                    state, wb1, out_port, wb2, in_port, material_name, ModPlant_ops
                )
                if conn_result is None:
                    continue
                local_state, connect_cand = conn_result
                # First add the Connect candidate
                local_cands.append(connect_cand)

            # On local_state: perform Transfer (Drain + Fill)
            state_drained = apply_draining(local_state, wb1)
            state_filled = apply_filling(state_drained, wb2, contents1)

            drain_speed = next(
                (float(op[1]) for op in ModPlant_ops[wb1] if op[0] == "Draining"),
                0.01
            )
            fill_speed = next(
                (float(op[1]) for op in ModPlant_ops[wb2] if op[0] == "Filling"),
                0.01
            )
            transfer_speed = min(drain_speed, fill_speed)
            duration = round(vol1 / transfer_speed, 2)

            drain_cost = next((op[2] for op in ModPlant_ops[wb1] if op[0] == "Draining"), 0) * vol1
            fill_cost = next((op[2] for op in ModPlant_ops[wb2] if op[0] == "Filling"), 0) * vol1
            cost = float(drain_cost + fill_cost)

            transfer_op = (
                f"Dosing: Open Valve of {out_port} only, "
                f"Draining({wb1}), Filling({wb2}), "
                f"({material_name}: {vol1:.1f} litre)"
            )


            local_cands.append((local_state, state_filled, transfer_op, cost, duration))
            results.extend(local_cands)

    return results


def check_transfer_part_candidates(
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
) -> List[CandidateTransition]:
    results: List[CandidateTransition] = []
    
    if get_symbolic_vars(state):
        return []

    for wb1, (content_key1, contents1), (_, _) in state:
        if wb1 == "__VARS__": continue
        mat_dict1 = get_material_dict(contents1)
        if not mat_dict1: continue
        
        # Assume transferring the first material
        material_name = next(iter(mat_dict1.keys()))
        available_amount = mat_dict1.get(material_name, 0.0)
        
        if available_amount <= 1e-6: continue
        if not any(op[0] == "Draining" for op in ModPlant_ops.get(wb1, [])): continue

        for wb2, (content_key2, contents2), (_, _) in state:
            if wb1 == wb2 or wb2 == "__VARS__": continue
            
            mat_dict2 = get_material_dict(contents2)
            if mat_dict2 and (len(mat_dict2) != 1 or material_name not in mat_dict2):
                continue

            max_capacity = ModPlant_maximum_volume.get(wb2, [0.0])[0]
            current_vol2 = sum(mat_dict2.values())
            space_available = max_capacity - current_vol2
            
            if space_available <= 1e-6: continue
            if not any(op[0] == "Filling" for op in ModPlant_ops.get(wb2, [])): continue

            # Connection logic
            out_port, in_port = get_connected_port_pair(state, wb1, wb2, material_name)
            local_state = state
            local_cands = []

            if not out_port or not in_port:
                out_port = get_free_output_port(state, wb1)
                in_port = get_free_input_port(wb2, state, ModPlant_interfaces)
                if not out_port or not in_port: continue

                conn_result = pure_set_connection(
                    state, wb1, out_port, wb2, in_port, material_name, ModPlant_ops
                )
                if conn_result is None: continue
                local_state, connect_cand = conn_result
                local_cands.append(connect_cand)

            # Cost/Duration calculation
            drain_cost_unit = next((float(op[2]) for op in ModPlant_ops.get(wb1, []) if op[0] == "Draining"), 0.0)
            fill_cost_unit = next((float(op[2]) for op in ModPlant_ops.get(wb2, []) if op[0] == "Filling"), 0.0)
            cost_coeff = drain_cost_unit + fill_cost_unit
            
            drain_speed = next((float(op[1]) for op in ModPlant_ops.get(wb1, []) if op[0] == "Draining"), 0.01)
            fill_speed = next((float(op[1]) for op in ModPlant_ops.get(wb2, []) if op[0] == "Filling"), 0.01)
            transfer_speed = min(drain_speed, fill_speed)
            
            dur_coeff = 1.0 / transfer_speed if transfer_speed > 1e-9 else 0.0

            # Generate state
            max_transfer = min(available_amount, space_available)
            var_name = "x"
            
            new_state_list = []
            for w, (c_key, c_val), (o_key, outputs) in local_state:
                if w == wb1:
                    new_c_list = []
                    for k, v in mat_dict1.items():
                        if k == material_name:
                            new_c_list.append(f"{k}: {v} + -1.0 {var_name} litre")
                        else:
                            new_c_list.append(f"{k}: {v} litre")
                    new_state_list.append((w, (c_key, tuple(sorted(new_c_list))), (o_key, outputs)))
                elif w == wb2:
                    base_v = mat_dict2.get(material_name, 0.0)
                    new_c = (f"{material_name}: {base_v} + 1.0 {var_name} litre",)
                    new_state_list.append((w, (c_key, new_c), (o_key, outputs)))
                else:
                    new_state_list.append((w, (c_key, c_val), (o_key, outputs)))
            
            constraint_entry = (
                "__VARS__", 
                ("vars", (f"{var_name}:0.0:{max_transfer}",)), 
                ("coeffs", (f"cost:{cost_coeff}", f"dur:{dur_coeff}")) 
            )
            new_state_list.append(constraint_entry)
            symbolic_state = tuple(sorted(new_state_list, key=lambda x: x[0]))

            # === Change: append 'for <material>' in the op string ===
            op_str = f"Variable Transfer setup ({wb1}->{wb2}) for {material_name}: {var_name} in [0, {max_transfer:.2f}]"
            
            local_cands.append((local_state, symbolic_state, op_str, 0.0, 0.0))
            
            results.extend(local_cands)

    return results

def compute_dosing_candidates(
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
    rule_row: pd.Series,
) -> List[CandidateTransition]:
    """Generate pure Dosing candidates for a single state and rule."""
    results: List[CandidateTransition] = []

    req_dose = parse_material_string(rule_row["Reaction Param"])
    req_inputs = parse_material_string(rule_row["Inputs"]) if rule_row["Inputs"] else None

    for wb1, (content_key1, contents1), (_, _) in state:
        if not any(op[0] == "Draining" for op in ModPlant_ops.get(wb1, [])):
            continue
        wb1_dict = get_material_dict(contents1)
        if any(wb1_dict.get(k, 0) < v for k, v in req_dose.items()):
            continue

        material_name = ", ".join(req_dose.keys()) if req_dose else ""

        for wb2, (content_key2, contents2), (_, _) in state:
            if wb1 == wb2:
                continue
            if not any(op[0] == "Filling" for op in ModPlant_ops.get(wb2, [])):
                continue
            wb2_dict = get_material_dict(contents2)

            if req_inputs:
                if wb2_dict != req_inputs:
                    continue
            else:
                if wb2_dict:
                    continue

            out_port, in_port = get_connected_port_pair(state, wb1, wb2, material_name)
            local_state = state
            local_cands: List[CandidateTransition] = []

            if not out_port or not in_port:
                out_port = get_free_output_port(state, wb1)
                in_port = get_free_input_port(wb2, state, ModPlant_interfaces)
                if not out_port or not in_port:
                    continue
                conn_result = pure_set_connection(
                    state, wb1, out_port, wb2, in_port, material_name, ModPlant_ops
                )
                if conn_result is None:
                    continue
                local_state, connect_cand = conn_result
                local_cands.append(connect_cand)

            drained = apply_partial_draining(local_state, wb1, req_dose)
            filled = apply_partial_filling(drained, wb2, req_dose)

            dose_amount = sum(req_dose.values())
            draining_speed = float(next(
                (op[1] for op in ModPlant_ops[wb1] if op[0] == "Draining"),
                1.0
            ))
            filling_speed = float(next(
                (op[1] for op in ModPlant_ops[wb2] if op[0] == "Filling"),
                1.0
            ))
            speed = min(draining_speed, filling_speed) if draining_speed and filling_speed else 1.0
            duration = dose_amount / speed if speed > 0 else 0.0

            cost_draining = next((op[2] for op in ModPlant_ops[wb1] if op[0] == "Draining"), 0) * dose_amount
            cost_filling = next((op[2] for op in ModPlant_ops[wb2] if op[0] == "Filling"), 0) * dose_amount
            cost = float(cost_draining + cost_filling)

            # === Change: format volumes to two decimal places ===
            raw_param = rule_row['Reaction Param']
            formatted_parts = []
            if raw_param:
                for p in raw_param.split(','):
                    if ':' in p:
                        name, rest = p.split(':', 1)
                        tokens = rest.strip().split()
                        if tokens:
                            try:
                                # Attempt to parse and keep two decimals
                                val = float(tokens[0])
                                unit = " ".join(tokens[1:])
                                formatted_parts.append(f"{name.strip()}: {val:.2f} {unit}")
                            except ValueError:
                                formatted_parts.append(p.strip())
                        else:
                            formatted_parts.append(p.strip())
                    else:
                        formatted_parts.append(p.strip())
                formatted_param = ", ".join(formatted_parts)
            else:
                formatted_param = ""
            # ==================================

            op_str = (
                f"Dosing: Open Valve of {out_port} only, "
                f"Draining({wb1}), Filling({wb2}), ({formatted_param})"
            )

            local_cands.append((local_state, filled, op_str, cost, duration))
            results.extend(local_cands)

    return results

def format_result_content(result_mat_str: str, total_vol: float, unit: str = "litre") -> str:
    """
    Format the result material string with total volume.
    Returns a string in the format "Material : volume unit"
    """
    if ":" in result_mat_str:
        # Already includes volume
        return result_mat_str
    else:
        # Only material name, append volume
        return f"{result_mat_str} : {total_vol:.1f} {unit}"

def compute_mixing_candidates(
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
    rule_row: pd.Series,
) -> List[CandidateTransition]:
    results: List[CandidateTransition] = []
    
    # Parse required inputs
    req_inputs = parse_material_string(rule_row["Inputs"])
    if not req_inputs: return []
    
    # Parse reaction parameters (rpm/duration)
    param_str = str(rule_row["Reaction Param"])
    if "/" in param_str:
        parts = param_str.split('/')
        rpm_str = parts[0].strip().split()[0]
        dur_str = parts[1].strip().split()[0]
    else:
        # If no '/', assume only duration provided
        rpm_str = "0"
        dur_str = param_str.strip().split()[0] if param_str.strip() else "0"

    try:
        duration = float(dur_str)
    except:
        duration = 0.0

    for wb, (content_key, contents), (_, _) in state:
        content_dict = get_material_dict(contents)

        # 1. Check materials
        if not are_materials_equal(content_dict, req_inputs, tol=1e-4): continue

        # 2. Check mixing operation with matching rpm
        # Attention: rpm is stored as string in operation
        matching_mix = next(
            (op for op in ModPlant_ops.get(wb, []) if op[0] == "Stirring" and str(op[1]) == rpm_str),
            None
        )
        #print(f"Checking Waben {wb} for Mixing at {rpm_str} rpm: {'Found' if matching_mix else 'Not Found'}")
        if not matching_mix: continue

        cost = float(matching_mix[2])
        
        # 3. Construct new content based on Result
        total_vol = sum(content_dict.values())
        unit = "litre"
        if contents:
            parts = contents[0].split()
            if parts: unit = parts[-1]
            
        result_mat = rule_row["Result"]
        # FIX: Use helper to format result content
        new_content_str = format_result_content(result_mat, total_vol, unit)
        
        new_state_list = []
        for w, (c_key, c_val), (out_key, outputs) in state:
            if w == wb:
                new_state_list.append((w, (c_key, (new_content_str,)), (out_key, outputs)))
            else:
                new_state_list.append((w, (c_key, c_val), (out_key, outputs)))
        new_state = tuple(sorted(new_state_list))

        op_str = f"Stirring ({wb}), {rpm_str}rpm for {duration}s"
        results.append((state, new_state, op_str, cost, duration))
        #print( f"Mixing candidate: {op_str}, cost: {cost}, duration: {duration}" )
    return results

def compute_usage_candidates(
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
    rule_row: pd.Series,
) -> List[CandidateTransition]:
    """Generate pure Usage candidates for a single state and rule."""
    
    #print(f"Computing Usage candidates for rule: {rule_row['Inputs']} -> {rule_row['Result']}")

    results: List[CandidateTransition] = []

    req_inputs = parse_material_string(rule_row["Inputs"])
    #print(f"Required inputs for Usage: {req_inputs}")

    try:
        duration_str = rule_row["Reaction Param"].strip().split()[0]
        duration_val = float(duration_str)
    except Exception:
        return results

    for wb, (content_key, contents), (_, _) in state:
        content_dict = get_material_dict(contents)
        
        if not are_materials_equal(content_dict, req_inputs): continue
        
        has_none_op = any(op[0] == "None" for op in ModPlant_ops.get(wb, []))
        if not has_none_op:
            continue
        
        
        result_mat = rule_row["Result"]

        # ========== FIX: Keep total volume ==========
        total_vol = sum(content_dict.values())
        
        unit = "litre"
        if contents:
            parts = contents[0].split()
            if parts: unit = parts[-1]
            
        new_content_str = f"{result_mat} : {total_vol:.2f} {unit}"
        new_content = (new_content_str,)
        # ==================================

        new_state_list: List[WabenStateTuple] = []
        for w, (c_key, c_val), (out_key, outputs) in state:
            if w == wb:
                new_state_list.append((w, (c_key, new_content), (out_key, outputs)))
            else:
                new_state_list.append((w, (c_key, c_val), (out_key, outputs)))
        new_state = tuple(sorted(new_state_list))

        op_str = f"Usage ({wb}), {duration_val}s: None"
        results.append((state, new_state, op_str, 0.0, 0.0))

    return results



def compute_settling_candidates(
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
    rule_row: pd.Series,
) -> List[CandidateTransition]:
    """Generate pure Settling candidates for a single state and rule."""
    results: List[CandidateTransition] = []

    req_inputs = parse_material_string(rule_row["Inputs"])

    try:
        duration_str = rule_row["Reaction Param"].strip().split()[0]
        duration_val = float(duration_str)
    except Exception:
        return results

    for wb, (content_key, contents), (_, _) in state:
        content_dict = get_material_dict(contents)
        if not are_materials_equal(content_dict, req_inputs): continue

        settling_op = next((op for op in ModPlant_ops.get(wb, []) if op[0] == "Settling"), None)
        if not settling_op:
            continue

        cost = float(settling_op[2])
        result_mat = rule_row["Result"]

        new_state_list: List[WabenStateTuple] = []
        for w, (c_key, c_val), (out_key, outputs) in state:
            if w == wb:
                new_content = (f"{result_mat}",)
                new_state_list.append((w, (c_key, new_content), (out_key, outputs)))
            else:
                new_state_list.append((w, (c_key, c_val), (out_key, outputs)))
        new_state = tuple(sorted(new_state_list))

        op_str = f"Settling ({wb}), {duration_val}s: Settling"
        # Original code used duration=0 in the transition, keep this semantics
        results.append((state, new_state, op_str, cost, 0.0))

    return results

def compute_separation_candidates(
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
    rule_row: pd.Series,
) -> List[CandidateTransition]:
    """Generate pure Separation candidates for a single state and rule."""
    results: List[CandidateTransition] = []

    req_inputs = parse_material_string(rule_row["Inputs"])
    result_material = parse_material_string(rule_row["Result"])
    separated_comp = rule_row["Reaction Param"].strip()

    for wb1, (content_key1, contents1), (_, _) in state:
        content_dict1 = get_material_dict(contents1)
        if content_dict1 != req_inputs:
            continue
        if not any(op[0] == "Draining" for op in ModPlant_ops.get(wb1, [])):
            continue

        for wb2, (content_key2, contents2), (_, _) in state:
            if wb1 == wb2:
                continue
            content_dict2 = get_material_dict(contents2)
            if content_dict2 and (list(content_dict2.keys()) != [separated_comp]):
                continue

            capacity_left = float('inf')
            for op in ModPlant_ops.get(wb2, []):
                if op[0] == "Filling":
                    capacity_left = ModPlant_maximum_volume.get(wb2, [0])[0]
                    break
            used_volume = sum(content_dict2.values())
            remaining_capacity = capacity_left - used_volume
            if remaining_capacity <= 0:
                continue

            source_volume = sum(req_inputs.values())
            target_volume = (
                0.0 if rule_row["Result"] == "End" else sum(result_material.values())
            )
            transfer_volume = max(0.0, source_volume - target_volume)
            if transfer_volume > remaining_capacity:
                continue

            out_port, in_port = get_connected_port_pair(state, wb1, wb2, separated_comp)
            local_state = state
            local_cands: List[CandidateTransition] = []

            if not out_port or not in_port:
                out_port = get_free_output_port(state, wb1)
                in_port = get_free_input_port(wb2, state, ModPlant_interfaces)
                if not out_port or not in_port:
                    continue
                conn_result = pure_set_connection(
                    state, wb1, out_port, wb2, in_port, separated_comp, ModPlant_ops
                )
                if conn_result is None:
                    continue
                local_state, connect_cand = conn_result
                local_cands.append(connect_cand)

            drain_dict = {separated_comp: transfer_volume}
            drained = apply_partial_draining(local_state, wb1, drain_dict)
            filled = apply_partial_filling(drained, wb2, drain_dict)

            draining_speed = float(next(
                (op[1] for op in ModPlant_ops[wb1] if op[0] == "Draining"),
                1.0
            ))
            filling_speed = float(next(
                (op[1] for op in ModPlant_ops[wb2] if op[0] == "Filling"),
                1.0
            ))
            speed = min(draining_speed, filling_speed)
            duration_val = transfer_volume / speed if speed > 0 else 0.0

            cost_draining = next((op[2] for op in ModPlant_ops[wb1] if op[0] == "Draining"), 0)
            cost_filling = next((op[2] for op in ModPlant_ops[wb2] if op[0] == "Filling"), 0)
            cost = float(cost_draining + cost_filling)

            final_state_list: List[WabenStateTuple] = []
            for w, (c_key, c_val), (out_key, outputs) in filled:
                if w == wb1:
                    new_content = tuple(f"{k}: {v} litre" for k, v in result_material.items())
                    final_state_list.append((w, (c_key, new_content), (out_key, outputs)))
                else:
                    final_state_list.append((w, (c_key, c_val), (out_key, outputs)))
            final_state = tuple(sorted(final_state_list))

            op_str = (
                f"Separation: Open Valve of {out_port} only, "
                f"Draining({wb1}), Filling({wb2}), "
                f"({separated_comp}: {transfer_volume:.2f} litre)"
            )

            if rule_row["Result"] == "End":
                # Direct transition to end_state
                local_cands.append((local_state, end_state, "End, " + op_str, cost, 0.0))
            else:
                local_cands.append((local_state, final_state, op_str, cost, 0.0))

            results.extend(local_cands)

    return results

def solve_and_instantiate(
    state: FullState,
    wb_name: str,
    required_material: Dict[str, float]
) -> Optional[Tuple[FullState, float]]:
    """
    Resolve symbolic variable x for a specific Waben and instantiate a concrete state.
    No persistent cache to keep runs stateless across kernels.
    """
    vars_list = get_symbolic_vars(state)
    if not vars_list:
        return None

    var_name, v_min, v_max = vars_list[0]

    # locate target Waben content
    target_content = None
    for w, (_, content), _ in state:
        if w == wb_name:
            target_content = content
            break
    if target_content is None:
        return None

    solved_x = None

    # Branch 1: required_material empty -> solve base + coeff * x = 0
    if not required_material:
        for entry in target_content:
            if var_name not in entry:
                continue
            if "_mixed" in entry:
                continue
            try:
                rhs = entry.split(":", 1)[1]
                tokens = rhs.split()
                if var_name in tokens:
                    idx = tokens.index(var_name)
                    if idx > 0 and idx + 1 < len(tokens):
                        coeff = float(tokens[idx - 1])
                        base = float(tokens[0])
                        if abs(coeff) > 1e-9:
                            solved_x = -base / coeff
                            break
            except Exception:
                continue

    # Branch 2: required_material present -> match quantities exactly
    else:
        current_sym = parse_material_string_symbolic(', '.join(target_content))
        for mat, req_qty in required_material.items():
            if mat not in current_sym:
                return None
            info = current_sym[mat]
            base = info['base']
            coeffs = info['vars']

            if var_name not in coeffs:
                if abs(base - req_qty) > 1e-6:
                    return None
            else:
                coeff = coeffs[var_name]
                if abs(coeff) < 1e-9:
                    if abs(base - req_qty) > 1e-6:
                        return None
                else:
                    val = (req_qty - base) / coeff
                    if solved_x is not None and abs(solved_x - val) > 1e-6:
                        return None
                    solved_x = val

    if solved_x is None:
        return None

    if not (v_min - 1e-6 <= solved_x <= v_max + 1e-6):
        return None
    if solved_x < v_min:
        solved_x = v_min
    if solved_x > v_max:
        solved_x = v_max

    # rebuild only ModPlants containing var_name
    new_state_list = []
    for w, (c_key, c_val), (o_key, outputs) in state:
        if w == "__VARS__":
            continue

        has_var = any(var_name in line for line in c_val)
        if not has_var:
            new_state_list.append((w, (c_key, c_val), (o_key, outputs)))
            continue

        parsed = parse_material_string_symbolic(', '.join(c_val))
        new_mat_strs = []
        for m, info in parsed.items():
            qty = info['base']
            if var_name in info['vars']:
                qty += info['vars'][var_name] * solved_x
            if abs(qty) < 1e-6:
                continue
            new_mat_strs.append(f"{m}: {qty:.2f} litre")

        new_state_list.append((w, (c_key, tuple(sorted(new_mat_strs))), (o_key, outputs)))

    concrete_state = tuple(sorted(new_state_list, key=lambda x: x[0]))
    return concrete_state, solved_x



def check_rules_candidates(
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
    reaction_rules_df: pd.DataFrame,
) -> List[CandidateTransition]:
    results: List[CandidateTransition] = []
    is_symbolic = bool(get_symbolic_vars(state))

    for _, rule_row in reaction_rules_df.iterrows():
        raw_inputs = rule_row["Inputs"]
        req_inputs = parse_material_string(raw_inputs)
        rtype = rule_row["Reaction Type"]
        
        # Pre-parse Mix rpm so symbolic mode skips unsupported vessels quickly
        rpm_required = None
        if rtype == "Mix":
            param_str = str(rule_row["Reaction Param"])
            if "/" in param_str:
                rpm_required = param_str.split('/')[0].strip().split()[0]
            else:
                tokens = param_str.strip().split()
                rpm_required = tokens[0] if tokens else None
        
        # === Path A: symbolic state ===
        if is_symbolic:
            # if is_symbolic and rule_row["Reaction Type"] != "Dosing":
            #     continue
            cost_coeff, dur_coeff = get_symbolic_coeffs(state)
            
            for wb, _, _ in state:
                if wb == "__VARS__": continue
                
                if rpm_required is not None:
                    ops = ModPlant_ops.get(wb, [])
                    if not any(op[0] == "Stirring" and str(op[1]) == rpm_required for op in ops):
                        continue
                
                sol = solve_and_instantiate(state, wb, req_inputs)
                if sol:
                    concrete_state, solved_x = sol
                    
                    resolve_cost = cost_coeff * solved_x
                    resolve_dur = dur_coeff * solved_x
                    
                    # === Change: include material name in Resolve description ===
                    mat_name = list(req_inputs.keys())[0] if req_inputs else "Unknown"
                    op_str = f"Resolve(x={solved_x:.2f}) for {mat_name}"
                    
                    results.append((state, concrete_state, op_str, resolve_cost, resolve_dur))
            continue

        # === Path B: concrete state (original logic) ===
        candidates = []
        
        if rtype == "Dosing":
            candidates = compute_dosing_candidates(state, ModPlant_interfaces, ModPlant_ops, rule_row)
        elif rtype == "Mix":
            candidates = compute_mixing_candidates(state, ModPlant_interfaces, ModPlant_ops, rule_row)
        elif rtype == "Usage":
            candidates = compute_usage_candidates(state, ModPlant_interfaces, ModPlant_ops, rule_row)
        elif rtype == "Settling":
            candidates = compute_settling_candidates(state, ModPlant_interfaces, ModPlant_ops, rule_row)
        elif rtype == "Separation":
            candidates = compute_separation_candidates(state, ModPlant_interfaces, ModPlant_ops, rule_row)
            
        for (cand_from_state, to_state, op_str, op_cost, op_dur) in candidates:
            # Detect implicit connection changes
            has_conn_change = False
            conn_op_str = ""
            conn_cost = 0.0
            conn_dur = 0.0
            
            if len(state) == len(cand_from_state):
                 for i in range(len(state)):
                    w1, _, out1 = state[i]
                    w2, _, out2 = cand_from_state[i]
                    if w1 == "__VARS__" or w2 == "__VARS__": continue
                    
                    if out1 != out2:
                        has_conn_change = True
                        for p_idx, port_def in enumerate(out2[1]):
                            old_def = out1[1][p_idx]
                            if port_def != old_def and port_def[1] != "":
                                target_full = port_def[1]
                                target_clean = target_full.split(":")[1] if ":" in target_full else target_full
                                conn_op_str += f"Connect({port_def[0]} -> {target_clean}) for {port_def[2]} & "
                                conn_cost += 2.0 
                                conn_dur += 3.0
            
            if has_conn_change:
                conn_op_str = conn_op_str.rstrip(" & ")
                results.append((state, cand_from_state, conn_op_str, conn_cost, conn_dur))
            else:
                results.append((state, to_state, op_str, op_cost, op_dur))

    return results



# ========== Pure Expansion for a Single State ==========
def expand_state_pure(
    state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    ModPlant_ops: Dict[str, List[Tuple[str, Union[str, float], int]]],
    reaction_rules_df: pd.DataFrame,
) -> List[CandidateTransition]:
    """
    Pure expansion of a single state:
    - Transfer operations
    - Reaction Rules (Dosing / Mix / Usage / Settling / Separation)
    """
    candidates: List[CandidateTransition] = []
    #candidates.extend(check_transfer_candidates(state, ModPlant_interfaces, ModPlant_ops))
    candidates.extend(check_transfer_part_candidates(state, ModPlant_interfaces, ModPlant_ops))
    candidates.extend(check_rules_candidates(state, ModPlant_interfaces, ModPlant_ops, reaction_rules_df))
    return candidates


# ========== Multi-process BFS Core (Python 3.11 friendly) ==========

def run_bfs(
    start_state: FullState,
    ModPlant_interfaces: Dict[str, List[Tuple[str, str]]],
    max_steps: Optional[int] = None,
    num_workers: int = 8,
    batch_size: Optional[int] = None,
) -> Tuple[List[Transition], List[FullState]]:
    """
    Multi-process BFS:
    - Uses a process pool to expand states in coarse-grained chunks.
    - Only the main process modifies visited / state_list / queue / transition_list.
    - max_steps: if not None, limit BFS depth; otherwise, run without a depth limit.
    """
    num_workers = max(1, num_workers)
    if batch_size is None:
        batch_size = max(128, num_workers * 32)

    visited: Dict[FullState, int] = {}
    state_list: List[FullState] = []
    transition_list: List[Transition] = []
    existing_transitions: Set[Tuple[int, str, int, FullState]] = set()
    queue: deque = deque()

    # Initialize start / end states
    start_id = 0
    visited[start_state] = start_id
    state_list.append(start_state)
    queue.append((start_state, start_id, 0))

    end_id = 1
    visited[end_state] = end_id
    state_list.append(end_state)
    # end_state is not expanded
    queue.append((end_state, end_id, 0))

    def add_transition(
        from_id: int,
        from_state: FullState,
        op_str: str,
        to_id: int,
        to_state: FullState,
        cost: float,
        duration: float,
    ) -> None:
        key = (from_id, op_str, to_id, to_state)
        if key in existing_transitions:
            return
        transition_list.append((from_id, from_state, op_str, to_id, to_state, cost, duration))
        existing_transitions.add(key)

    executor_kwargs = {
        "max_workers": num_workers,
        "initializer": _init_bfs_worker,
        "initargs": (ModPlant_interfaces, ModPlant_ops, reaction_rules_df, ModPlant_maximum_volume, end_state),
    }
    mp_context = _get_bfs_mp_context()
    if mp_context is not None:
        executor_kwargs["mp_context"] = mp_context

    with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
        while queue:
            batch: List[Tuple[FullState, int, int]] = []

            # Collect a larger frontier slice so worker processes get enough work.
            while queue and len(batch) < batch_size:
                current_state, current_id, depth = queue.popleft()
                if current_id == end_id:
                    continue
                if max_steps is not None and depth >= max_steps:
                    continue
                batch.append((current_state, current_id, depth))

            if not batch:
                if not queue:
                    break
                continue

            target_chunk_count = max(num_workers * 4, 1)
            chunk_size = max(1, (len(batch) + target_chunk_count - 1) // target_chunk_count)
            batch_chunks = [batch[i:i + chunk_size] for i in range(0, len(batch), chunk_size)]

            future_to_chunk = {}
            for chunk in batch_chunks:
                states_only = [state for state, _, _ in chunk]
                fut = executor.submit(_expand_state_batch_worker, states_only)
                future_to_chunk[fut] = chunk

            # Collect chunk results as they complete to reduce parent-side idle time.
            for fut in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[fut]
                try:
                    chunk_results = fut.result()
                except Exception as e:
                    print(f"Error expanding state chunk: {e}")
                    continue

                for (current_state, current_id, depth), cands in zip(chunk, chunk_results):
                    for from_state, to_state, op_str, cost, duration in cands:
                        # Ensure from_state is in visited
                        if from_state not in visited:
                            new_from_id = len(state_list)
                            visited[from_state] = new_from_id
                            state_list.append(from_state)
                            queue.append((from_state, new_from_id, depth + 1))
                        from_id = visited[from_state]

                        # Ensure to_state is in visited
                        if to_state not in visited:
                            new_to_id = len(state_list)
                            visited[to_state] = new_to_id
                            state_list.append(to_state)
                            queue.append((to_state, new_to_id, depth + 1))
                        to_id = visited[to_state]

                        add_transition(from_id, from_state, op_str, to_id, to_state, cost, duration)

    return transition_list, state_list

# ========== Run multi-process BFS and collect graph ==========
current_workers = 0
current_batch_size = 0

# Get CPU core count (default to 1 if detection fails)
import os
cpu_cores = os.cpu_count() or 1

# Calculate dynamic worker count: leave one core for the parent merge process.
env_workers = os.getenv("WABEN_BFS_WORKERS")
env_batch_size = os.getenv("WABEN_BFS_BATCH_SIZE")

if env_workers is not None:
    current_workers = max(1, int(env_workers))
else:
    dynamic_workers = max(1, cpu_cores - 1)
    current_workers = dynamic_workers

if env_batch_size is not None:
    current_batch_size = max(1, int(env_batch_size))
else:
    current_batch_size = max(128, current_workers * 32)

print(f"Using {current_workers} processes with batch size {current_batch_size}")

# Pre-check: ensure Mix RPM requirements exist in ModPlant_ops; otherwise skip BFS
mix_rpms = set()
for _, row in reaction_rules_df.iterrows():
    if str(row.get('Reaction Type','')).strip() == 'Mix':
        param_str = str(row.get('Reaction Param','')).strip()
        if '/' in param_str:
            rpm_token = param_str.split('/',1)[0].strip().split()[0]
        else:
            rpm_token = param_str.split()[0] if param_str else None
        if rpm_token:
            mix_rpms.add(str(rpm_token))
available_rpms = {str(op[1]) for ops in ModPlant_ops.values() for op in ops if op[0] == 'Stirring'}
missing_mix_rpms = {rpm for rpm in mix_rpms if rpm not in available_rpms}
skip_bfs = bool(missing_mix_rpms)
if skip_bfs:
    print(f'Skipping BFS: missing Mix RPMs {sorted(missing_mix_rpms)} in all ModPlants')
    transition_list = []
    state_list = [start_state, end_state]
    state_index = {start_state: 0, end_state: 1}
    index_state = {0: start_state, 1: end_state}
    bfs_feasible = False
    operation_rows_fallback = [
        {'Step': 1, 'Operation': 'Unfeasible - missing RPM', 'Cost': 0.0, 'Duration (s)': 0.0},
        {'Step': 2, 'Operation': 'End', 'Cost': 0.0, 'Duration (s)': 0.0},
    ]
    tools.display_dataframe_to_user(name='Operation Sequence Table', dataframe=pd.DataFrame(operation_rows_fallback))
else:
    transition_list, state_list = run_bfs(
        start_state,
        ModPlant_interfaces,
        max_steps=None,   # set to None for unlimited search depth if you like
        num_workers=current_workers,         # change this to control the number of worker processes
        batch_size=current_batch_size,       # None -> default = num_workers * 2
    )

# ========== Build state index mappings ==========

state_index = {state: idx for idx, state in enumerate(state_list)}
index_state = {idx: state for idx, state in enumerate(state_list)}

# ========== Convert transitions to DataFrame for inspection ==========

df = pd.DataFrame(
    [
        {
            "From_ID": fid,
            "From_State": str(fstate),
            "Operation": op,
            "Cost": cost,
            "Duration": duration,
            "To_ID": tid,
            "To_State": str(tstate),
        }
        for fid, fstate, op, tid, tstate, cost, duration in transition_list
    ]
)

tools.display_dataframe_to_user(name="Transition List", dataframe=df)

# ========== Check if at least one transition reaches end_state ==========

end_idx = state_index.get(end_state, -1)
has_end_transition = any(
    op.startswith("End,") and tid == end_idx
    for _, _, op, tid, _, _, _ in transition_list
)

bfs_feasible = has_end_transition and len(transition_list) > 0
operation_rows_fallback = None

if not bfs_feasible:
    operation_rows_fallback = [
        {"Step": 1, "Operation": "Unfeasible", "Cost": 0.0, "Duration (s)": 0.0},
        {"Step": 2, "Operation": "End", "Cost": 0.0, "Duration (s)": 0.0},
    ]
    tools.display_dataframe_to_user(name="Operation Sequence Table", dataframe=pd.DataFrame(operation_rows_fallback))
    print("BFS found no feasible solution; falling back to stub sequence.")


# Pyomo model
if not bfs_feasible:
    model = None
    trans_data = {}
    print("BFS infeasible: skipping model/optimization; proceeding directly to export.")
else:
    from pyomo.environ import ConcreteModel, RangeSet, Var, Binary
    model = ConcreteModel(name="Waben_Operation_Planner")
    model.S = RangeSet(0, len(state_list) - 1)
    model.T = RangeSet(0, len(transition_list) - 1)
    model.x = Var(model.T, domain=Binary)

    trans_data = {
        i: (fid, tid, op, cost, duration)
        for i, (fid, fstate, op, tid, tstate, cost, duration) in enumerate(transition_list)
    }

from pyomo.environ import Objective, maximize

if model is not None:
    def total_profit(m):
        total_cost = sum(model.x[t] * trans_data[t][3] for t in model.T)
        total_duration = sum(model.x[t] * trans_data[t][4] for t in model.T)
        fixed_profit = order_profit_factor[0]
        delay_penalty = total_duration * order_profit_factor[1]
        return fixed_profit - total_cost + delay_penalty

    model.obj = Objective(rule=total_profit, sense=maximize)
else:
    print("Objective skipped because model is None (BFS infeasible).")


# model.start_constraint = Constraint(
#     expr=sum(model.x[t] for t in model.T if trans_data[t][0] == state_index[start_state]) == 1
# )


from collections import defaultdict

if model is not None:
    in_edges = defaultdict(list)
    out_edges = defaultdict(list)

    for t in model.T:
        from_state, to_state, *_ = trans_data[t]
        in_edges[to_state].append(t)
        out_edges[from_state].append(t)

    start_idx = state_index[start_state]
    end_idx = state_index[end_state]

    def flow_rule(m, s):
        inflow = sum(m.x[t] for t in in_edges[s])
        outflow = sum(m.x[t] for t in out_edges[s])
        if s == start_idx:
            return outflow - inflow == 1
        elif s == end_idx:
            return inflow - outflow == 1
        else:
            return inflow - outflow == 0

    model.flow = Constraint(model.S, rule=flow_rule)
else:
    print("Flow constraints skipped because model is None (BFS infeasible).")

def solve_and_display(model):
    try:
        model.max_steps = Constraint(expr=sum(model.x[t] for t in model.T) <= 1000)
        solver_attempts = []
        for solver_name in ("gurobi", "glpk"):
            solver = SolverFactory(solver_name)
            try:
                available = solver is not None and solver.available(exception_flag=False)
            except Exception:
                available = False
            if not available:
                solver_attempts.append(f"{solver_name}: unavailable")
                continue

            print(f"Trying solver: {solver_name}")
            try:
                results = solver.solve(model, tee=True)
            except Exception as exc:
                solver_attempts.append(f"{solver_name}: {exc}")
                continue

            if (
                results.solver.termination_condition == TerminationCondition.optimal
                and results.solver.status == 'ok'
            ):
                df_result = pd.DataFrame([{"Optimal Objective Value": value(model.obj)}])
                tools.display_dataframe_to_user(name="Optimal Objective Value", dataframe=df_result)
                return

            solver_attempts.append(
                f"{solver_name}: status={results.solver.status}, termination={results.solver.termination_condition}"
            )

        raise Exception("; ".join(solver_attempts) or "no available solver")
    except Exception as e:
        raise Exception(f"Problem is infeasible or unsolvable: {str(e)}")

if model is not None:
    solve_and_display(model)
else:
    print("Skipping solver because BFS was infeasible; exporting corpus directly.")

from pyomo.environ import value
if not bfs_feasible:
    # direct fallback from BFS
    operation_rows = operation_rows_fallback
else:
    import pandas as pd
    import re
    import ace_tools_open as tools
    
    # ==========================================
    # CONFIGURATION SWITCHES
    # ==========================================
    # 1. Merge "Variable Transfer setup" + "Resolve" -> "Dosing..."
    merge_setup_resolve = True
    
    # 2. Move "Connect(...)" to top
    separate_connect_first = False
    
    # 3. Merge consecutive Dosing steps
    merge_consecutive_dosing = True
    
    # 4. Split "End, Operation" -> "Operation" + "End"
    split_end_step = True
    # ==========================================
    
    
    # === Utility: force volumes to 1 decimal place ===
    def _reformat_all_volumes(op_str: str) -> str:
        # Regex: find numbers after a colon followed by 'litre'
        # Example ": 2.00 litre" -> ": 2.0 litre"
        def repl(m):
            try:
                val = float(m.group(1))
                return f": {val:.1f} litre"
            except:
                return m.group(0)
        
        return re.sub(r":\s*(\d+\.?\d*)\s*litre", repl, op_str)
    
    
    # Helper: scale the volume(s) inside the final "(...)" of a Dosing op_str
    def _scale_dosing_volume(op_str: str, factor: int) -> str:
        m = re.search(r"\(([^()]*)\)\s*$", op_str)
        if not m: return op_str
        inner = m.group(1)
        parts = inner.split(",")
        new_parts = []
        for p in parts:
            p = p.strip()
            if ":" not in p:
                new_parts.append(p)
                continue
            name, rest = p.split(":", 1)
            tokens = rest.strip().split()
            if not tokens:
                new_parts.append(p)
                continue
            try:
                qty = float(tokens[0])
            except ValueError:
                new_parts.append(p)
                continue
            new_qty = qty * factor
            
            # .1f keeps one decimal when merging results
            tokens[0] = f"{new_qty:.1f}"
            new_rest = " ".join(tokens)
            new_parts.append(f"{name.strip()}: {new_rest}")
        new_inner = ", ".join(new_parts)
        return op_str[:m.start()] + f"({new_inner})"
    
    # 1. Extract selected transitions
    result_path = []
    for t in model.T:
        if value(model.x[t]) > 0.5:
            result_path.append(trans_data[t])
    
    # 2. Reconstruct path
    path_by_flow = []
    if start_state in state_index:
        curr_idx = state_index[start_state]
        while True:
            found = False
            for entry in result_path:
                fid, tid, op_str, cost, duration = entry
                if fid == curr_idx:
                    path_by_flow.append((op_str, cost, duration))
                    curr_idx = tid
                    found = True
                    break
            if not found:
                break
    
    # ==============================================================================
    # Processor A: Merge Setup + Resolve
    # ==============================================================================
    if merge_setup_resolve:
        merged_transfer_path = []
        skip_next = False
        n = len(path_by_flow)
    
        for i in range(n):
            if skip_next:
                skip_next = False
                continue
    
            op_str, cost, dur = path_by_flow[i]
    
            if "Variable Transfer setup" in op_str and i + 1 < n:
                next_op, next_cost, next_dur = path_by_flow[i+1]
                if "Resolve(x=" in next_op:
                    mat_match = re.search(r"for\s+([A-Za-z0-9_]+):", op_str)
                    mat_name = mat_match.group(1) if mat_match else "Transfer"
    
                    m_setup = re.search(r"\(([^)]+)->([^)]+)\)", op_str)
                    src, dst = (m_setup.group(1), m_setup.group(2)) if m_setup else ("??", "??")
                    
                    m_x = re.search(r"x=([\d\.]+)", next_op)
                    val = float(m_x.group(1)) if m_x else 0.0
    
                    # Build description using .1f here
                    final_desc = f"Dosing: Open Valve of {src}_Out1 only, Draining({src}), Filling({dst}), ({mat_name}: {val:.1f} litre)"
    
                    merged_transfer_path.append((final_desc, next_cost, next_dur))
                    skip_next = True 
                    continue
    
            merged_transfer_path.append((op_str, cost, dur))
        
        path_by_flow = merged_transfer_path
    
    # ==============================================================================
    # Processor B: Reorder Connect
    # ==============================================================================
    if separate_connect_first:
        connect_ops = [x for x in path_by_flow if x[0].strip().startswith("Connect")]
        other_ops   = [x for x in path_by_flow if not x[0].strip().startswith("Connect")]
        path_by_flow = connect_ops + other_ops
    
    # ==============================================================================
    # Processor C: Merge Consecutive Dosing (Advanced Regex-based)
    # ==============================================================================
    if merge_consecutive_dosing:
        merged_path = []
        i = 0
        n = len(path_by_flow)

        # Regex to capture critical identity info: Source, Dest, Material Name, Volume
        # Matches pattern: ... Draining(HC10), Filling(HC30), (A: 1.0 litre)
        # Group 1: Source (e.g. HC10)
        # Group 2: Dest (e.g. HC30)
        # Group 3: Material Name (e.g. A)
        # Group 4: Volume Value (e.g. 1.0)
        pattern_dosing = re.compile(r"Draining\(([^)]+)\),\s*Filling\(([^)]+)\),.*\(([^:]+):\s*([\d\.]+)\s*litre\)")

        while i < n:
            op_str, cost, duration = path_by_flow[i]
            
            # 1. Check if it's a Dosing operation
            if not op_str.strip().startswith("Dosing:"):
                merged_path.append((op_str, cost, duration))
                i += 1
                continue

            # 2. Try to parse the Dosing details
            match = pattern_dosing.search(op_str)
            if not match:
                # If regex fails (unexpected format), keep as is
                merged_path.append((op_str, cost, duration))
                i += 1
                continue

            # Initialize accumulation variables
            src, dst, mat, vol_str = match.groups()
            current_vol = float(vol_str)
            total_cost = cost
            total_duration = duration

            # 3. Look ahead for consecutive identical operations
            j = i + 1
            while j < n:
                next_op, next_cost, next_dur = path_by_flow[j]
                
                # Stop if next op is not Dosing
                if not next_op.strip().startswith("Dosing:"):
                    break
                
                next_match = pattern_dosing.search(next_op)
                if not next_match:
                    break
                
                n_src, n_dst, n_mat, n_vol_str = next_match.groups()
                
                # CRITICAL: Merge only if Source, Dest, and Material are identical
                if (n_src == src) and (n_dst == dst) and (n_mat == mat):
                    current_vol += float(n_vol_str)
                    total_cost += next_cost
                    total_duration += next_dur
                    j += 1
                else:
                    break
            
            # 4. Finalize the merged block
            # Only add to path if total volume is effectively > 0
            if current_vol > 1e-6:
                # Reconstruct the string with the new summed volume
                # We replace the volume part in the original string with the new total
                # Using regex sub to ensure we target the specific "number litre" pattern
                new_op_str = re.sub(
                    r":\s*[\d\.]+\s*litre", 
                    f": {current_vol:.1f} litre", 
                    op_str, 
                    count=1
                )
                merged_path.append((new_op_str, total_cost, total_duration))
            
            # Move index to the next unprocessed item
            i = j

        path_by_flow = merged_path
    
    # ==============================================================================
    # Processor D: Split End
    # ==============================================================================
    if split_end_step:
        final_split_path = []
        for op_str, cost, dur in path_by_flow:
            if op_str.strip().startswith("End, "):
                cleaned_op = op_str.replace("End, ", "", 1).strip()
                final_split_path.append((cleaned_op, cost, dur))
                final_split_path.append(("End", 0.0, 0.0))
            else:
                final_split_path.append((op_str, cost, dur))
        path_by_flow = final_split_path
    
    # ==============================================================================
    # Processor E: Final Format (force all volumes to 1 decimal place)
    # ==============================================================================
    final_formatted_path = []
    for op_str, cost, dur in path_by_flow:
        # Call regex-based replace helper
        new_op_str = _reformat_all_volumes(op_str)
        final_formatted_path.append((new_op_str, cost, dur))
    path_by_flow = final_formatted_path
    
    
    # 4. Build Result
    operation_rows = []
    for idx, (op_str, cost, duration) in enumerate(path_by_flow):
        operation_rows.append({
            "Step": idx + 1,
            "Operation": op_str,
            "Cost": cost,
            "Duration (s)": duration,
        })
    
    # 5. Display
    df_result = pd.DataFrame(operation_rows)
    tools.display_dataframe_to_user(name="Operation Sequence Table", dataframe=df_result)

# from pyvis.network import Network
# import webbrowser

# generate_picture = False

# if generate_picture:

#   net = Network(height="900px", width="100%", directed=True)

#   # Add nodes manually
#   for i in range(len(state_list)):
#       if i == 0:
#           net.add_node(i, label="", color="green", shape="star", size=300)
#       elif i == 1:
#           net.add_node(i, label="", color="blue", shape="star", size=300)
#       else:
#           net.add_node(i, label="", color="gray", shape="dot", size=10)

#   # Highlight optimal path edges
#   highlight_edges = set((entry[0], entry[1]) for entry in result_path)

#   # Add edges manually with width
#   for entry in transition_list:
#       if len(entry) < 4:
#           continue
#       src, _, _, dst = entry[:4]
#       if (src, dst) in highlight_edges:
#           net.add_edge(src, dst, color="red", width=80)
#       else:
#           net.add_edge(src, dst, color="rgba(200,200,200,0.2)", width=1)

#   # Set layout and render options
#   net.set_options("""
#   var options = {
#     "nodes": {
#       "font": {"size": 1}
#     },
#     "edges": {
#       "color": {"inherit": false},
#       "smooth": false
#     },
#     "physics": {
#       "enabled": false
#     },
#     "interaction": {
#       "hover": true,
#       "tooltipDelay": 100,
#       "zoomView": true,
#       "dragView": true
#     }
#   }
#   """)

#   # Save and open
#   net.write_html("interactive_state_path.html")
#   print("Graph saved to interactive_state_path.html")
#   # webbrowser.open("interactive_state_path.html")


# === Export operation sequence to structured LLM corpus ===
import json
import re
from pathlib import Path
import pandas as pd

step_lines = []
for row in operation_rows:
    try:
        step_lines.append(
            f"Step {int(row['Step'])} | Op: {row['Operation']} | Cost: {float(row['Cost']):.3f} | Dur: {float(row['Duration (s)']):.3f}"
        )
    except Exception:
        continue

save_corpus = os.getenv("WABEN_SAVE_CORPUS", "1").strip() == "1"
corpus_path = None
if save_corpus:
    if use_original_ModPlants:
        random_seed = 0
    corpus_path = Path(f'LLM/FeasibleTest/run-{random_seed}-operation_sequence_corpus.jsonl') if bfs_feasible else Path(f'LLM/UnfeasibleTest/run-{random_seed}-operation_sequence_corpus_fallback.jsonl')
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect static context
    corpus_context = {
        'ModPlant_ops': ModPlant_ops,
        'ModPlant_interfaces': ModPlant_interfaces,
        'ModPlant_maximum_volume': ModPlant_maximum_volume,
        'ModPlant_resources': ModPlant_resources,
        'order': order,
        'reaction_rules': reaction_rules_df.to_dict(orient='records'),
    }

    # Patterns for parsing
    _op_prefix_re = re.compile(r'^([A-Za-z]+)')
    _qty_re = re.compile(r'([A-Za-z0-9_]+):\s*([0-9.]+)\s*litre')
    _port_re = re.compile(r'Open Valve of ([A-Za-z0-9_]+)_Out\d+.*Filling\(([^)]+)\)')

    def normalize_op_type(op_str: str) -> str:
        m = _op_prefix_re.match(op_str.strip())
        if m:
            return m.group(1)
        return op_str.strip().split()[0]

    def parse_step(row):
        op_str = row['Operation']
        op_type = normalize_op_type(op_str)
        mats = [
            {'name': m, 'qty_l': float(q)}
            for m, q in _qty_re.findall(op_str)
        ]
        ports = {}
        pm = _port_re.search(op_str)
        if pm:
            ports = {'src': pm.group(1), 'dst': pm.group(2)}
        return {
            'role': 'step',
            'step_id': int(row['Step']),
            'op_type': op_type,
            'op_str': op_str,
            'materials': mats,
            'ports': ports,
            'cost': float(row['Cost']),
            'duration_s': float(row['Duration (s)']),
        }

    trajectory_id = f"run-{random_seed}-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"

    records = []
    records.append({'role': 'context', **corpus_context})
    records.append({'role': 'metadata', 'trajectory_id': trajectory_id, 'bfs_feasible': bfs_feasible})

    for row in operation_rows:
        records.append(parse_step(row))

    with corpus_path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Saved structured LLM corpus to {corpus_path} with {len(records)} records")
    tools.display_dataframe_to_user(name='LLM Corpus Preview', dataframe=pd.DataFrame(records))
else:
    print("Skipped LLM corpus export because WABEN_SAVE_CORPUS=0")



try:
    _res_feasible = locals().get('bfs_feasible', None)
    _res_path = locals().get('corpus_path', None)
    if _res_path: _res_path = str(_res_path)
    _res_step_lines = locals().get('step_lines', [])
    print(
        f"\n__WORKER_RESULT__:{json.dumps({'status': 'OK', 'feasible': _res_feasible, 'corpus_path': _res_path, 'step_lines': _res_step_lines}, ensure_ascii=False)}",
        flush=True
    )
except Exception as e:
    print(
        f"\n__WORKER_RESULT__:{json.dumps({'status': f'ERROR: {e}', 'feasible': None, 'corpus_path': None, 'step_lines': []})}",
        flush=True
    )
