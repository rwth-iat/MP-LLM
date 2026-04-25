#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .FSA.Waben_Flow_Generator import build_schedule
from .FSA.Waben_Flow_To_General_Recipe import save_general_recipe_xml
from .FSA.Waben_General_Recipe_To_Json import save_parsed_recipe_json_by_id
from .FSA.Waben_Reaction_Rules import generate_reaction_rules_from_general_recipe_json, rules_to_dataframe
from .ModPlant_fsa_checker_core import extract_first_step_line
from .modplant_unsloth_runtime import (
    run_stepwise_inference as run_unsloth_stepwise_inference,
    seed_everything as seed_runtime,
)

DEFAULT_BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"
DEFAULT_LOAD_IN_4BIT = True
DEFAULT_LLM_MAX_SEQ_LENGTH = 2560
DEFAULT_LLM_MAX_NEW_TOKENS = 64
DEFAULT_MAX_STEPS = 30
DEFAULT_LLM_BASE_TEMPERATURE = float(os.environ.get("WABEN_LLM_BASE_TEMPERATURE", "0.0"))
DEFAULT_LLM_BASE_TOP_P = float(os.environ.get("WABEN_LLM_BASE_TOP_P", "1.0"))
DEFAULT_LLM_ATTEMPT_TIMEOUT_S = int(os.environ.get("WABEN_LLM_ATTEMPT_TIMEOUT_S", "500"))
DEFAULT_PERSIST_TEMP_FILES = False
DEFAULT_UI_SEED = 14
CODE_SUBDIR_NAME = "ModPlant_ui_lib"
DEFAULT_ADAPTER_REL_PATH = "Model/3B-20260226_224018"

FIXED_INSTRUCTION = (
    "You are an industrial automation planning system.\n"
    "Given the system configuration (Input JSON) and the history of executed operations [Past Steps], "
    "predict ONLY the exact SINGLE next step required to fulfill the order.\n\n"
    "Format your output exactly as:\n"
    "Step N | Op: ... | Cost: ... | Dur: ...\n\n"
    "If the order is completely fulfilled and no further actions are needed, output:\n"
    "Step N | Op: End | Cost: 0.000 | Dur: 0.000\n\n"
    "Do NOT output any extra keys, JSON, reasoning text, or multiple steps."
)

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

def get_code_root(project_root: Path) -> Path:
    code_root = project_root / CODE_SUBDIR_NAME
    if code_root.is_dir():
        return code_root.resolve()
    return project_root.resolve()

def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

def format_step_record(step: dict[str, Any]) -> str:
    return (
        f"Step {int(step.get('step_id', 0))} | "
        f"Op: {step.get('op_str', '')} | "
        f"Cost: {float(step.get('cost', 0.0)):.3f} | "
        f"Dur: {float(step.get('duration_s', 0.0)):.3f}"
    )

def build_step_prompt(context_str: str, past_steps: list[str]) -> str:
    if not past_steps:
        past_str = "None"
    else:
        past_str = "\n".join(past_steps)
    combined_input = f"{context_str}\n\n[Past Steps]\n{past_str}\n\n[Next Step Prediction]"
    return PROMPT_TEMPLATE.format(instruction=FIXED_INSTRUCTION, input=combined_input)

def run_llm_autoregressive(
    context_str: str,
    project_root: Path,
    adapter_dir: Path,
    max_steps: int,
    seed: int,
    temperature: float = DEFAULT_LLM_BASE_TEMPERATURE,
    top_p: float = DEFAULT_LLM_BASE_TOP_P,
    attempt_timeout_s: float = float(DEFAULT_LLM_ATTEMPT_TIMEOUT_S),
    stop_cb: Callable[[], bool] | None = None,
    progress_cb: Callable[[int, str, float], None] | None = None,
    step_validator_cb: Callable[[int, str, float, list[str]], bool] | None = None,
) -> tuple[str, float, dict[str, Any], bool, bool]:
    return run_unsloth_stepwise_inference(
        context_str=context_str,
        project_root=project_root,
        adapter_dir=adapter_dir,
        default_base_model=DEFAULT_BASE_MODEL,
        build_prompt_cb=build_step_prompt,
        extract_first_step_line_cb=extract_first_step_line,
        max_steps=max_steps,
        max_seq_length=DEFAULT_LLM_MAX_SEQ_LENGTH,
        max_new_tokens=DEFAULT_LLM_MAX_NEW_TOKENS,
        load_in_4bit=DEFAULT_LOAD_IN_4BIT,
        temperature=temperature,
        top_p=top_p,
        attempt_timeout_s=attempt_timeout_s,
        seed=seed,
        stop_cb=stop_cb,
        progress_cb=progress_cb,
        step_validator_cb=step_validator_cb,
    )

def generate_random_ModPlants(seed: int, min_n: int, max_n: int) -> tuple[dict, dict, dict, dict]:
    random.seed(seed)
    n = random.randint(max(4, min_n), min(6, max_n))
    names = [f"HC{(i + 1) * 10}" for i in range(n)]

    w_ops: dict[str, list[tuple[str, Any, int]]] = {}
    w_ifaces: dict[str, list[tuple[str, str]]] = {}
    w_cap: dict[str, list[int]] = {}

    settling_assigned = False
    stirring_assigned = False
    rate_choices = [0.1, 0.2, 0.3]
    rpm_choices = list(range(50, 301, 50))

    for name in names:
        num_inputs = random.randint(1, 4)
        num_outputs = random.randint(1, 4)
        w_ifaces[name] = [("Input", f"{name}_In{i + 1}") for i in range(num_inputs)] + [
            ("Output", f"{name}_Out{i + 1}") for i in range(num_outputs)
        ]

        max_vol = random.choice(range(10, 31, 5))
        w_cap[name] = [max_vol]

        drain_rate = random.choice(rate_choices)
        fill_rate = random.choice(rate_choices)
        ops: list[tuple[str, Any, int]] = [
            ("Draining", drain_rate, 3),
            ("Filling", fill_rate, 0),
            ("Connect", "", 1),
            ("Disconnect", "", 0),
            ("None", "", 0),
        ]

        if (not settling_assigned) or random.random() < 0.5:
            ops.append(("Settling", "", random.randint(1, 3)))
            settling_assigned = True

        stir_count = random.randint(1, 2)
        rpms = random.sample(rpm_choices, stir_count)
        for rpm in rpms:
            ops.append(("Stirring", str(rpm), random.randint(1, 4)))
        stirring_assigned = stirring_assigned or bool(rpms)

        w_ops[name] = ops

    if not settling_assigned:
        name = names[0]
        w_ops[name].append(("Settling", "", random.randint(1, 3)))
    if not stirring_assigned:
        name = names[0]
        rpm = random.choice(rpm_choices)
        w_ops[name].append(("Stirring", str(rpm), random.randint(1, 4)))

    resources: dict[str, list[Any]] = {}
    chosen = random.sample(names, k=min(3, len(names)))
    for mat, wb in zip(["A", "B", "C"], chosen):
        cap = w_cap[wb][0]
        qty = min(random.randint(5, 15), cap)
        resources[wb] = [mat, qty]

    return w_ops, w_ifaces, w_cap, resources

def build_random_order(seed: int) -> dict[str, Any]:
    random.seed(seed)
    order_volume = 10
    letters = ["A", "B", "C"]
    random.shuffle(letters)
    rpm_choices = list(range(50, 301, 50))
    order_list = letters.copy()
    order_list.append(
        {
            "mix": {
                "rpm": random.choice(rpm_choices),
                "duration": random.randrange(100, 1001, 100),
            }
        }
    )

    parts = [random.randint(1, 8) for _ in letters]
    scale = 10 / sum(parts)
    ratio_vals = [max(1, round(v * scale)) for v in parts]
    ratio_vals[0] += 10 - sum(ratio_vals)
    ratio = {k: [v] for k, v in zip(letters, ratio_vals)}

    usage_and_settling = [random.randrange(100, 5001, 100), random.randrange(100, 1001, 100)]
    return {
        "volume": float(order_volume),
        "order": order_list,
        "ratio": ratio,
        "usage_and_settling": usage_and_settling,
    }

def build_context_from_seed(seed: int, persist_temp_files: bool = DEFAULT_PERSIST_TEMP_FILES) -> dict[str, Any]:
    seed_runtime(seed)
    ModPlant_ops, ModPlant_interfaces, ModPlant_maximum_volume, ModPlant_resources = generate_random_ModPlants(
        seed=seed,
        min_n=4,
        max_n=4,
    )
    order = build_random_order(seed)

    schedule = build_schedule(order)
    if persist_temp_files:
        xml_path = save_general_recipe_xml(schedule, order)
        json_path = save_parsed_recipe_json_by_id(xml_path)
        reaction_rules = generate_reaction_rules_from_general_recipe_json(json_path)
        xml_path_display = xml_path
        json_path_display = json_path
    else:
        with tempfile.TemporaryDirectory(prefix="ModPlant_recipe_") as tmp_dir:
            xml_path = save_general_recipe_xml(schedule, order, out_dir=tmp_dir)
            json_path = save_parsed_recipe_json_by_id(xml_path, out_dir=tmp_dir)
            reaction_rules = generate_reaction_rules_from_general_recipe_json(json_path)
        xml_path_display = "(disabled: temporary only)"
        json_path_display = "(disabled: temporary only)"

    reaction_rules_df = rules_to_dataframe(reaction_rules)
    reaction_rules_records = reaction_rules_df.to_dict(orient="records")

    input_context = {
        "ModPlant_ops": ModPlant_ops,
        "ModPlant_interfaces": ModPlant_interfaces,
        "ModPlant_resources": ModPlant_resources,
        "ModPlant_maximum_volume": ModPlant_maximum_volume,
        "reaction_rules": reaction_rules_records,
    }
    return {
        "seed": seed,
        "input_context": input_context,
        "order": order,
        "schedule": schedule,
        "reaction_rules": reaction_rules_records,
        "xml_path": xml_path_display,
        "json_path": json_path_display,
        "persist_temp_files": persist_temp_files,
    }

def run_fsa_solver_pipeline(
    project_root: Path,
    fsa_python: str | None,
    seed: int,
    persist_recipe_files: bool = True,
    save_corpus_jsonl: bool = True,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    code_root = get_code_root(project_root)
    script_path = code_root / "FSA" / "_temp_worker_script_hpc.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing FSA+BFS+OPT worker script: {script_path}")

    env = os.environ.copy()
    env["WABEN_SEED"] = str(seed)
    env["WABEN_PERSIST_RECIPE_FILES"] = "1" if persist_recipe_files else "0"
    env["WABEN_SAVE_CORPUS"] = "1" if save_corpus_jsonl else "0"
    # Ensure worker can import project modules like `from FSA...`
    existing_pp = env.get("PYTHONPATH", "")
    python_paths = [str(code_root), str(project_root)]
    if existing_pp:
        python_paths.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    python_bin = fsa_python or sys.executable
    cmd = [python_bin, str(script_path)]

    start = time.perf_counter()
    p = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    worker_result: dict[str, Any] | None = None
    try:
        assert p.stdout is not None
        for line in p.stdout:
            clean = line.rstrip("\n")
            lines.append(clean)
            if log_cb is not None:
                log_cb(clean)
            if "__WORKER_RESULT__:" in clean:
                raw = clean.split("__WORKER_RESULT__:", 1)[1].strip()
                try:
                    worker_result = json.loads(raw)
                except Exception:
                    worker_result = {"status": "ERROR", "raw": raw}
    finally:
        ret = p.wait()
    elapsed = time.perf_counter() - start

    if ret != 0:
        tail = "\n".join(lines[-80:])
        raise RuntimeError(
            "FSA+BFS+OPT worker failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"returncode={ret}\n"
            f"tail:\n{tail}"
        )

    if not worker_result:
        tail = "\n".join(lines[-80:])
        raise RuntimeError(f"FSA+BFS+OPT worker finished but no __WORKER_RESULT__ found.\n{tail}")

    corpus_path = worker_result.get("corpus_path")
    if corpus_path:
        cp = Path(corpus_path)
        if not cp.is_absolute():
            cp = (project_root / cp).resolve()
        corpus_path = str(cp)

    return {
        "elapsed_s": elapsed,
        "worker_result": worker_result,
        "corpus_path": corpus_path,
        "log_tail": "\n".join(lines[-120:]),
    }

def load_fsa_reference_steps(corpus_path: str) -> list[str]:
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("role") == "step":
                out.append(format_step_record(rec))
    return out

__all__ = [
    "DEFAULT_ADAPTER_REL_PATH",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_LOAD_IN_4BIT",
    "DEFAULT_LLM_ATTEMPT_TIMEOUT_S",
    "DEFAULT_LLM_BASE_TEMPERATURE",
    "DEFAULT_LLM_BASE_TOP_P",
    "DEFAULT_LLM_MAX_NEW_TOKENS",
    "DEFAULT_LLM_MAX_SEQ_LENGTH",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_PERSIST_TEMP_FILES",
    "DEFAULT_UI_SEED",
    "build_context_from_seed",
    "compact_json",
    "format_step_record",
    "get_code_root",
    "load_fsa_reference_steps",
    "run_fsa_solver_pipeline",
    "run_llm_autoregressive",
]
