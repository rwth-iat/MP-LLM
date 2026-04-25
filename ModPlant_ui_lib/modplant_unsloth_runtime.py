#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ModPlant_ui_lib.runtime_preload import preload_runtime_libs


_RUNTIME_LOCK = threading.Lock()
_RUNTIME_CACHE: dict[tuple[str, str, int, bool], tuple[Any, Any, dict[str, Any]]] = {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_peft_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()


def resolve_adapter_dir(selected: Path, project_root: Path) -> Path:
    selected = selected.expanduser()
    if not selected.is_absolute():
        selected = project_root / selected
    selected = selected.resolve()
    if _is_peft_adapter_dir(selected):
        return selected

    for base_dir in ("Model", "model-SOTA"):
        candidate = project_root / base_dir / selected.name
        if _is_peft_adapter_dir(candidate):
            return candidate.resolve()

    child_hits: list[Path] = []
    for child in selected.iterdir() if selected.exists() and selected.is_dir() else []:
        if child.is_dir() and _is_peft_adapter_dir(child):
            child_hits.append(child.resolve())
    if len(child_hits) == 1:
        return child_hits[0]

    recursive_hits: list[Path] = []
    if selected.exists() and selected.is_dir():
        for cfg in selected.rglob("adapter_config.json"):
            parent = cfg.parent
            if _is_peft_adapter_dir(parent):
                recursive_hits.append(parent.resolve())
    recursive_hits = sorted(set(recursive_hits))
    if len(recursive_hits) == 1:
        return recursive_hits[0]

    if not recursive_hits:
        raise FileNotFoundError(
            f"No PEFT adapter dir found under: {selected}. "
            "Expected adapter_config.json + adapter_model.safetensors."
        )
    names = "\n".join(str(p) for p in recursive_hits[:20])
    raise RuntimeError(
        "Multiple adapter dirs found. Please pass/select an exact adapter folder:\n" + names
    )


def resolve_base_model_name(adapter_dir: Path, default_base_model: str) -> str:
    cfg_path = adapter_dir / "adapter_config.json"
    try:
        cfg = _read_json(cfg_path)
    except Exception:
        return default_base_model
    base_model_name = str(cfg.get("base_model_name_or_path") or "").strip()
    return base_model_name or default_base_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def bootstrap_unsloth() -> None:
    preload_runtime_libs()
    import unsloth  # noqa: F401


def _detect_model_device(model: Any) -> str:
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    try:
        first_param = next(model.parameters())
        return str(first_param.device)
    except Exception:
        return "unknown"


def load_runtime(
    project_root: Path,
    adapter_dir: Path,
    default_base_model: str,
    max_seq_length: int,
    load_in_4bit: bool,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    project_root = project_root.resolve()
    resolved_adapter_dir = resolve_adapter_dir(adapter_dir, project_root)
    base_model_name = resolve_base_model_name(resolved_adapter_dir, default_base_model)
    cache_key = (
        str(resolved_adapter_dir),
        base_model_name,
        int(max_seq_length),
        bool(load_in_4bit),
    )

    with _RUNTIME_LOCK:
        cached = _RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            metadata = dict(cached[2])
            metadata["runtime_cache_hit"] = True
            return cached[0], cached[1], metadata

        seed_everything(seed)

        preload_runtime_libs()
        import unsloth  # noqa: F401  # Keep before any transformers import.
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=int(max_seq_length),
            dtype=None,
            load_in_4bit=bool(load_in_4bit),
        )
        try:
            model.load_adapter(str(resolved_adapter_dir))
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to load LoRA adapter due to model mismatch. "
                f"Adapter dir: {resolved_adapter_dir}. Loaded base model: {base_model_name}. "
                "Please ensure adapter_config.json base_model_name_or_path matches the loaded base model.\n"
                f"Original error: {e}"
            ) from e

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        FastLanguageModel.for_inference(model)
        model.eval()

        metadata = {
            "adapter_dir": str(resolved_adapter_dir),
            "base_model_name": base_model_name,
            "device": _detect_model_device(model),
            "load_in_4bit": bool(load_in_4bit),
            "max_seq_length": int(max_seq_length),
            "runtime_cache_hit": False,
        }
        _RUNTIME_CACHE[cache_key] = (model, tokenizer, dict(metadata))
        return model, tokenizer, metadata


def prepare_runtime(
    project_root: Path,
    adapter_dir: Path,
    default_base_model: str,
    max_seq_length: int,
    load_in_4bit: bool,
    seed: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    _, _, metadata = load_runtime(
        project_root=project_root,
        adapter_dir=adapter_dir,
        default_base_model=default_base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        seed=seed,
    )
    out = dict(metadata)
    out["elapsed_s"] = time.perf_counter() - start
    return out


def run_stepwise_inference(
    context_str: str,
    project_root: Path,
    adapter_dir: Path,
    default_base_model: str,
    build_prompt_cb: Callable[[str, list[str]], str],
    extract_first_step_line_cb: Callable[[str], str | None],
    max_steps: int,
    max_seq_length: int,
    max_new_tokens: int,
    load_in_4bit: bool,
    temperature: float,
    top_p: float,
    attempt_timeout_s: float,
    seed: int,
    stop_cb: Callable[[], bool] | None = None,
    progress_cb: Callable[[int, str, float], None] | None = None,
    step_validator_cb: Callable[[int, str, float, list[str]], bool] | None = None,
) -> tuple[str, float, dict[str, Any], bool, bool]:
    model, tokenizer, metadata = load_runtime(
        project_root=project_root,
        adapter_dir=adapter_dir,
        default_base_model=default_base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        seed=seed,
    )

    seed_everything(seed)

    import torch

    device = _detect_model_device(model)
    past_steps: list[str] = []
    total_latency = 0.0
    timeout_limit = max(0.0, float(attempt_timeout_s))
    timed_out = False
    stopped = False

    for step_idx in range(1, int(max_steps) + 1):
        if stop_cb is not None and stop_cb():
            stopped = True
            break
        if timeout_limit > 0.0 and total_latency >= timeout_limit:
            timed_out = True
            break

        prompt = build_prompt_cb(context_str, past_steps)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
        }
        if float(temperature) > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**gen_kwargs)
        elapsed = time.perf_counter() - t0
        total_latency += elapsed

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][prompt_len:]
        if generated_ids.numel() == 0:
            break

        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        eos_token = tokenizer.eos_token or ""
        if eos_token:
            text = text.replace(eos_token, "")
        text = text.strip()

        first_line = extract_first_step_line_cb(text) or ""
        if not first_line:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    first_line = line
                    break
        if not first_line:
            break

        past_steps.append(first_line)
        if progress_cb is not None:
            progress_cb(step_idx, first_line, total_latency)
        if step_validator_cb is not None:
            should_continue = step_validator_cb(step_idx, first_line, total_latency, past_steps)
            if not should_continue:
                stopped = True
                break
        if "op: end" in first_line.lower():
            break

    out_metadata = dict(metadata)
    out_metadata["device"] = device
    return "\n".join(past_steps), total_latency, out_metadata, timed_out, stopped
