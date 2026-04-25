#!/usr/bin/env python3
"""
Prepare local MLX runtime assets from a PEFT/Unsloth LoRA adapter directory.

What this script does:
1) Resolve selected adapter directory (supports selecting evaluation folder by mistake).
2) Read adapter_config.json to get base model name.
3) Download HF base model to project-local cache if missing.
4) Convert HF model to MLX 4-bit and cache it in the project.
5) Convert PEFT adapter weights to MLX adapter format and cache them.

All generated artifacts stay inside this project tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_BASE_MODEL = "unsloth/llama-3.2-3b-instruct-bnb-4bit"
DEFAULT_Q_BITS = 4


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().strip("/"))


def _is_peft_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()


def _resolve_adapter_dir(selected: Path, project_root: Path) -> Path:
    selected = selected.expanduser().resolve()
    if _is_peft_adapter_dir(selected):
        return selected

    # Common mistake: user selects evaluation/<run_name>; map to packaged model dirs.
    for base_dir in ("Model", "model-SOTA"):
        candidate = project_root / base_dir / selected.name
        if _is_peft_adapter_dir(candidate):
            return candidate.resolve()

    # If exactly one child adapter exists, use it.
    child_hits = []
    for child in selected.iterdir() if selected.exists() and selected.is_dir() else []:
        if child.is_dir() and _is_peft_adapter_dir(child):
            child_hits.append(child.resolve())
    if len(child_hits) == 1:
        return child_hits[0]

    # Last fallback: recursive search from selected.
    recursive_hits = []
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
        "Multiple adapter dirs found. Please pass --adapter-dir as an exact adapter folder:\n" + names
    )


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonicalize_repo_name(repo: str) -> str:
    repo = (repo or "").strip()
    low = repo.lower()

    # Pin llama-3.2-3b instruct family to the lightweight unsloth bnb repo.
    if low.startswith("unsloth/") and "llama-3.2-3b-instruct" in low:
        return DEFAULT_BASE_MODEL

    # Normalize unsloth naming variants.
    repo = re.sub(r"(?i)-unsloth(?=-bnb-4bit)", "", repo)
    return repo


def _expand_repo_variants(repo: str) -> List[str]:
    repo = (repo or "").strip()
    if not repo:
        return []
    out = []
    for v in [repo, _canonicalize_repo_name(repo)]:
        if v and v not in out:
            out.append(v)
        low = (v or "").lower()
        if low.startswith("unsloth/") and "bnb-4bit" not in low:
            vv = v + "-bnb-4bit"
            if vv not in out:
                out.append(vv)
    return out


def _non_bnb_repo_variants(repo: str) -> List[str]:
    repo = (repo or "").strip()
    if not repo:
        return []
    out = []

    def _add(v: str):
        if v and v not in out:
            out.append(v)

    base = re.sub(r"(?i)-bnb-4bit", "", repo)
    base = re.sub(r"(?i)_bnb-4bit", "", base)
    base = re.sub(r"(?i)\.bnb-4bit", "", base)
    base = re.sub(r"(?i)-4bit$", "", base)
    base = re.sub(r"(?i)_4bit$", "", base)
    _add(base)

    low = repo.lower()
    if "llama-3.2-3b-instruct" in low:
        _add("unsloth/llama-3.2-3b-instruct")
        _add("unsloth/Llama-3.2-3B-Instruct")
        _add("meta-llama/Llama-3.2-3B-Instruct")
    return out


def _build_repo_candidates(
    adapter_base: str,
    explicit_base: str | None,
    strict_default_base: bool = False,
) -> List[str]:
    if strict_default_base:
        return [DEFAULT_BASE_MODEL]

    # Priority is intentional:
    # 1) adapter_config base (ground truth for selected LoRA)
    # 2) pinned default bnb base
    # 3) explicit override (kept last to avoid stale notebook globals forcing huge downloads)
    candidates = []
    for source in [adapter_base, DEFAULT_BASE_MODEL, explicit_base or ""]:
        for v in _expand_repo_variants(source):
            if v and v not in candidates:
                candidates.append(v)

    # If any candidate maps to our pinned default, force it to the front.
    if DEFAULT_BASE_MODEL in candidates:
        candidates = [DEFAULT_BASE_MODEL] + [c for c in candidates if c != DEFAULT_BASE_MODEL]
    return candidates


def _is_hf_model_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    has_weights = bool(list(path.glob("*.safetensors"))) or (path / "model.safetensors.index.json").is_file()
    return has_weights


def _is_mlx_model_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    return bool(list(path.glob("*.safetensors")))


def _is_bnb_aux_key(key: str) -> bool:
    return (
        key.endswith(".absmax")
        or key.endswith(".quant_map")
        or ".nested_absmax" in key
        or ".nested_quant_map" in key
        or ".quant_state.bitsandbytes" in key
    )


def _checkpoint_weight_shape_is_packed(hf_dir: Path) -> bool:
    try:
        from safetensors import safe_open
    except Exception:
        return False

    for sf in sorted(hf_dir.glob("*.safetensors")):
        try:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if not k.endswith(".weight"):
                        continue
                    t = f.get_tensor(k)
                    if len(t.shape) == 2 and t.shape[1] == 1:
                        return True
                    # only inspect a handful tensors per file
                    break
        except Exception:
            continue
    return False


def _hf_checkpoint_has_bnb_aux(hf_dir: Path) -> bool:
    try:
        from safetensors import safe_open
    except Exception:
        return False

    for sf in sorted(hf_dir.glob("*.safetensors")):
        try:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if _is_bnb_aux_key(k):
                        return True
        except Exception:
            continue
    return False


def _make_sanitized_hf_dir(hf_dir: Path) -> Path:
    """
    Create a local copy of the HF checkpoint with bitsandbytes aux tensors removed.
    """
    try:
        from safetensors.torch import load_file, save_file
    except Exception as e:
        raise RuntimeError("Missing safetensors for bnb-checkpoint sanitization.") from e

    out_dir = hf_dir.parent / f"{hf_dir.name}__mlx_sanitized"
    marker = out_dir / ".sanitized_ok"
    if marker.is_file() and (out_dir / "config.json").is_file() and bool(list(out_dir.glob("*.safetensors"))):
        return out_dir

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in hf_dir.iterdir():
        if src.name.endswith(".safetensors"):
            continue
        if src.is_file():
            shutil.copy2(src, out_dir / src.name)
        elif src.is_dir() and src.name != ".cache":
            shutil.copytree(src, out_dir / src.name)

    for src in sorted(hf_dir.glob("*.safetensors")):
        tensors = load_file(str(src))
        filtered = {k: v for k, v in tensors.items() if not _is_bnb_aux_key(k)}
        if not filtered:
            raise RuntimeError(f"No non-aux tensors left after sanitization: {src}")
        save_file(filtered, str(out_dir / src.name))

    idx_src = hf_dir / "model.safetensors.index.json"
    idx_dst = out_dir / "model.safetensors.index.json"
    if idx_src.is_file():
        idx = _read_json(idx_src)
        wm = idx.get("weight_map", {})
        wm2 = {k: v for k, v in wm.items() if not _is_bnb_aux_key(k)}
        idx["weight_map"] = wm2
        if "metadata" in idx and isinstance(idx["metadata"], dict):
            idx["metadata"]["total_size"] = int(
                sum((out_dir / n).stat().st_size for n in set(wm2.values()) if (out_dir / n).exists())
            )
        idx_dst.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

    marker.write_text("ok\n", encoding="utf-8")
    return out_dir


def _run_mlx_convert(hf_dir: Path, out_dir: Path, python_bin: str, q_bits: int) -> Tuple[int, str, str]:
    cmd = [
        python_bin,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        str(hf_dir),
        "--mlx-path",
        str(out_dir),
        "-q",
        "--q-bits",
        str(q_bits),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _download_hf_model(
    repo_candidates: List[str],
    hf_root: Path,
    dry_run: bool,
) -> Tuple[str, Path, bool]:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError("Missing dependency huggingface_hub. Install it in the runtime environment.") from e

    errors = []
    for repo in repo_candidates:
        local_dir = hf_root / _sanitize_name(repo)
        if _is_hf_model_ready(local_dir):
            return repo, local_dir, True
        if dry_run:
            return repo, local_dir, False
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo,
                local_dir=str(local_dir),
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.jinja",
                    "*.model",
                    "*.tiktoken",
                ],
                ignore_patterns=["*.onnx", "*.ot", "*.h5", "*.msgpack", "*.gguf"],
            )
            if _is_hf_model_ready(local_dir):
                return repo, local_dir, False
            errors.append(f"{repo}: downloaded but model files incomplete")
        except Exception as e:
            errors.append(f"{repo}: {e}")
    raise RuntimeError("All HF model candidates failed:\n" + "\n".join(errors))


def _convert_hf_to_mlx(
    hf_dir: Path,
    repo_name: str,
    mlx_root: Path,
    python_bin: str,
    q_bits: int,
    dry_run: bool,
) -> Tuple[Path, bool]:
    out_dir = mlx_root / f"{_sanitize_name(repo_name)}_q{q_bits}"
    if _is_mlx_model_ready(out_dir):
        return out_dir, True
    if dry_run:
        return out_dir, False

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # bnb checkpoints are packed and cannot be converted directly by mlx_lm.convert.
    if _hf_checkpoint_has_bnb_aux(hf_dir) or _checkpoint_weight_shape_is_packed(hf_dir):
        raise RuntimeError(
            "HF source appears to be packed bnb weights and cannot be directly converted to MLX. "
            f"source={hf_dir}"
        )

    rc, out, err = _run_mlx_convert(hf_dir, out_dir, python_bin, q_bits)
    if rc == 0 and _is_mlx_model_ready(out_dir):
        return out_dir, False

    err_low = (err or "").lower()
    need_sanitize = _hf_checkpoint_has_bnb_aux(hf_dir) or ("bitsandbytes" in err_low) or ("quant_state" in err_low)
    if need_sanitize:
        sanitized_hf = _make_sanitized_hf_dir(hf_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        rc2, out2, err2 = _run_mlx_convert(sanitized_hf, out_dir, python_bin, q_bits)
        if rc2 == 0 and _is_mlx_model_ready(out_dir):
            return out_dir, False
        raise RuntimeError(
            "MLX conversion failed (original + sanitized retry).\n"
            f"original_hf={hf_dir}\n"
            f"sanitized_hf={sanitized_hf}\n"
            f"original_stdout:\n{out[-3000:]}\n"
            f"original_stderr:\n{err[-3000:]}\n"
            f"retry_stdout:\n{out2[-3000:]}\n"
            f"retry_stderr:\n{err2[-3000:]}"
        )

    raise RuntimeError(
        "MLX conversion failed.\n"
        f"hf={hf_dir}\n"
        f"stdout:\n{out[-4000:]}\n"
        f"stderr:\n{err[-4000:]}"
    )
    return out_dir, False


def _convert_with_source_fallback(
    selected_repo: str,
    selected_hf_dir: Path,
    hf_root: Path,
    mlx_root: Path,
    python_bin: str,
    q_bits: int,
    dry_run: bool,
) -> Tuple[Path, bool, str, Path]:
    """
    Convert to MLX using selected repo first; if source is packed bnb, try non-bnb source repos.
    """
    try:
        mlx_dir, cached = _convert_hf_to_mlx(
            hf_dir=selected_hf_dir,
            repo_name=selected_repo,
            mlx_root=mlx_root,
            python_bin=python_bin,
            q_bits=q_bits,
            dry_run=dry_run,
        )
        return mlx_dir, cached, selected_repo, selected_hf_dir
    except Exception as first_err:
        alt_repos = [r for r in _non_bnb_repo_variants(selected_repo) if r != selected_repo]
        alt_errors = [f"primary({selected_repo}): {first_err}"]
        for alt in alt_repos:
            try:
                alt_repo, alt_hf_dir, _ = _download_hf_model([alt], hf_root, dry_run)
                mlx_dir, cached = _convert_hf_to_mlx(
                    hf_dir=alt_hf_dir,
                    repo_name=selected_repo,  # keep output dir keyed by selected repo
                    mlx_root=mlx_root,
                    python_bin=python_bin,
                    q_bits=q_bits,
                    dry_run=dry_run,
                )
                return mlx_dir, cached, alt_repo, alt_hf_dir
            except Exception as e:
                alt_errors.append(f"fallback({alt}): {e}")
        raise RuntimeError("All conversion source attempts failed:\n" + "\n".join(alt_errors))


def _infer_num_layers(config_path: Path) -> int:
    cfg = _read_json(config_path)
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if key in cfg:
            return int(cfg[key])
    raise KeyError(f"Cannot infer num_layers from: {config_path}")


def _peft_key_to_mlx_key(peft_key: str) -> Tuple[str | None, bool]:
    k = peft_key
    if k.startswith("base_model.model."):
        k = k[len("base_model.model.") :]

    if k.endswith(".lora_A.weight"):
        return k[: -len(".lora_A.weight")] + ".lora_a", True
    if k.endswith(".lora_B.weight"):
        return k[: -len(".lora_B.weight")] + ".lora_b", True
    return None, False


def _build_adapter_cache_key(adapter_dir: Path, repo_name: str, q_bits: int) -> str:
    src_cfg = adapter_dir / "adapter_config.json"
    src_w = adapter_dir / "adapter_model.safetensors"
    stat = src_w.stat()
    raw = "|".join(
        [
            str(adapter_dir.resolve()),
            str(repo_name),
            str(q_bits),
            hashlib.sha1(src_cfg.read_bytes()).hexdigest(),
            str(stat.st_size),
            str(int(stat.st_mtime)),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _convert_peft_adapter_to_mlx(
    adapter_dir: Path,
    repo_name: str,
    mlx_model_dir: Path,
    hf_model_dir: Path,
    adapters_root: Path,
    dry_run: bool,
) -> Tuple[Path, bool]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except Exception as e:
        raise RuntimeError("Missing dependencies torch + safetensors for adapter conversion.") from e

    cache_key = _build_adapter_cache_key(adapter_dir, repo_name, DEFAULT_Q_BITS)
    out_dir = adapters_root / f"{_sanitize_name(adapter_dir.name)}__{cache_key}"
    out_cfg = out_dir / "adapter_config.json"
    out_w = out_dir / "adapters.safetensors"
    meta_path = out_dir / "meta.json"

    if out_cfg.is_file() and out_w.is_file() and meta_path.is_file():
        return out_dir, True
    if dry_run:
        return out_dir, False

    out_dir.mkdir(parents=True, exist_ok=True)

    src_cfg = _read_json(adapter_dir / "adapter_config.json")
    src_weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    converted = {}

    for key, tensor in src_weights.items():
        mapped_key, need_t = _peft_key_to_mlx_key(key)
        if not mapped_key:
            continue
        t = tensor.transpose(0, 1).contiguous() if need_t else tensor
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            t = t.to(torch.float16)
        converted[mapped_key] = t

    if not converted:
        raise RuntimeError("No LoRA tensors were converted from adapter_model.safetensors.")

    # MLX tuner expects num_layers and lora_parameters schema.
    num_layers = _infer_num_layers(hf_model_dir / "config.json")
    rank = int(src_cfg.get("r", 8))
    alpha = float(src_cfg.get("lora_alpha", rank))
    scale = (alpha / rank) if rank > 0 else alpha
    dropout = float(src_cfg.get("lora_dropout", 0.0))

    mlx_cfg = {
        "model": str(mlx_model_dir),
        "num_layers": num_layers,
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": dropout,
        },
    }

    save_file(converted, str(out_w))
    out_cfg.write_text(json.dumps(mlx_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "source_adapter_dir": str(adapter_dir.resolve()),
        "source_base_model": src_cfg.get("base_model_name_or_path"),
        "selected_hf_repo": repo_name,
        "mlx_model_dir": str(mlx_model_dir.resolve()),
        "converted_tensor_count": len(converted),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local MLX model + adapter cache from PEFT LoRA.")
    parser.add_argument("--adapter-dir", required=True, help="Selected adapter folder or its parent.")
    parser.add_argument("--base-model", default=None, help="Optional base model override.")
    parser.add_argument("--project-root", default=".", help="Project root for local caches.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used for mlx_lm.convert.")
    parser.add_argument("--q-bits", type=int, default=DEFAULT_Q_BITS, help="Quantization bits for MLX conversion.")
    parser.add_argument(
        "--strict-default-base",
        action="store_true",
        help=f"Only allow downloading/converting the pinned default base: {DEFAULT_BASE_MODEL}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve and plan only; do not download/convert.")
    parser.add_argument("--output", default=None, help="Optional path to write json result.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    hf_root = project_root / "models_hf"
    mlx_base_root = project_root / "models_mlx" / "base_models"
    mlx_adapter_root = project_root / "models_mlx" / "adapters"
    hf_root.mkdir(parents=True, exist_ok=True)
    mlx_base_root.mkdir(parents=True, exist_ok=True)
    mlx_adapter_root.mkdir(parents=True, exist_ok=True)

    adapter_dir = _resolve_adapter_dir(Path(args.adapter_dir), project_root)
    adapter_cfg = _read_json(adapter_dir / "adapter_config.json")
    adapter_base = adapter_cfg.get("base_model_name_or_path", "")
    repo_candidates = _build_repo_candidates(
        adapter_base,
        args.base_model,
        strict_default_base=bool(args.strict_default_base),
    )

    selected_repo, hf_dir, hf_cached = _download_hf_model(repo_candidates, hf_root, args.dry_run)
    mlx_model_dir, mlx_cached, convert_source_repo, convert_source_hf_dir = _convert_with_source_fallback(
        selected_repo=selected_repo,
        selected_hf_dir=hf_dir,
        hf_root=hf_root,
        mlx_root=mlx_base_root,
        python_bin=args.python_bin,
        q_bits=args.q_bits,
        dry_run=args.dry_run,
    )
    mlx_adapter_dir, mlx_adapter_cached = _convert_peft_adapter_to_mlx(
        adapter_dir=adapter_dir,
        repo_name=selected_repo,
        mlx_model_dir=mlx_model_dir,
        hf_model_dir=convert_source_hf_dir,
        adapters_root=mlx_adapter_root,
        dry_run=args.dry_run,
    )

    result = {
        "status": "ok",
        "dry_run": bool(args.dry_run),
        "adapter_dir": str(adapter_dir),
        "adapter_base_model": adapter_base,
        "repo_candidates": repo_candidates,
        "selected_hf_repo": selected_repo,
        "hf_model_dir": str(hf_dir),
        "convert_source_repo": convert_source_repo,
        "convert_source_hf_dir": str(convert_source_hf_dir),
        "mlx_model_dir": str(mlx_model_dir),
        "mlx_adapter_dir": str(mlx_adapter_dir),
        "cache_hit": {
            "hf_model": bool(hf_cached),
            "mlx_model": bool(mlx_cached),
            "mlx_adapter": bool(mlx_adapter_cached),
        },
    }

    out_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(out_text, encoding="utf-8")
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
