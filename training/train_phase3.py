#!/usr/bin/env python3
"""Phase 3 training script with progressive unfreezing and ultra-low LR.

Addresses catastrophic forgetting seen in a7/a10 runs (peaked epoch 2-3 then
declined) by supporting three strategies:

  two_stage      -- Freeze backbone 30 epochs, then unfreeze with ultra-low LR
  cosine_low_lr  -- Single stage with very low LR + long warmup + cosine schedule
  from_pretrained-- Start from YOLO pretrained weights (ignores base-model path)

Supports both YOLO12s and YOLO26s via --arch.

Usage examples:
    # Two-stage from Ultimate v5 base
    python train_phase3.py --strategy two_stage --arch yolo12s \\
        --base-model harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt \\
        --data datasets/yolo_harmony_v2_phase8_cleaned/harmony_phase8_balanced.yaml

    # Cosine low-LR from YOLO26s pretrained
    python train_phase3.py --strategy cosine_low_lr --arch yolo26s \\
        --base-model yolo26s.pt \\
        --data datasets/yolo_harmony_v2_phase8_cleaned/harmony_phase8_balanced.yaml

    # From pretrained (arch determines which .pt is downloaded)
    python train_phase3.py --strategy from_pretrained --arch yolo12s \\
        --data datasets/yolo_harmony_v2_phase8_cleaned/harmony_phase8_balanced.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

STRATEGIES = ("two_stage", "cosine_low_lr", "from_pretrained")
ARCHS = ("yolo12s", "yolo26s")

PRETRAINED_WEIGHTS: dict[str, str] = {
    "yolo12s": "yolo12s.pt",
    "yolo26s": "yolo26s.pt",
}

DEFAULT_BASE_MODELS: dict[str, str] = {
    "yolo12s": (
        "harmony_omr_v2_ultimate_v5_stable/stable_1280_resumed/weights/best.pt"
    ),
    "yolo26s": (
        "../runs/detect/harmony_omr_v2_yolo26/"
        "yolo26s_official_e250_w12_b4_amp0_m05_a7/weights/best.pt"
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3: progressive unfreezing / ultra-low-LR fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- identity ---
    p.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="two_stage",
        help="Training strategy (default: two_stage)",
    )
    p.add_argument(
        "--arch",
        choices=ARCHS,
        default="yolo12s",
        help="Model architecture (default: yolo12s)",
    )
    p.add_argument(
        "--base-model",
        default=None,
        help=(
            "Path to starting weights. Defaults to Ultimate v5 for yolo12s or "
            "a7 for yolo26s. Ignored when --strategy from_pretrained."
        ),
    )
    p.add_argument(
        "--data",
        default="datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml",
        help="Path to dataset YAML (default: phase8_final)",
    )
    p.add_argument("--project", default="harmony_omr_v2_phase3")
    p.add_argument(
        "--name",
        default=None,
        help="Run name. Defaults to {arch}_{strategy}_{timestamp}",
    )

    # --- hardware ---
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbs", type=int, default=64)

    # --- two_stage specific ---
    p.add_argument(
        "--s1-epochs",
        type=int,
        default=30,
        help="Stage-1 epochs (two_stage only, default: 30)",
    )
    p.add_argument(
        "--s1-lr0",
        type=float,
        default=0.001,
        help="Stage-1 initial LR (default: 0.001)",
    )
    p.add_argument(
        "--s1-warmup-epochs",
        type=float,
        default=5.0,
        help="Stage-1 warmup epochs (default: 5)",
    )
    p.add_argument(
        "--s1-freeze",
        type=int,
        default=10,
        help="Number of backbone layers to freeze in stage-1 (default: 10)",
    )
    p.add_argument(
        "--s2-epochs",
        type=int,
        default=150,
        help="Stage-2 epochs (two_stage only, default: 150)",
    )
    p.add_argument(
        "--s2-lr0",
        type=float,
        default=0.0001,
        help="Stage-2 initial LR (default: 0.0001)",
    )
    p.add_argument(
        "--s2-lrf",
        type=float,
        default=0.01,
        help="Stage-2 LR final ratio (default: 0.01)",
    )
    p.add_argument(
        "--s2-patience",
        type=int,
        default=40,
        help="Stage-2 early-stop patience (default: 40)",
    )

    # --- cosine_low_lr specific ---
    p.add_argument(
        "--cl-lr0",
        type=float,
        default=0.0005,
        help="cosine_low_lr initial LR (default: 0.0005)",
    )
    p.add_argument(
        "--cl-warmup-epochs",
        type=float,
        default=10.0,
        help="cosine_low_lr warmup epochs (default: 10)",
    )
    p.add_argument(
        "--cl-epochs",
        type=int,
        default=200,
        help="cosine_low_lr total epochs (default: 200)",
    )
    p.add_argument(
        "--cl-patience",
        type=int,
        default=40,
        help="cosine_low_lr early-stop patience (default: 40)",
    )

    # --- from_pretrained specific ---
    p.add_argument(
        "--fp-lr0",
        type=float,
        default=0.001,
        help="from_pretrained initial LR (default: 0.001)",
    )
    p.add_argument(
        "--fp-epochs",
        type=int,
        default=200,
        help="from_pretrained total epochs (default: 200)",
    )
    p.add_argument(
        "--fp-patience",
        type=int,
        default=40,
        help="from_pretrained early-stop patience (default: 40)",
    )
    p.add_argument(
        "--fp-warmup-epochs",
        type=float,
        default=5.0,
        help="from_pretrained warmup epochs (default: 5)",
    )

    # --- common hypers ---
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--warmup-momentum", type=float, default=0.8)
    p.add_argument("--warmup-bias-lr", type=float, default=0.1)
    p.add_argument("--cls", type=float, default=0.7)
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--mosaic", type=float, default=0.5)
    p.add_argument("--close-mosaic", type=int, default=20)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--copy-paste", type=float, default=0.0)

    # --- checkpointing / monitoring ---
    p.add_argument(
        "--save-period",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    p.add_argument(
        "--check-ckpt-every",
        type=int,
        default=5,
        help="Check EMA NaN every N epochs (default: 5)",
    )
    p.add_argument(
        "--watchdog-minutes",
        type=int,
        default=30,
        help="Kill process if no activity for N minutes (default: 30)",
    )

    # --- post-training eval ---
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip unified_eval.py after training finishes",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(log_dir: Path, run_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{stamp}.log"

    logger = logging.getLogger("phase3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("log_path=%s", log_path)
    return logger


# ---------------------------------------------------------------------------
# GPU / checkpoint helpers
# ---------------------------------------------------------------------------


def gpu_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        return {
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(idx),
            "cuda_version": torch.version.cuda,
            "total_gb": round(props.total_memory / 1024**3, 2),
            "alloc_gb": round(torch.cuda.memory_allocated(idx) / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(idx) / 1024**3, 2),
        }
    except Exception as exc:
        return {"cuda_available": False, "error": str(exc)}


def nvidia_smi_short() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "nvidia-smi-unavailable"


def count_ema_nan(ckpt_path: Path) -> int:
    """Return number of NaN values in the EMA state dict, or -1 if absent."""
    if not ckpt_path.exists():
        return -1
    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ema = ckpt.get("ema")
        if ema is None:
            return 0
        nan_total = 0
        for tensor in ema.state_dict().values():
            if torch.is_tensor(tensor):
                nan_total += int(torch.isnan(tensor.float()).sum().item())
        return nan_total
    except Exception:
        return -2


# ---------------------------------------------------------------------------
# Ultralytics save-dir resolution helper
# ---------------------------------------------------------------------------


def _find_weights(
    model: Any,
    project: str,
    run_name: str,
    weight_name: str = "best.pt",
) -> Path:
    """Locate weights saved by Ultralytics after model.train().

    Ultralytics may save to ``runs/detect/{project}/{name}/`` instead of
    ``{project}/{name}/`` depending on internal settings.  This helper tries
    the trainer's actual ``save_dir`` first, then falls back to common
    candidate paths.
    """
    candidates: list[Path] = []

    # 1. Preferred: use the trainer's recorded save_dir (most reliable)
    try:
        save_dir = Path(model.trainer.save_dir)
        candidates.append(save_dir / "weights" / weight_name)
    except Exception:
        pass

    # 2. Relative to cwd: {project}/{name}/weights/
    candidates.append(Path(project) / run_name / "weights" / weight_name)

    # 3. Ultralytics default: runs/detect/{project}/{name}/weights/
    candidates.append(Path("runs") / "detect" / project / run_name / "weights" / weight_name)

    for p in candidates:
        if p.exists():
            return p

    # None found -- raise with all tried paths for debugging
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot find {weight_name} for run '{run_name}'. Tried: {tried}"
    )


def _find_ckpt_dir(
    project: str,
    run_name: str,
) -> Path | None:
    """Return the checkpoint directory for a run (for postmortem), or None."""
    for base in [Path(project), Path("runs") / "detect" / project]:
        d = base / run_name / "weights"
        if d.is_dir():
            return d
    return None


# ---------------------------------------------------------------------------
# Shared training-argument builder
# ---------------------------------------------------------------------------


def _common_train_args(args: argparse.Namespace, run_name: str) -> dict[str, Any]:
    """Return kwargs shared by all strategies."""
    return {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cache": False,
        "amp": False,  # prevent inf/NaN issues (per prior experiments)
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "nbs": args.nbs,
        "cls": args.cls,
        "box": args.box,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "scale": args.scale,
        "copy_paste": args.copy_paste,
        "mixup": 0.0,
        "save": True,
        "save_period": args.save_period,
        "plots": False,
        "val": True,
        "verbose": True,
        "project": args.project,
        "name": run_name,
        "exist_ok": True,
        "deterministic": False,
        "resume": False,
    }


# ---------------------------------------------------------------------------
# Watchdog + callbacks factory
# ---------------------------------------------------------------------------


def make_activity_state() -> dict[str, Any]:
    return {
        "last_activity": time.time(),
        "last_tag": "startup",
        "stopped": False,
    }


def start_watchdog(state: dict[str, Any], timeout_minutes: int, logger: logging.Logger) -> None:
    timeout_sec = timeout_minutes * 60

    def _loop() -> None:
        while not state["stopped"]:
            idle = time.time() - state["last_activity"]
            if idle > timeout_sec:
                logger.error(
                    "watchdog_timeout idle_sec=%s last_tag=%s -- force-exit",
                    int(idle),
                    state["last_tag"],
                )
                logger.error("watchdog_gpu=%s", json.dumps(gpu_snapshot(), ensure_ascii=False))
                logger.error("watchdog_smi=%s", nvidia_smi_short())
                os._exit(3)
            time.sleep(15)

    threading.Thread(target=_loop, daemon=True, name="watchdog").start()


def attach_callbacks(
    model: Any,
    state: dict[str, Any],
    logger: logging.Logger,
    check_ckpt_every: int,
    stage_label: str = "",
) -> None:
    """Register on_train_batch_end, on_val_batch_end, on_train_epoch_end."""

    prefix = f"[{stage_label}] " if stage_label else ""

    def on_train_batch_end(trainer: Any) -> None:
        batch_i = int(getattr(trainer, "batch_i", -1))
        state["last_activity"] = time.time()
        state["last_tag"] = f"{stage_label}:train_batch:{batch_i}"

    def on_val_batch_end(validator: Any) -> None:
        batch_i = int(getattr(validator, "batch_i", -1))
        state["last_activity"] = time.time()
        state["last_tag"] = f"{stage_label}:val_batch:{batch_i}"

    def on_train_epoch_end(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        state["last_activity"] = time.time()
        state["last_tag"] = f"{stage_label}:epoch:{epoch}"

        lr = None
        if getattr(trainer, "optimizer", None) is not None:
            try:
                lr = trainer.optimizer.param_groups[0]["lr"]
            except (IndexError, KeyError):
                pass

        tloss = getattr(trainer, "tloss", None)

        logger.info(
            "%sepoch_end=%s lr=%s tloss=%s gpu=%s smi=%s",
            prefix,
            epoch,
            f"{lr:.8f}" if isinstance(lr, float) else str(lr),
            str(tloss),
            json.dumps(gpu_snapshot(), ensure_ascii=False),
            nvidia_smi_short(),
        )

        if epoch % check_ckpt_every == 0:
            save_dir = Path(getattr(trainer, "save_dir", ""))
            last_ckpt = save_dir / "weights" / "last.pt"
            nan_cnt = count_ema_nan(last_ckpt) if last_ckpt.exists() else -1
            logger.info(
                "%sckpt_check epoch=%s last_ckpt=%s ema_nan=%s",
                prefix,
                epoch,
                last_ckpt,
                nan_cnt,
            )

    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_val_batch_end", on_val_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------


def run_unified_eval(
    best_model_path: Path,
    data: str,
    logger: logging.Logger,
) -> None:
    """Invoke unified_eval.py as a subprocess on the best checkpoint."""
    eval_script = Path(__file__).parent / "unified_eval.py"
    if not eval_script.exists():
        logger.warning("unified_eval.py not found at %s -- skipping eval", eval_script)
        return
    if not best_model_path.exists():
        logger.warning("best.pt not found at %s -- skipping eval", best_model_path)
        return

    cmd = [
        sys.executable,
        str(eval_script),
        "--models",
        str(best_model_path),
        "--data",
        data,
        "--imgsz",
        "1280",
        "--device",
        "0",
    ]
    logger.info("post_eval_cmd=%s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=False)
        logger.info("post_eval_returncode=%s", proc.returncode)
    except Exception as exc:
        logger.exception("post_eval_error=%s", exc)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def resolve_base_model(args: argparse.Namespace, logger: logging.Logger) -> Path:
    """Return the verified starting-weights path."""
    if args.base_model is not None:
        path = Path(args.base_model)
    else:
        path = Path(DEFAULT_BASE_MODELS[args.arch])
    if not path.exists():
        logger.error("base_model_not_found=%s", path)
        raise FileNotFoundError(f"base model not found: {path}")
    logger.info("base_model=%s", path)
    return path


def strategy_two_stage(
    args: argparse.Namespace,
    run_name: str,
    state: dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """Two-stage training: frozen backbone then progressive unfreeze.

    Stage 1: freeze first N layers, train detection head at moderate LR
    Stage 2: unfreeze all, continue at ultra-low LR with cosine schedule
    """
    import torch
    from ultralytics import YOLO

    base_model_path = resolve_base_model(args, logger)

    # ------------------------------------------------------------------
    # Stage 1 -- frozen backbone
    # ------------------------------------------------------------------
    s1_name = f"{run_name}_s1"
    s1_args = _common_train_args(args, s1_name)
    s1_args.update(
        {
            "epochs": args.s1_epochs,
            "lr0": args.s1_lr0,
            "lrf": 0.1,
            "warmup_epochs": args.s1_warmup_epochs,
            "patience": 0,  # no early stop during stage-1 (always run full)
            "cos_lr": True,
            "freeze": args.s1_freeze,
        }
    )

    logger.info("=== Stage 1: frozen backbone ===")
    logger.info("s1_args=%s", json.dumps(s1_args, ensure_ascii=False, sort_keys=True))

    model = YOLO(str(base_model_path))
    attach_callbacks(model, state, logger, args.check_ckpt_every, stage_label="s1")

    model.train(**s1_args)
    state["last_tag"] = "s1_complete"
    logger.info("stage1_done")

    # Locate stage-1 best weights as the stage-2 starting point.
    try:
        s1_best = _find_weights(model, args.project, s1_name, "best.pt")
    except FileNotFoundError:
        # Fall back to last.pt if best.pt wasn't written yet
        s1_best = _find_weights(model, args.project, s1_name, "last.pt")
    logger.info("s1_best=%s exists=%s", s1_best, s1_best.exists())

    nan_cnt = count_ema_nan(s1_best)
    logger.info("s1_ema_nan=%s", nan_cnt)
    if nan_cnt > 0:
        logger.warning("s1 EMA has %s NaN values; stage-2 may diverge", nan_cnt)

    # ------------------------------------------------------------------
    # Stage 2 -- full fine-tune at ultra-low LR
    # ------------------------------------------------------------------
    s2_name = f"{run_name}_s2"
    s2_args = _common_train_args(args, s2_name)
    s2_args.update(
        {
            "epochs": args.s2_epochs,
            "lr0": args.s2_lr0,
            "lrf": args.s2_lrf,
            "warmup_epochs": 3.0,
            "patience": args.s2_patience,
            "cos_lr": True,
            "freeze": 0,  # unfreeze everything
        }
    )

    logger.info("=== Stage 2: full fine-tune ===")
    logger.info("s2_args=%s", json.dumps(s2_args, ensure_ascii=False, sort_keys=True))
    logger.info("s2_starting_from=%s", s1_best)

    model2 = YOLO(str(s1_best))
    attach_callbacks(model2, state, logger, args.check_ckpt_every, stage_label="s2")

    model2.train(**s2_args)
    state["last_tag"] = "s2_complete"
    logger.info("stage2_done")

    best_pt = _find_weights(model2, args.project, s2_name, "best.pt")
    logger.info("final_best=%s", best_pt)
    return best_pt


def strategy_cosine_low_lr(
    args: argparse.Namespace,
    run_name: str,
    state: dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """Single-stage with very low LR, long warmup, and cosine annealing.

    Safer than two-stage when the base model is already well-trained.
    Prevents catastrophic forgetting by keeping the LR very small throughout.
    """
    from ultralytics import YOLO

    base_model_path = resolve_base_model(args, logger)

    train_args = _common_train_args(args, run_name)
    train_args.update(
        {
            "epochs": args.cl_epochs,
            "lr0": args.cl_lr0,
            "lrf": 0.01,
            "warmup_epochs": args.cl_warmup_epochs,
            "patience": args.cl_patience,
            "cos_lr": True,
            "freeze": 0,
        }
    )

    logger.info("=== cosine_low_lr single-stage ===")
    logger.info("train_args=%s", json.dumps(train_args, ensure_ascii=False, sort_keys=True))

    model = YOLO(str(base_model_path))
    attach_callbacks(model, state, logger, args.check_ckpt_every, stage_label="cl")

    model.train(**train_args)
    state["last_tag"] = "cl_complete"
    logger.info("cosine_low_lr_done")

    best_pt = _find_weights(model, args.project, run_name, "best.pt")
    logger.info("final_best=%s", best_pt)
    return best_pt


def strategy_from_pretrained(
    args: argparse.Namespace,
    run_name: str,
    state: dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """Train from official YOLO pretrained weights (not from a fine-tuned model).

    Useful when base fine-tuned weights are suspected to be the source of
    catastrophic forgetting -- start fresh with strong COCO pretraining.
    """
    from ultralytics import YOLO

    pretrained_weight = PRETRAINED_WEIGHTS[args.arch]
    logger.info("from_pretrained=%s (will be downloaded if missing)", pretrained_weight)

    train_args = _common_train_args(args, run_name)
    train_args.update(
        {
            "epochs": args.fp_epochs,
            "lr0": args.fp_lr0,
            "lrf": 0.01,
            "warmup_epochs": args.fp_warmup_epochs,
            "patience": args.fp_patience,
            "cos_lr": True,
            "freeze": 0,
            "pretrained": True,
        }
    )

    logger.info("=== from_pretrained ===")
    logger.info("arch=%s weight=%s", args.arch, pretrained_weight)
    logger.info("train_args=%s", json.dumps(train_args, ensure_ascii=False, sort_keys=True))

    model = YOLO(pretrained_weight)
    attach_callbacks(model, state, logger, args.check_ckpt_every, stage_label="fp")

    model.train(**train_args)
    state["last_tag"] = "fp_complete"
    logger.info("from_pretrained_done")

    best_pt = _find_weights(model, args.project, run_name, "best.pt")
    logger.info("final_best=%s", best_pt)
    return best_pt


# ---------------------------------------------------------------------------
# Error recovery: log checkpoint health on failure
# ---------------------------------------------------------------------------


def _postmortem_ckpt_scan(
    project: str,
    run_names: list[str],
    logger: logging.Logger,
) -> None:
    for name in run_names:
        weights_dir = _find_ckpt_dir(project, name)
        if weights_dir is None:
            logger.error("postmortem run=%s weights_dir_not_found", name)
            continue
        for ckpt_stem in ("best.pt", "last.pt"):
            ckpt_path = weights_dir / ckpt_stem
            if ckpt_path.exists():
                try:
                    nan_cnt = count_ema_nan(ckpt_path)
                    logger.error("postmortem ckpt=%s ema_nan=%s", ckpt_path, nan_cnt)
                except Exception as scan_exc:
                    logger.error("postmortem ckpt=%s scan_error=%s", ckpt_path, scan_exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # Resolve run name early so the logger file is correctly named.
    if args.name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.arch}_{args.strategy}_{stamp}"
    else:
        run_name = args.name

    # Logger lives in <project>/logs/
    log_dir = Path(args.project) / "logs"
    logger = setup_logger(log_dir, run_name)

    # ----- signal handling -----
    def handle_signal(signum: int, _frame: Any) -> None:
        logger.error("received_signal=%s; graceful stop", signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # ----- environment diagnostics -----
    import torch
    from ultralytics import __version__ as ultralytics_version

    logger.info("python=%s", sys.version.split()[0])
    logger.info("torch=%s ultralytics=%s", torch.__version__, ultralytics_version)
    logger.info("startup_gpu=%s", json.dumps(gpu_snapshot(), ensure_ascii=False))
    logger.info("startup_smi=%s", nvidia_smi_short())
    logger.info(
        "run=%s strategy=%s arch=%s data=%s",
        run_name,
        args.strategy,
        args.arch,
        args.data,
    )

    # ----- shared activity state -----
    state = make_activity_state()
    start_watchdog(state, args.watchdog_minutes, logger)

    # ----- run selected strategy -----
    # Collect run-name(s) for postmortem if training fails.
    candidate_run_names: list[str] = []
    if args.strategy == "two_stage":
        candidate_run_names = [f"{run_name}_s1", f"{run_name}_s2"]
    else:
        candidate_run_names = [run_name]

    best_pt: Path | None = None
    exit_code = 0

    try:
        if args.strategy == "two_stage":
            best_pt = strategy_two_stage(args, run_name, state, logger)
        elif args.strategy == "cosine_low_lr":
            best_pt = strategy_cosine_low_lr(args, run_name, state, logger)
        elif args.strategy == "from_pretrained":
            best_pt = strategy_from_pretrained(args, run_name, state, logger)
        else:
            logger.error("unknown_strategy=%s", args.strategy)
            return 2

        state["stopped"] = True
        logger.info("training_finished status=success best_pt=%s", best_pt)

    except (KeyboardInterrupt, SystemExit) as exc:
        state["stopped"] = True
        logger.error("training_interrupted exc=%s", exc)
        _postmortem_ckpt_scan(args.project, candidate_run_names, logger)
        return 1

    except Exception as exc:
        state["stopped"] = True
        logger.error("training_failed error=%s", exc)
        logger.error("traceback:\n%s", traceback.format_exc())
        _postmortem_ckpt_scan(args.project, candidate_run_names, logger)
        return 1

    # ----- post-training evaluation -----
    if not args.skip_eval and best_pt is not None:
        logger.info("running_post_eval best_pt=%s", best_pt)
        try:
            run_unified_eval(best_pt, args.data, logger)
        except Exception as exc:
            logger.exception("post_eval_failed=%s", exc)
            # Not a fatal error; training already succeeded.

    logger.info("done exit_code=%s", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
