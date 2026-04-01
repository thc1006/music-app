#!/usr/bin/env python3
"""Aggressive YOLO26 training runner with crash-focused diagnostics."""

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

import torch
from ultralytics import YOLO, __version__ as ultralytics_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 aggressive training with robust logging")
    parser.add_argument(
        "--model",
        default="../runs/detect/harmony_omr_v2_yolo26/yolo26s_official_e250_w12_b4_amp0_m05_a7/weights/best.pt",
    )
    parser.add_argument(
        "--data",
        default="datasets/yolo_harmony_v2_phase8_final/harmony_phase8_final.yaml",
    )
    parser.add_argument("--project", default="harmony_omr_v2_yolo26")
    parser.add_argument("--name", default="yolo26s_attack_a9_sgd_b5_m07_cp01_l1")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.0035)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=5)
    parser.add_argument("--warmup-momentum", type=float, default=0.85)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--nbs", type=int, default=96)
    parser.add_argument("--mosaic", type=float, default=0.7)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--copy-paste", type=float, default=0.1)
    parser.add_argument("--cls", type=float, default=1.0)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--save-period", type=int, default=2)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--check-ckpt-every", type=int, default=5)
    parser.add_argument("--watchdog-minutes", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def setup_logger(log_dir: Path, run_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{stamp}.log"

    logger = logging.getLogger("yolo26_attack")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("logger_path=%s", log_path)
    return logger


def gpu_snapshot() -> dict[str, Any]:
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


def count_ema_nan(ckpt_path: Path) -> int:
    if not ckpt_path.exists():
        return -1
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema")
    if ema is None:
        return 0
    nan_total = 0
    for tensor in ema.state_dict().values():
        if torch.is_tensor(tensor):
            nan_total += int(torch.isnan(tensor.float()).sum().item())
    return nan_total


def nvidia_smi_short() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return out
    except Exception:
        return "nvidia-smi-unavailable"


def main() -> int:
    args = parse_args()
    logger = setup_logger(Path("logs"), args.name)
    state = {"last_activity": time.time(), "last_tag": "startup", "stopped": False}

    def handle_signal(signum: int, _frame: Any) -> None:
        logger.error("received_signal=%s; attempting graceful stop", signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("model_not_found=%s", model_path)
        return 2

    logger.info("python=%s", sys.version.split()[0])
    logger.info("torch=%s ultralytics=%s", torch.__version__, ultralytics_version)
    logger.info("startup_gpu=%s", json.dumps(gpu_snapshot(), ensure_ascii=False))
    logger.info("startup_nvidia_smi=%s", nvidia_smi_short())

    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cache": False,
        "amp": False,  # keep numerically safer due prior NaN/Inf incidents
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "nbs": args.nbs,
        "cos_lr": True,
        "patience": args.patience,
        "close_mosaic": 20,
        "mosaic": args.mosaic,
        "scale": args.scale,
        "copy_paste": args.copy_paste,
        "mixup": 0.0,
        "cls": args.cls,
        "box": args.box,
        "save": True,
        "save_period": args.save_period,
        "plots": False,
        "val": True,
        "verbose": True,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "deterministic": False,
        "resume": args.resume,
    }
    logger.info("train_args=%s", json.dumps(train_args, ensure_ascii=False, sort_keys=True))

    model = YOLO(str(model_path))

    def mark_activity(tag: str) -> None:
        state["last_activity"] = time.time()
        state["last_tag"] = tag

    def watchdog_loop() -> None:
        timeout = args.watchdog_minutes * 60
        while not state["stopped"]:
            idle = time.time() - state["last_activity"]
            if idle > timeout:
                logger.error(
                    "watchdog_timeout idle_sec=%s last_tag=%s; force-exit",
                    int(idle),
                    state["last_tag"],
                )
                logger.error("watchdog_gpu=%s", json.dumps(gpu_snapshot(), ensure_ascii=False))
                logger.error("watchdog_nvidia_smi=%s", nvidia_smi_short())
                os._exit(3)
            time.sleep(15)

    threading.Thread(target=watchdog_loop, daemon=True).start()

    def on_train_batch_end(trainer: Any) -> None:
        batch_i = int(getattr(trainer, "batch_i", -1))
        mark_activity(f"train_batch_end:{batch_i}")

    def on_val_batch_end(trainer: Any) -> None:
        batch_i = int(getattr(trainer, "batch_i", -1))
        mark_activity(f"val_batch_end:{batch_i}")

    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_val_batch_end", on_val_batch_end)

    def on_train_epoch_end(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        mark_activity(f"train_epoch_end:{epoch}")
        lr = trainer.optimizer.param_groups[0]["lr"] if getattr(trainer, "optimizer", None) else None
        tloss = getattr(trainer, "tloss", None)
        logger.info(
            "epoch_end=%s lr=%s tloss=%s gpu=%s nvsmi=%s",
            epoch,
            f"{lr:.8f}" if isinstance(lr, float) else lr,
            str(tloss),
            json.dumps(gpu_snapshot(), ensure_ascii=False),
            nvidia_smi_short(),
        )
        if epoch % args.check_ckpt_every == 0:
            save_dir = Path(getattr(trainer, "save_dir", ""))
            last_ckpt = save_dir / "weights" / "last.pt"
            nan_cnt = count_ema_nan(last_ckpt) if last_ckpt.exists() else -1
            logger.info("ckpt_check epoch=%s last_ckpt=%s ema_nan=%s", epoch, last_ckpt, nan_cnt)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    try:
        logger.info("training_start model=%s", model_path)
        model.train(**train_args)
        state["stopped"] = True
        logger.info("training_finished status=success")
    except Exception as exc:
        state["stopped"] = True
        logger.error("training_failed error=%s", exc)
        logger.error("traceback:\n%s", traceback.format_exc())
        save_dir = Path(args.project) / args.name / "weights"
        for ckpt_name in ("best.pt", "last.pt"):
            ckpt_path = save_dir / ckpt_name
            if ckpt_path.exists():
                try:
                    nan_cnt = count_ema_nan(ckpt_path)
                    logger.error("postmortem ckpt=%s ema_nan=%s", ckpt_path, nan_cnt)
                except Exception as ckpt_exc:
                    logger.error("postmortem ckpt=%s scan_error=%s", ckpt_path, ckpt_exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
