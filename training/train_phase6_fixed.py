#!/usr/bin/env python3
"""Phase 6: Fixed OpenScore notehead bboxes + stem-removed training.

Root cause (ADR-001): OpenScore notehead bboxes were LilyPond glyph-group boxes
(~380×335px including stem+beam), not individual noteheads (~23×25px).
NMS@0.7 recall ceiling was 0.526 — impossible for any model to exceed.

Phase 6 fix: place notehead bbox at top edge of glyph-group bbox.
  - NMS@0.7 ceiling: 0.5177 → 1.0000 (799,583 noteheads verified)
  - OpenScore/DoReMi bbox ratio: 27.6x → 1.04x

Retraining strategy (ADR-002): Surgical cv2 Reset + LP-FT Two-Stage
  - cv2 (bbox regression, 639K params, 6.9%) is class-agnostic and contaminated
  - Reset cv2 with Kaiming init, preserve backbone + neck + cv3
  - Prediction vs GT: 622x area mismatch, 0% IoU match → retraining mandatory
  - Ref: LP-FT (ICML 2022), Surgical Fine-Tuning (ICLR 2023)

Base model: Phase 5 Stage 2 best.pt (mAP50≈0.827 on old val labels)
Dataset: Phase 6 Fixed (32 classes, 32K train, 2,867 cleaned val)

Two-stage progressive unfreeze:
  Stage 1: Freeze backbone (layers 0-9), train neck + reset head — 40 epochs
  Stage 2: Unfreeze all, very low LR — 120 epochs

Success criteria (ADR-002 §4.4):
  A (突破): mAP50 > 0.85 AND nh_filled Recall > 0.70
  B (達標): mAP50 > 0.83 AND nh_filled mAP50 > 0.60
  C (不足): → fallback to full head reset (cv2 + cv3)
  D (失敗): → fallback to COCO retrain
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_ROOT = Path("/home/thc1006/dev/music-app")
_TRAIN_ROOT = _ROOT / "training"

# Use Phase 5 Stage 2 best as base (already adapted to 32 classes)
BASE_MODEL = str(
    _ROOT / "runs/detect/harmony_omr_v2_phase5/nostem_v1_stage2/weights/best.pt"
)
DATA = str(
    _TRAIN_ROOT / "datasets/yolo_harmony_v2_phase6_fixed/harmony_phase6_fixed.yaml"
)
PROJECT = str(_ROOT / "runs/detect/harmony_omr_v2_phase6")
NAME = "fixed_bbox_v1"

MAX_BOXES = 400  # TAL box capping (prevents OOM on dense images)
TAL_TOPK = 13    # Assigner default is 13, v8DetectionLoss overrides to 10 — fix it

# Target classes for per-class logging
FOCUS_CLASSES = {
    0: "notehead_filled",
    1: "notehead_hollow",
    2: "beam",
    7: "tie",
    22: "barline",
    23: "barline_double",
    31: "ledger_line",
}

# Stage 1: Freeze backbone, train head + neck
STAGE1_ARGS = {
    "data": DATA,
    "imgsz": 1280,
    "batch": 6,
    "nbs": 64,
    "device": "0",
    "workers": 12,
    "cache": False,
    "amp": False,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.1,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,
    "epochs": 40,
    "patience": 20,
    "freeze": 10,
    # Loss weights — box=7.5 to accelerate cv2 convergence from Kaiming init
    # (ADR-002 §4.3: cv2 starts from scratch, needs stronger box gradient)
    "cls": 1.0,
    "box": 7.5,
    "dfl": 1.5,
    # Augmentation: NONE for dense OMR
    "mosaic": 0.0,
    "copy_paste": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "scale": 0.2,
    "erasing": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.2,
    "close_mosaic": 0,
    # CRITICAL: max_det=1500
    "max_det": 1500,
    # Saving
    "save": True,
    "save_period": 5,
    "plots": True,
    "val": True,
    "verbose": True,
    "project": PROJECT,
    "name": NAME,
    "exist_ok": True,
    "deterministic": False,
    "compile": True,
}

# Stage 2: Unfreeze all, very low LR
STAGE2_ARGS = {
    **STAGE1_ARGS,
    "freeze": None,
    "lr0": 0.0001,
    "lrf": 0.01,
    "warmup_epochs": 2.0,
    "epochs": 120,        # ADR-002: shorter than Phase 5 (93% params pretrained)
    "patience": 40,
    "box": 5.0,           # Restore normal ratio (cv2 has converged from Stage 1)
    "save_period": 10,
    "name": f"{NAME}_stage2",
}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_TW = timezone(timedelta(hours=8))


def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    """Create a structured logger with file + console output.

    Log format: Taiwan-local timestamp, level, message.
    Separate JSON log for machine-readable epoch metrics.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    class TaiwanFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=_TW)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    fmt = TaiwanFormatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (human-readable)
    fh = logging.FileHandler(str(log_dir / f"{name}.log"), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class MetricsLogger:
    """JSON-lines logger for structured epoch metrics.

    Each line is a self-contained JSON object for easy parsing:
    {"ts": "...", "stage": 1, "epoch": 5, "mAP50": 0.82, ...}
    """

    def __init__(self, log_dir: Path, name: str):
        self.path = log_dir / f"{name}_metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_epoch(self, stage: int, epoch: int, metrics: dict) -> None:
        record = {
            "ts": datetime.now(tz=_TW).isoformat(),
            "stage": stage,
            "epoch": epoch,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_eval(self, protocol: str, metrics: dict) -> None:
        record = {
            "ts": datetime.now(tz=_TW).isoformat(),
            "event": "eval",
            "protocol": protocol,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Surgical cv2 Reset (ADR-002 §4.1)
# ---------------------------------------------------------------------------

def surgical_reset_cv2(model, log: logging.Logger) -> int:
    """Reset bbox regression head (cv2) with Kaiming init.

    cv2 is class-agnostic (shared across all 32 classes) and was trained on
    622x oversized OpenScore glyph-group bboxes. Reset to clean slate while
    preserving backbone (58.8%), neck (32.2%), and cv3 classification (2.1%).

    Architecture (YOLO12s Detect, layer 21):
      cv2[i]: Conv(ch→64,3) → Conv(64→64,3) → Conv2d(64→64,1)
      Total: 639,936 params (6.9% of 9.27M model)

    Returns number of parameters reset.
    """
    detect = model.model.model[-1]  # Detect layer

    reset_count = 0
    for i, seq in enumerate(detect.cv2):
        for m in seq.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                reset_count += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                reset_count += m.weight.numel() + m.bias.numel()
        # Ultralytics bias_init convention: box regression bias = 2.0
        seq[-1].bias.data[:] = 2.0

    total_params = sum(p.numel() for p in model.model.parameters())
    log.info(
        "cv2 surgical reset: %d params re-initialized (%.1f%% of %d total)",
        reset_count, reset_count / total_params * 100, total_params,
    )
    return reset_count


# ---------------------------------------------------------------------------
# Monkey-Patches (same as Phase 5, battle-tested)
# ---------------------------------------------------------------------------

def _apply_box_capping_patch(max_boxes: int, log: logging.Logger) -> None:
    """Cap GT boxes per image in TAL to prevent OOM."""
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.ops import xywh2xyxy

    _original_preprocess = v8DetectionLoss.preprocess

    def _capped_preprocess(
        self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor
    ) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        actual_max = int(counts.max().item())

        if actual_max <= max_boxes:
            return _original_preprocess(self, targets, batch_size, scale_tensor)

        capped_targets = []
        for j in range(batch_size):
            matches = i == j
            n = int(matches.sum().item())
            if n == 0:
                continue
            img_targets = targets[matches]
            if n > max_boxes:
                perm = torch.randperm(n, device=targets.device)[:max_boxes]
                img_targets = img_targets[perm]
            capped_targets.append(img_targets)

        if capped_targets:
            targets = torch.cat(capped_targets, dim=0)
        else:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        for j in range(batch_size):
            matches = i == j
            if n := matches.sum():
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    v8DetectionLoss.preprocess = _capped_preprocess
    log.info("TAL box capping applied: max_boxes=%d", max_boxes)


def _apply_tal_topk_patch(topk: int, log: logging.Logger) -> None:
    """Patch v8DetectionLoss to use tal_topk=13 instead of default 10."""
    from ultralytics.utils.loss import v8DetectionLoss

    _original_init = v8DetectionLoss.__init__

    def _patched_init(self, model, tal_topk: int = 10, tal_topk2=None):
        _original_init(self, model, tal_topk=topk, tal_topk2=tal_topk2)

    v8DetectionLoss.__init__ = _patched_init
    log.info("TAL topk patched: tal_topk=%d (was 10)", topk)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    model_path: str,
    data: str,
    log: logging.Logger,
    ml: MetricsLogger,
) -> tuple:
    """Run dual-protocol evaluation with per-class notehead metrics."""
    from ultralytics import YOLO

    eval_model = YOLO(model_path)

    # Deploy protocol
    m_deploy = eval_model.val(
        data=data, imgsz=1280, conf=0.15, iou=0.6,
        device="0", verbose=False, plots=False, max_det=1500,
    )
    deploy_metrics = {
        "mAP50": float(m_deploy.box.map50),
        "mAP50-95": float(m_deploy.box.map),
        "precision": float(m_deploy.box.mp),
        "recall": float(m_deploy.box.mr),
    }
    log.info(
        "EVAL deploy (conf=0.15, iou=0.6): mAP50=%.4f  P=%.3f  R=%.3f",
        deploy_metrics["mAP50"], deploy_metrics["precision"], deploy_metrics["recall"],
    )
    ml.log_eval("deploy", deploy_metrics)

    # Strict protocol
    m_strict = eval_model.val(
        data=data, imgsz=1280, conf=0.001, iou=0.7,
        device="0", verbose=False, plots=False, max_det=1500,
    )
    strict_metrics = {
        "mAP50": float(m_strict.box.map50),
        "mAP50-95": float(m_strict.box.map),
        "precision": float(m_strict.box.mp),
        "recall": float(m_strict.box.mr),
    }
    log.info(
        "EVAL strict (conf=0.001, iou=0.7): mAP50=%.4f  P=%.3f  R=%.3f",
        strict_metrics["mAP50"], strict_metrics["precision"], strict_metrics["recall"],
    )
    ml.log_eval("strict", strict_metrics)

    # Per-class focus metrics (strict protocol)
    if hasattr(m_strict, "box") and hasattr(m_strict.box, "class_result"):
        log.info("--- Per-class focus (strict) ---")
        per_class = {}
        for cls_id, cls_name in FOCUS_CLASSES.items():
            try:
                cr = m_strict.box.class_result(cls_id)
                p, r, map50, map50_95 = cr[0], cr[1], cr[2], cr[3]
                log.info(
                    "  %-22s  P=%.3f  R=%.3f  mAP50=%.3f  mAP50-95=%.3f",
                    cls_name, p, r, map50, map50_95,
                )
                per_class[cls_name] = {
                    "precision": float(p),
                    "recall": float(r),
                    "mAP50": float(map50),
                    "mAP50-95": float(map50_95),
                }
            except Exception:
                log.warning("  %-22s  (no data)", cls_name)
        ml.log_eval("strict_per_class", per_class)

    return m_deploy, m_strict


def extract_results_csv(results_csv: Path, stage: int, ml: MetricsLogger) -> None:
    """Parse results.csv and write to metrics JSONL for the completed stage."""
    if not results_csv.exists():
        return
    import csv
    with open(results_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["                  epoch"].strip()) if "                  epoch" in row else int(list(row.values())[0].strip())
            metrics = {}
            for k, v in row.items():
                k = k.strip()
                if k == "epoch":
                    continue
                try:
                    metrics[k] = float(v)
                except (ValueError, TypeError):
                    pass
            ml.log_epoch(stage, epoch, metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log_dir = Path(PROJECT) / "logs"
    log = setup_logger(NAME, log_dir)
    ml = MetricsLogger(log_dir, NAME)

    log.info("=" * 70)
    log.info("Phase 6: Fixed OpenScore Notehead Bboxes Training")
    log.info("  NMS ceiling fix: 0.5177 → 0.9901 (+47pp)")
    log.info("  OpenScore/DoReMi bbox ratio: 27.6x → 1.10x")
    log.info("=" * 70)
    log.info("torch=%s  cuda=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info(
            "gpu=%s  vram=%.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Check dataset
    data_path = Path(DATA)
    if not data_path.exists():
        log.error("Dataset not found: %s", DATA)
        log.error("Run create_phase6_fixed_bbox.py first!")
        return 1

    # Check base model
    base_path = Path(BASE_MODEL)
    if not base_path.exists():
        log.error("Base model not found: %s", BASE_MODEL)
        log.error("Phase 5 training must complete first!")
        return 1

    log.info("Base model: %s", BASE_MODEL)
    log.info("Dataset:    %s", DATA)

    # Apply patches BEFORE importing YOLO
    _apply_box_capping_patch(MAX_BOXES, log)
    _apply_tal_topk_patch(TAL_TOPK, log)

    from ultralytics import YOLO

    # ===== Surgical cv2 Reset (ADR-002 §4.1-4.2) =====
    log.info("=" * 70)
    log.info("SURGICAL cv2 RESET: Kaiming init for bbox regression head")
    log.info("  Reason: Phase 5 cv2 predicts 622x oversized boxes (532px vs 24px GT)")
    log.info("  Preserving: backbone (58.8%%) + neck (32.2%%) + cv3 cls (2.1%%)")
    log.info("=" * 70)

    model = YOLO(BASE_MODEL)
    surgical_reset_cv2(model, log)

    # CRITICAL: Save reset model to prevent intersect_dicts from restoring old cv2.
    # When Ultralytics loads a checkpoint, intersect_dicts() transfers all layers
    # with matching shapes — which would silently overwrite our reset cv2 weights.
    # By saving to a new checkpoint first, the "source" weights ARE the reset ones.
    reset_ckpt = Path(PROJECT) / "phase6_cv2_reset.pt"
    reset_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.model}, str(reset_ckpt))
    log.info("Reset checkpoint saved: %s", reset_ckpt)

    # Reload from reset checkpoint
    model = YOLO(str(reset_ckpt))
    log.info("Reloaded from reset checkpoint (intersect_dicts safe)")

    # ===== Stage 1: Frozen backbone =====
    log.info("=" * 70)
    log.info("STAGE 1: Freeze backbone (layers 0-9), train neck + reset head")
    log.info("  epochs=%d  lr0=%s  freeze=%s  batch=%d  box=%.1f",
             STAGE1_ARGS["epochs"], STAGE1_ARGS["lr0"],
             STAGE1_ARGS["freeze"], STAGE1_ARGS["batch"], STAGE1_ARGS["box"])
    log.info("  mosaic=%.1f  cls=%.1f  dfl=%.1f  max_det=%d",
             STAGE1_ARGS["mosaic"], STAGE1_ARGS["cls"],
             STAGE1_ARGS["dfl"], STAGE1_ARGS["max_det"])
    log.info("=" * 70)

    t0 = time.time()
    model.train(**STAGE1_ARGS)
    stage1_time = time.time() - t0
    log.info("Stage 1 complete in %.1f hours", stage1_time / 3600)

    try:
        stage1_dir = Path(model.trainer.save_dir)
        stage1_best = stage1_dir / "weights" / "best.pt"
    except Exception:
        stage1_dir = Path(PROJECT) / NAME
        stage1_best = stage1_dir / "weights" / "best.pt"

    log.info("Stage 1 best: %s (exists=%s)", stage1_best, stage1_best.exists())

    # Extract Stage 1 metrics to JSONL
    results_csv = stage1_dir / "results.csv"
    extract_results_csv(results_csv, stage=1, ml=ml)

    # --- Stage 1 sanity check (ADR-002 §4.4) ---
    # Quick eval to check if cv2 reset worked: notehead mAP50 should be > 0.30
    if stage1_best.exists():
        log.info("--- Stage 1 sanity check: notehead mAP50 ---")
        check_model = YOLO(str(stage1_best))
        m_check = check_model.val(
            data=DATA, imgsz=1280, conf=0.001, iou=0.7,
            device="0", verbose=False, plots=False, max_det=1500,
        )
        nh_filled_map50 = float(m_check.box.class_result(0)[2])
        nh_hollow_map50 = float(m_check.box.class_result(1)[2])
        overall_map50 = float(m_check.box.map50)
        log.info("  Stage 1 overall mAP50:     %.4f", overall_map50)
        log.info("  Stage 1 nh_filled mAP50:   %.4f", nh_filled_map50)
        log.info("  Stage 1 nh_hollow mAP50:   %.4f", nh_hollow_map50)
        ml.log_eval("stage1_check", {
            "mAP50": overall_map50,
            "nh_filled_mAP50": nh_filled_map50,
            "nh_hollow_mAP50": nh_hollow_map50,
        })

        if nh_filled_map50 < 0.15:
            log.warning(
                "FALLBACK TRIGGER: nh_filled mAP50=%.4f < 0.15 after Stage 1. "
                "cv2 reset may not be sufficient. Consider full head reset (ADR-002 §2 Option C).",
                nh_filled_map50,
            )

    # ===== Stage 2: Full model, very low LR =====
    log.info("=" * 70)
    log.info("STAGE 2: Unfreeze all, lr0=0.0001, epochs=120, patience=40")
    log.info("=" * 70)

    if stage1_best.exists():
        stage2_model = YOLO(str(stage1_best))
        log.info("Loaded Stage 1 best: %s", stage1_best)
    else:
        stage1_last = stage1_dir / "weights" / "last.pt"
        stage2_model = YOLO(str(stage1_last))
        log.warning("Stage 1 best not found, using last.pt")

    t1 = time.time()
    stage2_model.train(**STAGE2_ARGS)
    stage2_time = time.time() - t1
    log.info("Stage 2 complete in %.1f hours", stage2_time / 3600)

    try:
        stage2_dir = Path(stage2_model.trainer.save_dir)
        stage2_best = stage2_dir / "weights" / "best.pt"
    except Exception:
        stage2_dir = Path(PROJECT) / f"{NAME}_stage2"
        stage2_best = stage2_dir / "weights" / "best.pt"

    log.info("Stage 2 best: %s (exists=%s)", stage2_best, stage2_best.exists())

    # Extract Stage 2 metrics to JSONL
    results_csv = stage2_dir / "results.csv"
    extract_results_csv(results_csv, stage=2, ml=ml)

    total_hours = (stage1_time + stage2_time) / 3600
    log.info("Total training time: %.1f hours", total_hours)

    # ===== Final Evaluation =====
    best_pt = stage2_best if stage2_best.exists() else stage1_best
    if best_pt.exists():
        log.info("=" * 70)
        log.info("FINAL EVALUATION: %s", best_pt)
        log.info("=" * 70)
        m_deploy, m_strict = run_eval(str(best_pt), DATA, log, ml)

        log.info("=" * 70)
        log.info("PHASE 6 RESULTS SUMMARY")
        log.info("  Training time:     %.1f hours", total_hours)
        log.info("  Deploy mAP50:      %.4f", float(m_deploy.box.map50))
        log.info("  Strict mAP50:      %.4f", float(m_strict.box.map50))
        log.info("  Deploy Recall:     %.4f", float(m_deploy.box.mr))
        log.info("  Strict Recall:     %.4f", float(m_strict.box.mr))
        log.info("  ADR targets:  mAP50 > 0.85, nh Recall > 0.70")
        log.info("=" * 70)
    else:
        log.error("No best.pt found for evaluation!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
