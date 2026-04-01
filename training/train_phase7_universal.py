#!/usr/bin/env python3
"""Phase 7: Universal OpenScore bbox fix — retrain ALL 32 classes.

Phase 6 only fixed noteheads (2/32 classes). Phase 7 fixes ALL remaining 30 classes
using two empirically validated rules (ADR-003):
  - TOP edge rule (NMS 99.9-100%): beam, flags, augdot, tie, accidentals, fermata, ledger, clefs
  - CENTER rule (NMS 99.8-100%): rests, time_sig, key_sig, dynamics, barlines
  - Dedup pass: removes near-identical annotations (IoU >= 0.7 same-class)

NMS@0.7 ceiling: 57.1% → 100.00% (0 annotations killed, verified on 14.6M pairs)

Retraining strategy (ADR-002, same as Phase 6):
  - Surgical cv2 reset: ALL 30 fixed classes need bbox re-learning
  - cv2 is class-agnostic → single reset covers all classes
  - Two-stage LP-FT: freeze backbone → unfreeze

Base model: Phase 6 Stage 2 best.pt (mAP50≈0.813, only noteheads fixed)
Dataset: Phase 7 Universal (32 classes, 32K train, 2,867 cleaned val, 100% NMS ceiling)

Success criteria:
  A (突破): mAP50 > 0.87 AND overall Recall > 0.75
  B (達標): mAP50 > 0.85
  C (不足): → analyze per-class, consider class-specific fine-tune
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# RTX 5090 Hardware Optimization (Blackwell, compute 12.0, 33.7GB VRAM)
# ---------------------------------------------------------------------------
# TF32: ~1.5x matmul speedup, negligible accuracy loss (same exponent as fp32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# cuDNN auto-tuner: picks fastest convolution algorithm for fixed input sizes
torch.backends.cudnn.benchmark = True
# 'high' enables TF32 for torch.compile and matmul globally
torch.set_float32_matmul_precision('high')


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_ROOT = Path("/home/thc1006/dev/music-app")
_TRAIN_ROOT = _ROOT / "training"

# Base: Phase 6 Stage 2 best (noteheads fixed, 32 classes, mAP50≈0.813)
BASE_MODEL = str(
    _ROOT / "runs/detect/harmony_omr_v2_phase6/fixed_bbox_v1_stage2/weights/best.pt"
)
DATA = str(
    _TRAIN_ROOT / "datasets/yolo_harmony_v2_phase7_universal/harmony_phase7_universal.yaml"
)
PROJECT = str(_ROOT / "runs/detect/harmony_omr_v2_phase7")
NAME = "universal_v1"

MAX_BOXES = 400
TAL_TOPK = 13

FOCUS_CLASSES = {
    0: "notehead_filled",
    1: "notehead_hollow",
    2: "beam",
    6: "augmentation_dot",
    7: "tie",
    12: "accidental_sharp",
    22: "barline",
    27: "key_signature",
    28: "fermata",
    31: "ledger_line",
}

# Stage 1: Freeze backbone, train neck + reset cv2 — 40 epochs
STAGE1_ARGS = {
    "data": DATA,
    "imgsz": 1280,
    "batch": 8,       # RTX 5090 33.7GB → batch=8 uses ~24GB, safe with 10GB headroom
    "nbs": 64,
    "device": "0",
    "workers": 16,    # i9-14900 24 cores → 16 workers for data loading
    "cache": False,
    "amp": False,      # fp16/bf16 both cause mAP→0.25 (model weights precision-sensitive)
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
    # box=7.5 to accelerate cv2 convergence from Kaiming init (proven in Phase 6)
    "cls": 1.0,
    "box": 7.5,
    "dfl": 1.5,
    # No augmentation for dense OMR
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
    "max_det": 1500,
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

# Stage 2: Unfreeze all, very low LR — 120 epochs
STAGE2_ARGS = {
    **STAGE1_ARGS,
    "freeze": None,
    "lr0": 0.0001,
    "lrf": 0.01,
    "warmup_epochs": 2.0,
    "epochs": 120,
    "patience": 40,
    "box": 5.0,
    "save_period": 10,
    "name": f"{NAME}_stage2",
}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_TW = timezone(timedelta(hours=8))


def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    class TaiwanFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=_TW)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    fmt = TaiwanFormatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_dir / f"{name}.log"), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class MetricsLogger:
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
# Surgical cv2 Reset (same as Phase 6, proven working)
# ---------------------------------------------------------------------------

def surgical_reset_cv2(model, log: logging.Logger) -> int:
    """Reset bbox regression head (cv2) with Kaiming init.

    Phase 7 changes ALL 30 classes' bboxes (not just noteheads).
    cv2 is class-agnostic, so the same single reset handles all classes.
    """
    detect = model.model.model[-1]
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
        seq[-1].bias.data[:] = 2.0

    total_params = sum(p.numel() for p in model.model.parameters())
    log.info(
        "cv2 surgical reset: %d params re-initialized (%.1f%% of %d total)",
        reset_count, reset_count / total_params * 100, total_params,
    )
    return reset_count


# ---------------------------------------------------------------------------
# Monkey-Patches
# ---------------------------------------------------------------------------

def _apply_box_capping_patch(max_boxes: int, log: logging.Logger) -> None:
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.ops import xywh2xyxy

    # Idempotency guard: don't double-wrap
    if getattr(v8DetectionLoss, '_box_capping_patched', False):
        log.warning("Box capping patch already applied, skipping.")
        return

    _original_preprocess = v8DetectionLoss.preprocess
    _logged_cap = [False]  # mutable container for closure

    def _capped_preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        actual_max = int(counts.max().item())

        if actual_max <= max_boxes:
            return _original_preprocess(self, targets, batch_size, scale_tensor)

        if not _logged_cap[0]:
            log.warning("Box capping triggered: %d GT boxes in one image, capping to %d",
                        actual_max, max_boxes)
            _logged_cap[0] = True

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

        if not capped_targets:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        targets = torch.cat(capped_targets, dim=0)
        i = targets[:, 0]

        # Allocate output with max_boxes columns (known upper bound, avoids extra unique())
        out = torch.zeros(batch_size, max_boxes, ne - 1, device=self.device)
        for j in range(batch_size):
            matches = i == j
            n = int(matches.sum().item())
            if n > 0:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    v8DetectionLoss.preprocess = _capped_preprocess
    v8DetectionLoss._box_capping_patched = True
    log.info("TAL box capping applied: max_boxes=%d", max_boxes)


def _apply_tal_topk_patch(topk: int, log: logging.Logger) -> None:
    from ultralytics.utils.loss import v8DetectionLoss

    _original_init = v8DetectionLoss.__init__

    def _patched_init(self, model, tal_topk=10, tal_topk2=None):
        _original_init(self, model, tal_topk=topk, tal_topk2=tal_topk2)

    v8DetectionLoss.__init__ = _patched_init
    log.info("TAL topk patched: tal_topk=%d (was 10)", topk)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(model_path: str, data: str, log: logging.Logger, ml: MetricsLogger):
    from ultralytics import YOLO

    eval_model = YOLO(model_path)

    # Deploy protocol
    m_deploy = eval_model.val(
        data=data, imgsz=1280, conf=0.15, iou=0.6,
        device="0", verbose=False, plots=False, max_det=1500,
    )
    log.info(
        "EVAL deploy: mAP50=%.4f  P=%.3f  R=%.3f",
        float(m_deploy.box.map50), float(m_deploy.box.mp), float(m_deploy.box.mr),
    )
    ml.log_eval("deploy", {
        "mAP50": float(m_deploy.box.map50),
        "mAP50-95": float(m_deploy.box.map),
        "precision": float(m_deploy.box.mp),
        "recall": float(m_deploy.box.mr),
    })

    # Strict protocol
    m_strict = eval_model.val(
        data=data, imgsz=1280, conf=0.001, iou=0.7,
        device="0", verbose=False, plots=False, max_det=1500,
    )
    log.info(
        "EVAL strict: mAP50=%.4f  P=%.3f  R=%.3f",
        float(m_strict.box.map50), float(m_strict.box.mp), float(m_strict.box.mr),
    )
    ml.log_eval("strict", {
        "mAP50": float(m_strict.box.map50),
        "mAP50-95": float(m_strict.box.map),
        "precision": float(m_strict.box.mp),
        "recall": float(m_strict.box.mr),
    })

    # Per-class focus metrics
    if hasattr(m_strict.box, "class_result"):
        log.info("--- Per-class focus (strict) ---")
        per_class = {}
        for cls_id, cls_name in FOCUS_CLASSES.items():
            try:
                p, r, map50, map50_95 = m_strict.box.class_result(cls_id)
                log.info("  %-22s  P=%.3f  R=%.3f  mAP50=%.3f", cls_name, p, r, map50)
                per_class[cls_name] = {
                    "P": float(p), "R": float(r),
                    "mAP50": float(map50), "mAP50-95": float(map50_95),
                }
            except Exception:
                log.warning("  %-22s  (no data)", cls_name)
        ml.log_eval("strict_per_class", per_class)

    return m_deploy, m_strict


def extract_results_csv(results_csv: Path, stage: int, ml: MetricsLogger) -> None:
    """Parse results.csv and write per-epoch metrics to JSONL."""
    if not results_csv.exists():
        return
    import csv
    with open(results_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ultralytics results.csv has padded headers; find epoch robustly
            epoch = None
            metrics = {}
            for k, v in row.items():
                k_clean = k.strip()
                v_clean = v.strip() if v else ""
                if k_clean == "epoch":
                    try:
                        epoch = int(v_clean)
                    except (ValueError, TypeError):
                        pass
                    continue
                try:
                    metrics[k_clean] = float(v_clean)
                except (ValueError, TypeError):
                    pass
            if epoch is None:
                # Fallback: first column is epoch (legacy format)
                try:
                    epoch = int(list(row.values())[0].strip())
                except (ValueError, TypeError, IndexError):
                    continue
            ml.log_epoch(stage, epoch, metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log_dir = Path(PROJECT) / "logs"
    log = setup_logger(NAME, log_dir)
    ml = MetricsLogger(log_dir, NAME)

    log.info("=" * 70)
    log.info("Phase 7: Universal OpenScore Bbox Fix — ALL 32 classes")
    log.info("  Phase 6: only noteheads fixed → mAP50=0.813")
    log.info("  Phase 7: ALL 30 remaining classes fixed")
    log.info("  NMS@0.7 ceiling: 57.1%% → 100.00%% (verified 14.6M pairs)")
    log.info("  Target: mAP50 > 0.87")
    log.info("=" * 70)
    log.info("torch=%s  cuda=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info("gpu=%s  vram=%.1f GB  compute=%d.%d  SMs=%d",
                 torch.cuda.get_device_name(0), props.total_memory / 1e9,
                 props.major, props.minor, props.multi_processor_count)
    log.info("TF32=%s  cuDNN_bench=%s  precision=%s  amp=False(fp32+TF32)",
             torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.benchmark,
             torch.get_float32_matmul_precision())

    # Verify dataset
    data_path = Path(DATA)
    if not data_path.exists():
        log.error("Dataset not found: %s", DATA)
        log.error("Run create_phase7_dataset.py first!")
        return 1

    # Verify base model
    base_path = Path(BASE_MODEL)
    if not base_path.exists():
        log.error("Base model not found: %s", BASE_MODEL)
        return 1

    log.info("Base model: %s", BASE_MODEL)
    log.info("Dataset:    %s", DATA)

    # NOTE: PYTORCH_CUDA_ALLOC_CONF must be set BEFORE torch import to take effect.
    # Since torch is already imported at module level, we set it here for child processes
    # (e.g., DataLoader workers). For the main process, CUDA allocator is already initialized.
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Apply patches BEFORE importing YOLO
    # NOTE: bf16 patch NOT applied — both fp16 and bf16 cause mAP→0.25 on this model.
    # TF32 + cuDNN benchmark + batch=8 give ~1.8x speedup without precision risk.
    _apply_box_capping_patch(MAX_BOXES, log)
    _apply_tal_topk_patch(TAL_TOPK, log)

    from ultralytics import YOLO

    # ===== Surgical cv2 Reset =====
    log.info("=" * 70)
    log.info("SURGICAL cv2 RESET")
    log.info("  Phase 7 changes bboxes for ALL 30 non-notehead classes")
    log.info("  cv2 is class-agnostic → single reset covers everything")
    log.info("=" * 70)

    model = YOLO(BASE_MODEL)
    surgical_reset_cv2(model, log)

    # Save reset checkpoint (intersect_dicts protection)
    reset_ckpt = Path(PROJECT) / "phase7_cv2_reset.pt"
    reset_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.model}, str(reset_ckpt))
    log.info("Reset checkpoint saved: %s", reset_ckpt)

    model = YOLO(str(reset_ckpt))
    log.info("Reloaded from reset checkpoint (intersect_dicts safe)")

    # ===== Stage 1: Frozen backbone =====
    log.info("=" * 70)
    log.info("STAGE 1: Freeze backbone, train neck + reset head — %d epochs", STAGE1_ARGS["epochs"])
    log.info("  lr0=%s  box=%.1f  cls=%.1f  batch=%d",
             STAGE1_ARGS["lr0"], STAGE1_ARGS["box"],
             STAGE1_ARGS["cls"], STAGE1_ARGS["batch"])
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

    # Extract Stage 1 per-epoch metrics
    extract_results_csv(stage1_dir / "results.csv", stage=1, ml=ml)

    # Stage 1 sanity check
    if stage1_best.exists():
        log.info("--- Stage 1 sanity check ---")
        check_model = YOLO(str(stage1_best))
        m_check = check_model.val(
            data=DATA, imgsz=1280, conf=0.001, iou=0.7,
            device="0", verbose=False, plots=False, max_det=1500,
        )
        overall_map50 = float(m_check.box.map50)
        log.info("  Stage 1 overall mAP50: %.4f", overall_map50)
        for cls_id, cls_name in FOCUS_CLASSES.items():
            try:
                map50 = float(m_check.box.class_result(cls_id)[2])
                log.info("  %-22s mAP50=%.3f", cls_name, map50)
            except Exception:
                pass
        ml.log_eval("stage1_check", {"mAP50": overall_map50})

        if overall_map50 < 0.50:
            log.warning("Stage 1 mAP50=%.4f < 0.50 — cv2 reset may need more epochs", overall_map50)

    # ===== Stage 2: Full model, very low LR =====
    log.info("=" * 70)
    log.info("STAGE 2: Unfreeze all — %d epochs, lr0=%s, patience=%d",
             STAGE2_ARGS["epochs"], STAGE2_ARGS["lr0"], STAGE2_ARGS["patience"])
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

    # Extract Stage 2 per-epoch metrics
    try:
        extract_results_csv(stage2_dir / "results.csv", stage=2, ml=ml)
    except Exception:
        pass

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
        log.info("PHASE 7 RESULTS")
        log.info("  Training time:  %.1f hours", total_hours)
        log.info("  Deploy mAP50:   %.4f", float(m_deploy.box.map50))
        log.info("  Strict mAP50:   %.4f", float(m_strict.box.map50))
        log.info("  Deploy Recall:  %.4f", float(m_deploy.box.mr))
        log.info("  Strict Recall:  %.4f", float(m_strict.box.mr))
        log.info("  Targets: mAP50 > 0.87, Recall > 0.75")
        log.info("=" * 70)
    else:
        log.error("No best.pt found for evaluation!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
