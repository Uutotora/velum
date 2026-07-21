"""Velum — the Models & Train tab controller.

Qt-free glue between the real one-shot LoRA training pipeline
(``velum_core/train_model.py`` + ``train_state_manager.py`` — the exact
code the classic app's Train tab already uses) and Studio's Models & Train
screen: resolving a SAM backbone, validating an annotated image has a mask,
running training on a background thread with live progress, and listing
trained models / recent runs from disk.

Mirrors ``velum_core/predict_controller.py``'s shape — plain data in,
plain callbacks out, no Qt — so it is unit-tested without PyQt6. Heavy deps
(torch, cellseg1_train) are never imported at module top level; they're
pulled in lazily inside ``velum_core.train_model.train_model`` itself,
exactly as the classic Train tab already relies on.

Data-directory design: unlike the classic Train tab (whose "Images folder" /
"Masks folder" fields point at one shared, accumulating pair of folders),
Studio's mockup shows a single "Annotated image" field — the product's
"one-shot fine-tuning from a single annotated image" framing. To make that
framing literally true (not "this image plus whatever was ever picked
before"), each run's chosen image+mask pair is copied into a fresh,
isolated ``<storage>/studio_train_runs/<run_id>/{images,masks}/`` — never
into the classic app's shared folders, so the two apps' training data never
silently mixes. The trained-model checkpoint + Aim-free JSON sidecar still
land in the same shared ``loras/`` folder either app uses, so a model
trained in one app shows up in the other.
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

MASK_EXTS = (".png", ".tif", ".tiff", ".npy", ".bmp")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy")

# SAM backbone checkpoint filenames, keyed the same as velum_core's engines.
BACKBONE_FILES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
BACKBONE_LABELS = {"vit_h": "ViT-H", "vit_l": "ViT-L", "vit_b": "ViT-B"}

RANK_OPTIONS = ["4", "8", "16", "32"]
EPOCH_OPTIONS = ["50", "100", "150", "300", "500"]
DEFAULT_RANK = "8"
DEFAULT_EPOCHS = "100"


def default_storage_dir() -> Path:
    """The conventional storage root, shared with the classic app.

    Imported lazily so this module has no hard import-time dependency on
    ``project_root`` — mirrors ``studio.project.default_store_root``.
    """
    from project_root import STORAGE_DIR
    return Path(STORAGE_DIR)


def _read_mask_array(path: Path) -> Optional[np.ndarray]:
    """Load a mask file as an int array, or ``None`` if unreadable."""
    try:
        if path.suffix.lower() == ".npy":
            return np.load(str(path)).astype(np.int32)
        import cv2
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return None if m is None else np.ascontiguousarray(m).astype(np.int32)
    except Exception:
        return None


def count_cells_in_mask(path: str | Path) -> Optional[int]:
    """Real cell count for one mask file (``int(mask.max())``), or ``None``."""
    m = _read_mask_array(Path(path))
    return None if m is None else int(m.max())


def count_cells_in_dir(dir_path: str | Path) -> Optional[int]:
    """Total cells across every mask file in ``dir_path``, or ``None`` if the
    directory is gone or has nothing readable (a training run's per-run mask
    folder is never deleted, but nothing guarantees it survives forever)."""
    d = Path(dir_path)
    if not d.is_dir():
        return None
    total = 0
    found = False
    for f in sorted(d.iterdir()):
        if f.suffix.lower() in MASK_EXTS:
            n = count_cells_in_mask(f)
            if n is not None:
                total += n
                found = True
    return total if found else None


def find_mask_for_image(image_path: str | Path, mask_dir: Optional[Path] = None) -> Optional[Path]:
    """Look for an existing mask matching ``image_path`` by filename stem.

    Tries, in order: a sibling ``<stem>_mask.*``/``<stem>_masks.*`` next to
    the image, a sibling ``masks/<stem>.*`` folder, and (if given) the
    classic app's shared mask folder — so an image already annotated via the
    classic Train tab's "use active napari layers" export is found too.
    """
    image_path = Path(image_path)
    stem = image_path.stem
    candidates: list[Path] = []
    for ext in MASK_EXTS:
        candidates.append(image_path.with_name(f"{stem}_mask{ext}"))
        candidates.append(image_path.with_name(f"{stem}_masks{ext}"))
    for ext in MASK_EXTS:
        candidates.append(image_path.parent / "masks" / f"{stem}{ext}")
    if mask_dir is not None:
        for ext in MASK_EXTS:
            candidates.append(Path(mask_dir) / f"{stem}{ext}")
    for c in candidates:
        if c.exists():
            return c
    return None


def available_backbones(sam_backbone_dir: Path) -> list[tuple[str, str]]:
    """``[(vit_key, label)]`` for every backbone whose weight file exists."""
    out = []
    for key, label in BACKBONE_LABELS.items():
        if (Path(sam_backbone_dir) / BACKBONE_FILES[key]).exists():
            out.append((key, label))
    return out


def guess_vit_name(path: str | Path) -> str:
    """Best-effort SAM architecture guess from a manually-picked checkpoint's
    filename (``sam_vit_l_...`` -> ``vit_l``), defaulting to the flagship
    ``vit_h`` when the name gives no hint — mirrors the classic Train tab's
    manual "SAM backbone" browse field, which doesn't cross-validate the
    picked file against the architecture either; it's the same trust-the-
    user contract, just inferred from one field instead of two."""
    name = Path(path).name.lower()
    if "vit_l" in name or "vit-l" in name:
        return "vit_l"
    if "vit_b" in name or "vit-b" in name:
        return "vit_b"
    return "vit_h"


def duration_str(seconds: float) -> str:
    """``"8m 12s"`` style duration, matching the mockup's run rows."""
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s"


def parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


@dataclass
class TrainedModel:
    """One row of the "Trained models" list — real data from a checkpoint's
    JSON sidecar (``velum_core.train_model._save_config_sidecar``)."""

    name: str
    checkpoint: Path
    meta: str
    f1: Optional[str]
    saved_at: str
    n_cells: Optional[int]


@dataclass
class RecentRun:
    """One row of the "Recent training runs" aside — either the live
    in-progress run (``state="run"``) or a finished one from history."""

    name: str
    meta: str
    state: str  # "run" | "done" | "stopped"


def _model_meta(sidecar: dict) -> str:
    vit = sidecar.get("vit_name", "")
    vit_label = BACKBONE_LABELS.get(vit, vit)
    rank = sidecar.get("image_encoder_lora_rank", "?")
    n_images = len(sidecar.get("train_id", []) or [])
    plural = "image" if n_images == 1 else "images"
    return f"{vit_label} · rank {rank} · {n_images} {plural}"


def list_trained_models(lora_out_dir: Path) -> list[TrainedModel]:
    """Real "Trained models" list: every ``*.json`` sidecar in ``lora_out_dir``,
    newest ``saved_at`` first. A checkpoint whose sidecar is missing/corrupt is
    skipped rather than crashing the list (mirrors ``ProjectStore.list()``)."""
    out: list[TrainedModel] = []
    d = Path(lora_out_dir)
    if not d.is_dir():
        return out
    for jf in d.glob("*.json"):
        try:
            sidecar = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        n_cells = count_cells_in_dir(sidecar.get("train_mask_dir", ""))
        out.append(TrainedModel(
            name=jf.stem,
            checkpoint=jf.with_suffix(".pth"),
            meta=_model_meta(sidecar),
            f1=None,  # F1 needs ground-truth benchmarking; never computed at train time
            saved_at=sidecar.get("saved_at", ""),
            n_cells=n_cells,
        ))
    out.sort(key=lambda m: m.saved_at, reverse=True)
    return out


def list_recent_runs(state_manager, limit: int = 3) -> list[RecentRun]:
    """Real "Recent training runs" history (excludes any live in-progress run
    — see ``TrainController.current_run()`` for that), newest first."""
    out: list[RecentRun] = []
    for entry in state_manager.load_history()[:limit]:
        name = Path(entry.get("checkpoint", "")).stem or "training run"
        started, finished = parse_iso(entry.get("started_at", "")), parse_iso(entry.get("finished_at", ""))
        status = entry.get("status", "completed")
        if started and finished:
            dur = duration_str((finished - started).total_seconds())
            meta = f"done · {dur}" if status == "completed" else f"stopped · {dur}"
        else:
            meta = status
        out.append(RecentRun(name=name, meta=meta, state="done" if status == "completed" else "stopped"))
    return out


class TrainController:
    """Owns training config-building, background orchestration, and the
    model/run listings — independent of Qt (see module docstring)."""

    def __init__(self, storage_dir: Optional[Path | str] = None):
        self.storage_dir = Path(storage_dir) if storage_dir is not None else default_storage_dir()
        self.run_data_dir = self.storage_dir / "studio_train_runs"
        self.lora_out_dir = self.storage_dir / "loras"
        self.sam_backbone_dir = self.storage_dir / "sam_backbone"
        self.lora_out_dir.mkdir(parents=True, exist_ok=True)

        from velum_core.train_state_manager import TrainingStateManager
        self.state_manager = TrainingStateManager(str(self.storage_dir))

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._progress_queue: "queue.Queue[dict]" = queue.Queue()
        self._loss_history: list[dict] = []
        self._active_name: Optional[str] = None
        self._active_total_epochs: Optional[int] = None

    # ── listings ─────────────────────────────────────────────────────────────
    def list_trained_models(self) -> list[TrainedModel]:
        return list_trained_models(self.lora_out_dir)

    def list_recent_runs(self, limit: int = 3) -> list[RecentRun]:
        return list_recent_runs(self.state_manager, limit=limit)

    def available_backbones(self) -> list[tuple[str, str]]:
        return available_backbones(self.sam_backbone_dir)

    def find_mask_for_image(self, image_path: str | Path) -> Optional[Path]:
        return find_mask_for_image(image_path)

    # ── live progress ────────────────────────────────────────────────────────
    def is_training(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def current_run(self) -> Optional[RecentRun]:
        """The live in-progress run as an aside row, or ``None`` when idle.

        Drains whatever progress has arrived since the last call — cheap and
        safe to call on a UI poll timer, matching the classic Train tab.
        """
        if not self.is_training() or self._active_name is None:
            return None
        last = None
        while not self._progress_queue.empty():
            item = self._progress_queue.get_nowait()
            self._loss_history.append({"epoch": item["epoch"], "loss": item["loss"]})
            last = item
        if last is None:
            if not self._loss_history:
                return RecentRun(self._active_name, "starting…", "run")
            last = self._loss_history[-1]
        total = self._active_total_epochs or "?"
        return RecentRun(self._active_name, f"epoch {last['epoch']}/{total} · loss {last['loss']:.2f}", "run")

    def current_loss_history(self) -> list[dict]:
        """The in-progress run's loss history accumulated so far (``{"epoch",
        "loss"}`` dicts). Call ``current_run()`` first on the same poll tick
        to drain any newly-arrived progress into it."""
        return list(self._loss_history)

    def active_epoch_max(self) -> Optional[int]:
        """The in-progress run's target epoch count, or ``None`` when idle."""
        return self._active_total_epochs if self.is_training() else None

    # ── config building ──────────────────────────────────────────────────────
    def _prepare_run_data(self, image_path: Path, mask_path: Path) -> tuple[str, str]:
        """Copy the chosen image+mask into a fresh, isolated run folder.

        Never touches the caller's originals or the classic app's shared
        ``train_images``/``train_masks`` — see the module docstring for why.
        """
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        image_dir = self.run_data_dir / run_id / "images"
        mask_dir = self.run_data_dir / run_id / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        dst_image = image_dir / image_path.name
        dst_mask = mask_dir / mask_path.name
        shutil.copy2(image_path, dst_image)
        shutil.copy2(mask_path, dst_mask)
        return str(image_dir), str(mask_dir)

    @staticmethod
    def _detect_device() -> str:
        """Best available device, degrading to "cpu" with no torch installed
        (keeps this callable from the light, torch-free CI test group)."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "0"
        except Exception:
            pass
        return "cpu"

    def resolve_backbone(self, vit_name: str) -> Path:
        path = self.sam_backbone_dir / BACKBONE_FILES[vit_name]
        if not path.exists():
            raise ValueError(f"SAM backbone not found: {path.name}")
        return path

    def build_config(self, *, image_path: str | Path, mask_path: str | Path,
                      vit_name: Optional[str], lora_rank: int, epochs: int,
                      backbone_path: Optional[str | Path] = None) -> dict:
        """Build a full training config from Studio's 4-field form.

        ``backbone_path``, when given, is used directly instead of resolving
        ``vit_name`` against ``sam_backbone_dir`` — the manual "browse for a
        checkpoint" fallback for when nothing was auto-detected there (or the
        user wants a different one); ``vit_name`` is then only a label
        (best-effort-guessed from the filename if not given either).

        Raises ``ValueError`` on anything not ready to train (mirrors
        ``PredictController.build_config``'s contract). Copies the chosen
        image+mask into an isolated run folder as a side effect — see
        ``_prepare_run_data``.
        """
        if image_path is None:
            raise ValueError("No annotated image selected")
        if mask_path is None:
            raise ValueError("No mask found for this image")
        image_path, mask_path = Path(image_path), Path(mask_path)
        if not image_path.exists():
            raise ValueError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise ValueError(f"Mask not found: {mask_path}")

        if backbone_path is not None:
            sam_path = Path(backbone_path)
            if not sam_path.exists():
                raise ValueError(f"SAM backbone not found: {sam_path}")
            vit_name = vit_name or guess_vit_name(sam_path)
        elif vit_name is not None:
            sam_path = self.resolve_backbone(vit_name)
        else:
            raise ValueError("No SAM backbone selected")
        image_dir, mask_dir = self._prepare_run_data(image_path, mask_path)

        resize = 512
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = self.lora_out_dir / f"lora_{vit_name}_r{lora_rank}_s{resize}_{ts}.pth"

        return {
            "deterministic": True, "seed": 0,
            "allow_tf32_on_cudnn": True, "allow_tf32_on_matmul": True,
            "vit_name": vit_name, "model_path": str(sam_path),
            "train_image_dir": image_dir, "train_mask_dir": mask_dir,
            "result_pth_path": str(out_path),
            "resize_size": [resize, resize], "patch_size": resize // 2, "sam_image_size": resize,
            "train_id": [0], "duplicate_data": 32,
            "epoch_max": int(epochs), "batch_size": 1,
            "gradient_accumulation_step": 32,
            "base_lr": 3e-3, "onecycle_lr_pct_start": 0.3, "num_workers": 0,
            "image_encoder_lora_rank": int(lora_rank),
            "mask_decoder_lora_rank": int(lora_rank),
            "freeze_image_encoder": True, "freeze_prompt_encoder": True,
            "freeze_mask_decoder_transformer": True, "freeze_upscaling_cnn": True,
            "freeze_output_hypernetworks_mlps": True,
            "freeze_mask_decoder_mask_tokens": True, "freeze_mask_decoder_iou": True,
            "lora_dropout": 0.1,
            "pos_rate": 1.0, "neg_rate": 0.5, "max_point_num": 30,
            "edge_distance": 20, "neg_area_ratio_threshold": 5,
            "neg_area_threshold": 1000, "min_cell_area": 100,
            "foreground_sample_area_ratio": 0.2, "background_sample_area_ratio": 0.2,
            "foreground_equal_prob": True, "background_equal_prob": True,
            "data_augmentation": True, "bright_limit": 0.1, "contrast_limit": 0.1,
            "bright_prob": 0.5, "flip_prob": 0.75, "rotate_prob": 0.8,
            "scale_limit": [-0.5, 0.5], "crop_prob": 0.5,
            "crop_scale": [0.3, 1.0], "crop_ratio": [0.75, 1.3333],
            "ce_loss_weight": 1.0, "punish_background_point": False,
            "track_gpu_memory": False, "selected_device": self._detect_device(),
        }

    # ── orchestration ────────────────────────────────────────────────────────
    def start_training(self, config: dict, *, on_log: Optional[Callable[[str], None]] = None,
                        on_finish: Optional[Callable[[], None]] = None) -> threading.Thread:
        """Run ``train_model`` on a background daemon thread (mirrors the
        classic ``TrainWidget._start_training``). Call ``current_run()`` on a
        poll timer for live progress."""
        self.state_manager.clear_training_state()
        self.state_manager.clear_stop_flag()
        self.state_manager.clear_loss_history()
        self._stop_event.clear()
        while not self._progress_queue.empty():
            self._progress_queue.get_nowait()
        self._loss_history = []
        self._active_name = Path(config["result_pth_path"]).stem
        self._active_total_epochs = config["epoch_max"]

        pq, se = self._progress_queue, self._stop_event

        def run():
            from velum_core.train_model import train_model
            try:
                train_model(config, self.state_manager, progress_queue=pq, stop_event=se)
                if on_log:
                    on_log("Training complete")
            except Exception as e:
                if on_log:
                    on_log(f"[ERROR] {e}")
            finally:
                self._active_name = None
                if on_finish:
                    on_finish()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self._thread

    def stop_training(self) -> None:
        self._stop_event.set()
        self.state_manager.set_stop_flag()

    # ── model management ─────────────────────────────────────────────────────
    def import_model(self, src_path: str | Path) -> Path:
        """Copy an externally-trained ``.pth`` (+ matching ``.json`` sidecar,
        if present next to it) into the shared ``loras/`` folder."""
        src = Path(src_path)
        if not src.exists():
            raise ValueError(f"File not found: {src}")
        self.lora_out_dir.mkdir(parents=True, exist_ok=True)
        dst = self.lora_out_dir / src.name
        shutil.copy2(src, dst)
        src_json = src.with_suffix(".json")
        if src_json.exists():
            shutil.copy2(src_json, dst.with_suffix(".json"))
        return dst

    def select_model_for_project(self, model: TrainedModel, project) -> None:
        """Point ``project.settings`` at ``model`` — the "select into
        workspace" hook the Segment tab will read once it's wired."""
        sidecar_path = model.checkpoint.with_suffix(".json")
        vit_name, rank = "vit_h", 8
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            vit_name = sidecar.get("vit_name", vit_name)
            rank = sidecar.get("image_encoder_lora_rank", rank)
        except (OSError, json.JSONDecodeError):
            pass
        project.settings.engine = "cellseg1"
        project.settings.model_name = str(model.checkpoint)
        project.settings.vit_name = vit_name
        project.settings.lora_rank = rank
