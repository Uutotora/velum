"""Velum — the Segment tab's predict/GT/batch/benchmark controller.

Qt-free glue between a ``Project``'s settings (``studio/project.py``) and the
real segmentation pipeline — reuses the classic app's proven ML core
(``velum_core.predict_controller.PredictController``, ``analysis``,
``benchmark``, ``cohort``) exactly as ``docs/velum/ARCHITECTURE.md`` prescribes
for this tab, imported lazily so this module itself never pulls in torch.

Mirrors ``studio/train_controller.py``'s shape: plain data in, plain
callbacks out, no Qt, background threads for anything that touches a model.
Also mirrors its save convention — methods here *mutate* the ``Project``
object handed in (settings/stats) but never call ``ProjectStore.save``
themselves; the caller (``studio/workspace.py``) persists, exactly like
``TrainController.select_model_for_project`` leaves saving to
``ModelsScreen``. That keeps this controller independent of any particular
``ProjectStore`` instance, just like ``TrainController`` is.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from studio.project import Project, ProjectSettings

# Engine/threshold triples for the Segment pane's "Quality preset" control —
# Balanced matches PredictController.sam_config's own long-standing defaults;
# Fast/Accurate trade detection recall for speed in either direction.
QUALITY_PRESETS: dict[str, dict[str, float]] = {
    "Fast":     {"points_per_side": 16, "pred_iou_thresh": 0.75, "stability_score_thresh": 0.50},
    "Balanced": {"points_per_side": 32, "pred_iou_thresh": 0.80, "stability_score_thresh": 0.60},
    "Accurate": {"points_per_side": 48, "pred_iou_thresh": 0.86, "stability_score_thresh": 0.70},
}


def apply_quality_preset(settings: ProjectSettings, preset: str) -> None:
    values = QUALITY_PRESETS.get(preset)
    if values is None:
        return
    settings.quality_preset = preset
    settings.points_per_side = int(values["points_per_side"])
    settings.pred_iou_thresh = float(values["pred_iou_thresh"])
    settings.stability_score_thresh = float(values["stability_score_thresh"])


def _detect_device() -> str:
    """Best available device, degrading to "cpu" with no torch installed —
    duplicated (not imported) from ``TrainController._detect_device``,
    matching that controller's own convention of staying self-contained
    rather than cross-importing a four-line static method."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


class SegmentController:
    """Owns predict/GT-evaluation/batch/benchmark orchestration for the
    Segment workspace — independent of Qt and of any particular project
    store (see module docstring)."""

    def __init__(self, storage_dir: Optional[Path | str] = None):
        from studio.train_controller import default_storage_dir
        self.storage_dir = Path(storage_dir) if storage_dir is not None else default_storage_dir()
        self.run_root = self.storage_dir / "studio_segment_runs"
        self.__predict_controller = None
        self._benchmark_stop = threading.Event()

    def _pc(self):
        if self.__predict_controller is None:
            from velum_core.predict_controller import PredictController
            self.__predict_controller = PredictController()
        return self.__predict_controller

    @staticmethod
    def _ensure_engines_registered() -> None:
        import velum_core.engines  # noqa: F401 — registers cellseg1 + cellpose
        import velum_core.engines_sam2  # noqa: F401 — registers sam2

    # ── engines ──────────────────────────────────────────────────────────────
    def list_available_engines(self) -> list[tuple[str, str, bool]]:
        """``[(key, label, available)]`` for every registered engine."""
        self._ensure_engines_registered()
        from velum_core.engine_registry import all_engines
        return [(spec.key, spec.label, spec.available()) for spec in all_engines()]

    def list_lora_models(self):
        """Trained CellSeg1/LoRA checkpoints, for the Segment pane's "Model"
        selector when ``settings.engine == "cellseg1"`` — the exact list
        Models & Train shows, so a model trained there is selectable here."""
        from studio.train_controller import list_trained_models
        return list_trained_models(self.storage_dir / "loras")

    # ── config building ──────────────────────────────────────────────────────
    def build_params(self, project: Project, image_path: str | Path) -> dict:
        """Flatten ``project.settings`` into the params dict
        ``PredictController.build_config``/``sam_config``/``sam2_config``
        expect — every key any of the three engine branches might read is
        always present so any of them can be built without a KeyError."""
        s = project.settings
        return {
            "image_path": str(image_path),
            "resize_size": int(s.resize_size),
            "engine": s.engine,
            # CellSeg1 (SAM + LoRA)
            "lora_custom_text": s.model_name or "",
            "lora_combo_text": "",
            "lora_paths": {},
            "sam_path_text": "",
            "vit_name": s.vit_name,
            "lora_rank": int(s.lora_rank),
            "storage_dir": str(self.storage_dir),
            "points_per_side": int(s.points_per_side),
            "pred_iou_thresh": float(s.pred_iou_thresh),
            "stability_score_thresh": float(s.stability_score_thresh),
            "box_nms_thresh": float(s.box_nms_thresh),
            "min_mask_area": int(s.min_mask_area),
            "device": _detect_device(),
            "half_precision": False,
            "compile_decoder": False,
            # Cellpose-SAM
            "cp_diameter": float(s.cp_diameter),
            "cp_flow_threshold": float(s.cp_flow_threshold),
            "cp_cellprob_threshold": float(s.cp_cellprob_threshold),
            # SAM 2
            "sam2_model_type": s.sam2_model,
            "sam2_checkpoint_text": "",
            "sam2_config_text": "",
            "sam2_tracking_mode": "propagate" if s.sam2_tracking_mode == "propagate" else "automatic",
            "sam2_max_objects": 40,
            # Image / channels / large-image
            "channels": list(s.channels),
            "channel_low": float(s.channel_low),
            "channel_high": float(s.channel_high),
            "clahe": bool(s.clahe),
            "tiled": bool(s.tiled),
            "tile_size": int(s.tile_size),
            "tile_overlap": int(s.tile_overlap),
            # 2-D only here — a project's images are individual files, not a
            # loaded stack; z-stack/time-lapse predict (_predict_volume) is a
            # deliberately separate, not-yet-wired path (Canvas can already
            # *display* a loaded volume via its MIP toggle, but nothing here
            # triggers volume prediction on one yet).
            "zstack": False,
            "stitch_iou": float(s.stitch_iou),
            # TIFF/OME-TIFF needs the channel-aware path just as much as
            # ND2/CZI/LIF.  Otherwise OpenCV can silently flatten or reject
            # the microscopy layout before an engine ever sees it.
            "microscopy_stack": Path(image_path).suffix.lower()
            in (".tif", ".tiff", ".nd2", ".czi", ".lif"),
        }

    def build_config(self, project: Project, image_path: str | Path) -> dict:
        from velum_core.predict_controller import PredictController
        self._ensure_engines_registered()
        return PredictController.build_config(self.build_params(project, image_path))

    # ── image loading (for the canvas, before/without a predict run) ────────
    @staticmethod
    def load_preview_with_metadata(image_path: str | Path) -> tuple[np.ndarray, float | None]:
        """Read a canvas image through the microscopy-aware core path.

        The optional-reader ``MissingReaderError`` deliberately reaches the
        workspace unchanged, where it becomes the reader's actionable install
        instruction instead of an OpenCV-style generic decode failure.
        """
        from velum_core.channels import read_pixel_size_um
        from velum_core.predict_controller import _read_for_predict

        suffix = Path(image_path).suffix.lower()
        microscopy_stack = suffix in (".tif", ".tiff", ".nd2", ".czi", ".lif")
        rgb, stack = _read_for_predict({
            "image_path": str(image_path),
            "microscopy_stack": microscopy_stack,
        })
        pixel_size_um = stack.pixel_size_um if stack is not None else read_pixel_size_um(image_path)
        return rgb, pixel_size_um

    @classmethod
    def load_preview_image(cls, image_path: str | Path) -> np.ndarray:
        """RGB uint8 array for ``image_path`` — the exact read/normalise path
        a real predict run would see, reused so what you preview is what you
        segment (channel projection, 16-bit stretch, native-format support)."""
        return cls.load_preview_with_metadata(image_path)[0]

    # ── single-image predict ─────────────────────────────────────────────────
    def run_predict_async(self, project: Project, image_path: str | Path, *,
                          on_result: Optional[Callable] = None,
                          on_log: Optional[Callable[[str], None]] = None,
                          on_finish: Optional[Callable[[], None]] = None,
                          on_tile: Optional[Callable[[int, int], None]] = None) -> threading.Thread:
        config = self.build_config(project, image_path)
        return self._pc().run_prediction_async(
            config, on_tile=on_tile, on_result=on_result, on_log=on_log, on_finish=on_finish)

    # ── measurements / colour-by ─────────────────────────────────────────────
    @staticmethod
    def compute_measurements(mask: np.ndarray, image: Optional[np.ndarray] = None,
                             pixel_size_um: float = 0.0) -> dict:
        from velum_core import analysis
        return analysis.compute_measurements(mask, intensity_image=image, pixel_size_um=pixel_size_um)

    @staticmethod
    def color_overrides_for(result: dict, key: str) -> dict[int, tuple]:
        from velum_core import analysis
        return analysis.label_colormap_from_measurement(result, key)

    @staticmethod
    def measurement_range(result: dict, key: str):
        from velum_core import analysis
        return analysis.measurement_range(result, key)

    @staticmethod
    def summary_line(result: dict) -> str:
        from velum_core import analysis
        return analysis.summary_line(result)

    # ── save / export ────────────────────────────────────────────────────────
    def project_run_dir(self, project: Project) -> Path:
        d = self.run_root / project.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_mask(self, mask: np.ndarray, out_path: str | Path) -> Path:
        import cv2
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), mask.astype(np.uint16))
        return out_path

    def export_measurements_csv(self, result: dict, out_path: str | Path) -> Path:
        from velum_core import analysis
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(analysis.rows_as_csv(result), encoding="utf-8")
        return out_path

    # ── persisted per-(project, image) results ───────────────────────────────
    # A predict run previously only ever lived in the in-memory LayerList —
    # navigate away (or close and reopen the project) and it was gone, with
    # no way to tell an image had already been segmented. These give it a
    # real home on disk, keyed by project + image so two projects (or two
    # images with the same filename in different folders) never collide.
    def mask_path_for_image(self, project: Project, image_path: str | Path) -> Path:
        key = hashlib.sha1(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()[:20]
        d = self.project_run_dir(project) / "masks"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.png"

    def save_result_mask(self, project: Project, image_path: str | Path, mask: np.ndarray) -> Path:
        return self.save_mask(mask, self.mask_path_for_image(project, image_path))

    def load_result_mask(self, project: Project, image_path: str | Path) -> Optional[np.ndarray]:
        path = self.mask_path_for_image(project, image_path)
        if not path.exists():
            return None
        import cv2
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            return None
        return np.ascontiguousarray(m).astype(np.int32)

    def has_result_mask(self, project: Project, image_path: str | Path) -> bool:
        return self.mask_path_for_image(project, image_path).exists()

    # ── ground truth + evaluation ─────────────────────────────────────────────
    @staticmethod
    def find_gt_for_image(image_path: str | Path):
        """Discover a ground-truth mask for ``image_path`` via the same
        sibling-file convention training masks use (``<stem>_mask.*``,
        ``masks/<stem>.*``) — the natural GT-annotation layout already in use
        elsewhere in this app, reused rather than inventing a second one."""
        from studio.train_controller import find_mask_for_image
        return find_mask_for_image(image_path)

    @staticmethod
    def load_gt_mask(gt_path: str | Path, target_shape: tuple[int, int]) -> np.ndarray:
        gt_path = Path(gt_path)
        if gt_path.suffix.lower() == ".npy":
            gt = np.load(str(gt_path))
        else:
            import cv2
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if gt is None:
                raise ValueError(f"Cannot read ground truth: {gt_path}")
        gt = np.ascontiguousarray(gt).astype(np.int32)
        if gt.shape != tuple(target_shape):
            import cv2
            gt = cv2.resize(gt.astype(np.float32), (target_shape[1], target_shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return gt

    def evaluate_against_gt(self, gt_path: str | Path, pred_mask: np.ndarray) -> dict:
        gt = self.load_gt_mask(gt_path, pred_mask.shape)
        return self.evaluate_masks(gt, pred_mask)

    @staticmethod
    def evaluate_masks(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
        """Like :meth:`evaluate_against_gt`, but for a GT mask already loaded
        into memory (e.g. as a workspace Ground-truth layer) — no file path
        or resizing involved, just the metrics."""
        from velum_core import benchmark
        return benchmark.evaluate(gt_mask, pred_mask)

    def discover_gt_pairs(self, project: Project) -> list[tuple[Path, Path]]:
        """Every project image that has a discoverable ground-truth mask —
        what "Benchmark engines vs GT" runs over."""
        pairs = []
        for p in project.image_paths:
            gt = self.find_gt_for_image(p)
            if gt is not None:
                pairs.append((Path(p), gt))
        return pairs

    # ── batch prediction ─────────────────────────────────────────────────────
    def run_batch_async(self, project: Project, *, out_dir: Optional[Path | str] = None,
                        on_log: Optional[Callable[[str], None]] = None,
                        on_progress: Optional[Callable[[int, int], None]] = None,
                        on_cohort_ready: Optional[Callable] = None,
                        on_finish: Optional[Callable[[], None]] = None) -> threading.Thread:
        if not project.image_paths:
            raise ValueError("This project has no images to batch-predict")
        out_dir = Path(out_dir) if out_dir is not None else (self.project_run_dir(project) / "batch")
        out_dir.mkdir(parents=True, exist_ok=True)
        config = self.build_config(project, project.image_paths[0])
        images = [Path(p) for p in project.image_paths]

        def _cohort_ready(records, out_dir_):
            # Mutate project.stats in memory (a full batch really did cover
            # every image, so progress=100 is honest here — unlike a single
            # predict run, see record_run's docstring). Saving is still the
            # caller's job, same convention as everywhere else in this file.
            pop = self.population_stats(records)
            self.record_run(project, n_cells=pop.get("total_cells", 0), progress=100)
            # PredictController.run_batch_async already wrote each mask to
            # out_dir/<stem>_mask.png — copy each into the same per-(project,
            # image) cache a single predict uses, so reopening any of these
            # images later shows its batch-computed result too, not just a
            # single Run's.
            import shutil
            for img_path in images:
                src = Path(out_dir_) / f"{img_path.stem}_mask.png"
                if src.exists():
                    try:
                        shutil.copy2(src, self.mask_path_for_image(project, img_path))
                    except OSError:
                        pass
            if on_cohort_ready:
                on_cohort_ready(records, out_dir_)

        return self._pc().run_batch_async(
            config, images, out_dir, project.settings.pixel_size_um,
            on_log=on_log, on_progress=on_progress, on_cohort_ready=_cohort_ready,
            on_finish=on_finish)

    def stop_batch(self) -> None:
        self._pc().stop_batch()

    @staticmethod
    def population_stats(records: list) -> dict:
        from velum_core import cohort
        return cohort.population_stats(records)

    # ── benchmark engines vs GT ──────────────────────────────────────────────
    def run_benchmark_async(self, project: Project, *,
                            on_row: Optional[Callable[[str], None]] = None,
                            on_log: Optional[Callable[[str], None]] = None,
                            on_done: Optional[Callable] = None) -> threading.Thread:
        self._ensure_engines_registered()
        from velum_core.engine_registry import all_engines

        pairs = self.discover_gt_pairs(project)
        if not pairs:
            raise ValueError(
                "No images with a discoverable ground-truth mask in this project "
                "(expected <image>_mask.* next to it, or a masks/ subfolder)")

        engines: list[str] = []
        bases: dict[str, dict] = {}
        for spec in all_engines():
            if not spec.available():
                if on_log:
                    on_log(f"  [SKIP] {spec.label} — not installed")
                continue
            try:
                params = self.build_params(project, pairs[0][0])
                params["engine"] = spec.key
                from velum_core.predict_controller import PredictController
                bases[spec.key] = PredictController.build_config(params)
                engines.append(spec.key)
            except ValueError as e:
                if on_log:
                    on_log(f"  [SKIP] {spec.label} — {e}")
        if not engines:
            raise ValueError("No engine is ready to benchmark (check LoRA/backbone/SAM2 setup)")

        img_dir = pairs[0][0].parent
        return self._pc().run_benchmark_async(
            engines, bases, pairs, img_dir, on_row=on_row, on_log=on_log, on_done=on_done)

    # ── Dashboard integration ─────────────────────────────────────────────────
    @staticmethod
    def record_run(project: Project, *, n_cells: Optional[int] = None,
                   f1: Optional[float] = None, progress: Optional[int] = None) -> None:
        """Mutate ``project.stats`` from a completed run. Doesn't persist —
        the caller saves via its own ``ProjectStore`` (see module docstring)
        — but once saved, ``DashboardController.runs_table()`` picks this
        project up automatically on the next Dashboard visit: this is the
        whole "segmentation activity shows up in the dashboard" hook.
        """
        if n_cells is not None:
            project.stats.n_cells = int(n_cells)
        if f1 is not None:
            project.stats.last_f1 = float(f1)
        if progress is not None:
            project.stats.progress = max(0, min(100, int(progress)))
        project.stats.n_images = len(project.image_paths)
