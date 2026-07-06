"""Prediction core: config building + predict/batch/benchmark orchestration.

Split out of the ``PredictWidget`` god-object so this logic is testable
without Qt/torch/napari. Everything here is plain Python — heavy deps
(torch via inference_cache, cv2, cellpose) are imported lazily inside the
functions that need them, exactly as they were in the widget, so importing
this module stays cheap. ``PredictWidget`` owns a single ``PredictController``
instance and wires its callbacks to Qt signals; the widget only builds the UI
and reads/writes widget state.
"""
import threading
from pathlib import Path

import numpy as np

from napari_app import engines as _builtin_engines  # noqa: F401 — registers built-in engines
from napari_app.engine_registry import get as get_engine, all_engines

ENGINE_LABELS = {spec.key: spec.result_label for spec in all_engines()}


# ── Read + inference core (module functions: the "service" layer) ────────────

def _to_display_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce an image to 8-bit for engines that require uint8 (e.g. SAM).

    uint8 input is returned unchanged (the default path stays byte-for-byte).
    Higher bit-depth or float input (16-bit PNG/TIFF, e.g. a fluorescence image
    or a uint16 label/GT file) is percentile-stretched (1–99%) into 0–255 —
    the same normalisation the multi-channel path already uses — so SAM no
    longer raises "Input type uint16 is not supported".
    """
    if img.dtype == np.uint8:
        return img
    a = img.astype(np.float32)
    lo = float(np.percentile(a, 1.0))
    hi = float(np.percentile(a, 99.0))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    a = (a - lo) / (hi - lo)
    return (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


def _read_for_predict(config):
    """Read ``config['image_path']`` into ``(rgb_uint8_HxWx3, stack_or_None)``.

    Default path (no ``channels`` key): the exact legacy ``cv2`` BGR→RGB read
    and a ``None`` stack, so ordinary RGB/grayscale images are byte-for-byte
    unchanged. Multi-channel path (opt-in via a ``channels`` list of channel
    indices): read the full-depth stack with tifffile, percentile-normalise and
    project the selected channels to the RGB frame the engine expects, and also
    return the raw :class:`~napari_app.channels.ChannelStack` for per-channel
    intensity measurement.
    """
    channels = config.get("channels")
    # Native microscopy formats (.nd2/.czi/.lif) can't be read by cv2, so they
    # always go through the channel-stack path — projecting the first channels
    # by default when the user hasn't picked any explicitly.
    is_native = Path(config["image_path"]).suffix.lower() in (".nd2", ".czi", ".lif")
    if channels or is_native:
        from napari_app.channels import read_channel_stack, project_to_rgb
        stack = read_channel_stack(config["image_path"])
        rgb = project_to_rgb(stack, channels,
                             low=float(config.get("channel_low", 1.0)),
                             high=float(config.get("channel_high", 99.0)))
        return rgb, stack

    import cv2
    img = cv2.imread(config["image_path"], cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read: {config['image_path']}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # SAM (and the downstream cv2/torchvision transforms) require uint8; 16-bit
    # and float images crash otherwise. uint8 is returned unchanged.
    img = _to_display_uint8(img)
    return img, None


def _predict_cached(config, on_tile=None, sink=None):
    from data.utils import resize_image
    import cv2

    img, stack = _read_for_predict(config)
    if sink is not None:
        sink["stack"] = stack

    # Large-image path: tile at native resolution instead of shrinking the
    # whole image (which loses small cells). Opt-in via the "Large image" box.
    from napari_app.tiling import should_tile
    if config.get("tiled") and should_tile(img.shape, tile=int(config.get("tile_size") or 1024)):
        return img, _predict_tiled(config, img, on_tile=on_tile)

    orig_h, orig_w = img.shape[:2]
    resized = resize_image(img, config["resize_size"])
    if config.get("clahe"):
        resized = _apply_clahe(resized)

    spec = get_engine(config.get("engine") or "cellseg1")
    small = spec.predict(resized, config)

    if small.shape != (orig_h, orig_w):
        mask = cv2.resize(small.astype(np.float32), (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST).astype(small.dtype)
    else:
        mask = small
    return img, mask


def _predict_tiled(config, img, on_tile=None):
    """Segment a large RGB image tile-by-tile at native resolution and stitch.

    Reuses the exact per-image engine calls of the normal path, applied to each
    overlapping tile; cells crossing a seam are merged by the stitcher. Returns
    a full-resolution instance mask the same H×W as ``img``. ``on_tile(done,
    total)`` is forwarded to the tiler for per-tile progress reporting.
    """
    from napari_app.tiling import recommend_overlap, tiled_predict

    tile = int(config.get("tile_size") or 1024)
    overlap = int(config.get("tile_overlap") or 0)
    if overlap <= 0:
        overlap = recommend_overlap(float(config.get("cp_diameter") or 0), tile)

    spec = get_engine(config.get("engine") or "cellseg1")

    def _fn(t):
        if config.get("clahe"):
            t = _apply_clahe(t)
        return spec.predict(t, config)

    min_area = int(config.get("min_mask_area") or config.get("min_mask_region_area") or 0)
    return tiled_predict(img, _fn, tile=tile, overlap=overlap, min_area=min_area,
                         on_tile=on_tile)


def _apply_clahe(rgb: np.ndarray) -> np.ndarray:
    """Adaptive histogram equalisation on the luminance channel (uint8 RGB)."""
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


# ── Controller: config building + predict/batch/benchmark orchestration ──────

class PredictController:
    """Owns prediction config-building and predict/batch/benchmark
    orchestration, independent of Qt.

    Config-building methods take a plain ``params`` dict (every current UI
    value, gathered by the widget — see ``PredictWidget._gather_params``) and
    return the engine config dict; they raise ``ValueError`` on bad input,
    exactly as the widget methods they replace did. Orchestration methods spawn
    the same daemon-thread background work the widget used to, but report
    progress/results/errors through plain callback functions instead of Qt
    signals, so the widget's only job is to connect those callbacks to its
    existing signals.
    """

    def __init__(self):
        self._batch_stop = threading.Event()

    # ---- config building -----------------------------------------------

    @staticmethod
    def build_config(params: dict) -> dict:
        """Build the engine config dict from a snapshot of UI values.

        Mirrors the previous ``PredictWidget._build_config`` 1:1: Cellpose-SAM
        needs no LoRA/SAM checkpoint (a much shorter config); everything else
        delegates to :meth:`sam_config`.
        """
        img = params["image_path"]
        if not img or not Path(img).exists():
            raise ValueError(f"Image not found: {img}")
        rs = int(params["resize_size"])

        if params["engine"] == "cellpose":
            if not get_engine("cellpose").available():
                raise ValueError("Cellpose is not installed — run: pip install cellpose")
            return {
                "engine": "cellpose", "image_path": img,
                "resize_size": [rs, rs],
                "cp_diameter": params["cp_diameter"],
                "cp_flow_threshold": params["cp_flow_threshold"],
                "cp_cellprob_threshold": params["cp_cellprob_threshold"],
                "selected_device": params["device"],
                "clahe": params["clahe"],
                "tiled": params["tiled"],
                "tile_size": rs, "tile_overlap": 0,
                # kept so downstream (refine, caching keys) stays valid
                "vit_name": params["vit_name"],
                "image_encoder_lora_rank": params["lora_rank"],
                "sam_image_size": rs, "result_pth_path": "",
                "channels": params["channels"],
            }

        return PredictController.sam_config(params)

    @staticmethod
    def sam_config(params: dict) -> dict:
        """Full SAM + LoRA config. Used by the CellSeg1 engine and always by
        the interactive Annotate session (which needs SAM regardless of the
        engine selector). Requires an image, a LoRA checkpoint and a SAM
        backbone."""
        img = params["image_path"]
        if not img or not Path(img).exists():
            raise ValueError(f"Image not found: {img}")
        lora = PredictController.resolve_lora(
            params["lora_custom_text"], params["lora_combo_text"], params["lora_paths"])
        sam = PredictController.resolve_sam(
            params["sam_path_text"], params["vit_name"], params["storage_dir"])
        if not lora or not Path(lora).exists():
            raise ValueError(f"LoRA checkpoint not found: {lora}")
        rs = int(params["resize_size"])
        return {
            "engine": "cellseg1",
            "vit_name": params["vit_name"],
            "model_path": sam, "result_pth_path": lora, "image_path": img,
            "image_encoder_lora_rank": params["lora_rank"],
            "mask_decoder_lora_rank":  params["lora_rank"],
            "freeze_image_encoder": True, "freeze_prompt_encoder": True,
            "freeze_mask_decoder_transformer": True, "freeze_upscaling_cnn": True,
            "freeze_output_hypernetworks_mlps": True,
            "freeze_mask_decoder_mask_tokens": True, "freeze_mask_decoder_iou": True,
            "lora_dropout": 0.1,
            "sam_image_size": rs, "resize_size": [rs, rs],
            "points_per_side":          params["points_per_side"],
            "points_per_batch":         64,
            "pred_iou_thresh":          params["pred_iou_thresh"],
            "stability_score_thresh":   params["stability_score_thresh"],
            "stability_score_offset":   0.8,
            "box_nms_thresh":           params["box_nms_thresh"],
            "crop_nms_thresh": 0.05, "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area":     params["min_mask_area"],
            "max_mask_region_area_ratio": 0.1,
            "selected_device": params["device"],
            "deterministic": True, "seed": 0,
            "allow_tf32_on_cudnn": True, "allow_tf32_on_matmul": True,
            "clahe": params["clahe"],
            "tiled": params["tiled"],
            "tile_size": rs, "tile_overlap": 0,
            "channels": params["channels"],
        }

    @staticmethod
    def resolve_lora(lora_custom_text: str, lora_combo_text: str, lora_paths: dict) -> str:
        if lora_custom_text and lora_custom_text.strip():
            return lora_custom_text.strip()
        return lora_paths.get(lora_combo_text, "")

    @staticmethod
    def resolve_sam(sam_path_text: str, vit_name: str, storage_dir) -> str:
        p = (sam_path_text or "").strip()
        if p and Path(p).exists():
            return p
        names = {"vit_h": "sam_vit_h_4b8939.pth",
                 "vit_l": "sam_vit_l_0b3195.pth",
                 "vit_b": "sam_vit_b_01ec64.pth"}
        c = Path(storage_dir) / "sam_backbone" / names[vit_name]
        if c.exists():
            return str(c)
        raise ValueError(
            f"SAM backbone not found. Place {names[vit_name]} in "
            f"{Path(storage_dir) / 'sam_backbone'}/")

    # ---- orchestration ---------------------------------------------------

    def run_prediction_async(self, config: dict, *, on_tile=None, on_result=None,
                              on_log=None, on_finish=None) -> threading.Thread:
        """Predict ``config['image_path']`` on a background daemon thread.

        Mirrors the previous ``PredictWidget._run_prediction`` inner closure:
        ``on_result(img_arr, mask, stack)`` fires once on success — before any
        "done" log line, so a caller can stash the channel stack first —
        ``on_log`` carries the "✓ N cells" line (followed by a "[HINT]" line
        if the image was large enough that "Large image" tiling would have
        helped but was off) and any error text, and ``on_finish`` always
        fires last, success or failure.
        """
        def run():
            try:
                sink = {}
                img_arr, mask = _predict_cached(config, on_tile=on_tile, sink=sink)
                if on_result:
                    on_result(img_arr, mask, sink.get("stack"))
                if on_log:
                    spec = get_engine(config.get("engine") or "cellseg1")
                    status = spec.status_line() if spec.status_line else spec.result_label
                    on_log(f"✓ {int(mask.max())} cells  [{status}]")
                    from napari_app.tiling import should_warn_no_tiling
                    tile = int(config.get("tile_size") or 1024)
                    if should_warn_no_tiling(img_arr.shape, bool(config.get("tiled")), tile=tile):
                        on_log(
                            "[HINT] Large image — inference resized it, which can lose "
                            "small cells. Enable \"Large image: tile at native "
                            "resolution\" for full detail.")
            except Exception as e:
                import traceback
                if on_log:
                    on_log(f"[ERROR] {e}\n{traceback.format_exc()}")
            finally:
                if on_finish:
                    on_finish()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def run_batch_async(self, config: dict, images: list, out_dir: Path,
                         pixel_size_um: float, *, on_log=None, on_progress=None,
                         on_cohort_ready=None, on_finish=None) -> threading.Thread:
        """Predict every image in ``images``, save masks to ``out_dir``, and
        (unless stopped early via :meth:`stop_batch`) write cohort CSVs.

        ``on_cohort_ready(records, out_dir)`` fires only when the batch runs to
        completion — mirrors the previous ``for/else`` in
        ``PredictWidget._run_batch``, where a stop mid-batch skips the cohort
        step entirely.
        """
        self._batch_stop.clear()

        def run():
            import cv2
            from napari_app import analysis
            n = len(images)
            done = 0
            records = []
            for img_path in images:
                if self._batch_stop.is_set():
                    if on_log:
                        on_log(f"■ Stopped at {done}/{n}")
                    break
                if on_log:
                    on_log(f"[{done + 1}/{n}] {img_path.name}")
                try:
                    cfg = {**config, "image_path": str(img_path)}
                    img_arr, mask = _predict_cached(cfg)
                    cv2.imwrite(str(out_dir / f"{img_path.stem}_mask.png"),
                                mask.astype(np.uint16))
                    result = analysis.compute_measurements(
                        mask, intensity_image=img_arr, pixel_size_um=pixel_size_um)
                    cov = float((mask > 0).sum()) / mask.size * 100.0
                    records.append((img_path.name, result, cov))
                except Exception as e:
                    if on_log:
                        on_log(f"  [ERROR] {e}")
                done += 1
                if on_progress:
                    on_progress(done, n)
            else:
                try:
                    from napari_app import cohort
                    cohort.write_cohort_csvs(out_dir, records)
                    pop = cohort.population_stats(records)
                    if on_log:
                        on_log(
                            f"✓ Batch done — {n} masks + cohort CSVs in {out_dir.name}/  "
                            f"({pop['total_cells']} cells across {pop['n_images']} images)")
                except Exception as e:
                    if on_log:
                        on_log(f"  [WARN] cohort analysis failed: {e}")
                if on_cohort_ready:
                    on_cohort_ready(records, out_dir)
            if on_finish:
                on_finish()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def stop_batch(self):
        self._batch_stop.set()

    def run_benchmark_async(self, engines: list, bases: dict, pairs: list,
                             img_dir: Path, *, on_row=None, on_log=None,
                             on_done=None) -> threading.Thread:
        """Run every (engine × image) pair against its ground truth, then hand
        the aggregated results table back via ``on_done(cols, rows)``.

        ``on_row`` fires once per pair regardless of success ("N / total
        (engine)"); ``on_log`` carries per-pair error lines.
        """
        total = len(pairs) * len(engines)

        def run():
            from napari_app import benchmark
            import cv2
            per_engine = {e: [] for e in engines}
            done = 0
            for eng in engines:
                for img_path, gt_path in pairs:
                    try:
                        cfg = {**bases[eng], "image_path": str(img_path)}
                        _, pred = _predict_cached(cfg)
                        if str(gt_path).lower().endswith(".npy"):
                            gt = np.load(str(gt_path))
                        else:
                            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
                        gt = np.ascontiguousarray(gt).astype(np.int32)
                        if gt.shape != pred.shape:
                            gt = cv2.resize(gt.astype(np.float32),
                                            (pred.shape[1], pred.shape[0]),
                                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
                        per_engine[eng].append(benchmark.evaluate(gt, pred))
                    except Exception as ex:
                        if on_log:
                            on_log(f"  [ERROR] {eng} {img_path.name}: {ex}")
                    done += 1
                    if on_row:
                        on_row(f"{done} / {total}  ({ENGINE_LABELS[eng]})")
            summaries = {ENGINE_LABELS[e]: benchmark.summarize(per_engine[e])
                         for e in engines}
            cols, rows = benchmark.results_table(summaries)
            try:
                benchmark.write_csv(str(img_dir / "benchmark.csv"), cols, rows)
            except Exception:
                pass
            if on_done:
                on_done(cols, rows)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t
