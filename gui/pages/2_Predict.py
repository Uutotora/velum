import io
import multiprocessing
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import psutil
import streamlit as st

from gui.pages.utils.predict_state_manager import PredictionStateManager
from gui.pages.utils.web_utils import (
    DEVICE_LABELS,
    LORA_PTH_DIR,
    PRED_MASK_DIR,
    SUPPORT_EXTENSION,
    TEST_IMAGE_DIR,
    delete_file,
    get_available_devices,
    get_sam_model_path,
    inject_css,
    initialize_session_state,
    list_files,
    load_default_config,
    render_sidebar,
    show_compact_file_list,
)
from project_root import STORAGE_DIR

LORA_META = {
    "cellpose_specialized_12.pth":     ("General cells",        "Cellpose",  0.917),
    "cellseg_blood_117.pth":           ("Blood cells",          "CSB",       0.941),
    "deepbacs_rod_brightfield_9.pth":  ("E. coli brightfield",  "ECB",       0.860),
    "deepbacs_rod_fluorescence_75.pth":("B. subtilis fluor.",   "BSF",       0.821),
    "dsb2018_stardist_435.pth":        ("Cell nuclei",          "DSB2018",   0.872),
}

LORA_CATEGORY_COLORS = {
    "General cells":       "#2563eb",
    "Blood cells":         "#ef4444",
    "E. coli brightfield": "#16a34a",
    "B. subtilis fluor.":  "#16a34a",
    "Cell nuclei":         "#7c3aed",
}


def main():
    initialize_session_state()
    st.set_page_config(page_title="Predict — CellSeg1", layout="wide")
    inject_css()
    render_sidebar()

    st.markdown("## Predict")
    st.caption("Instance segmentation · SAM + LoRA inference")
    st.divider()

    default_config = load_default_config()

    # ── Input section ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<p class="sec-label">Input images</p>', unsafe_allow_html=True)
        image_files = st.file_uploader(
            "Upload images",
            accept_multiple_files=True,
            type=SUPPORT_EXTENSION,
            key="predict_image_files",
            label_visibility="collapsed",
        )
        if image_files:
            TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            for f in image_files:
                with open(TEST_IMAGE_DIR / f.name, "wb") as fp:
                    fp.write(f.getbuffer())

        existing_images = list_files(TEST_IMAGE_DIR)
        if existing_images:
            show_compact_file_list(TEST_IMAGE_DIR, existing_images, "del_pred_img", show_preview=True)
        else:
            st.caption("No images loaded.")

    with col_right:
        st.markdown('<p class="sec-label">LoRA checkpoint (.pth)</p>', unsafe_allow_html=True)
        lora_file = st.file_uploader(
            "Upload LoRA",
            type=["pth"],
            key="predict_lora_file",
            label_visibility="collapsed",
        )
        if lora_file:
            with open(LORA_PTH_DIR / lora_file.name, "wb") as fp:
                fp.write(lora_file.getbuffer())

        existing_loras = list_files(LORA_PTH_DIR, extensions=[".pth"])
        if existing_loras:
            for lora in existing_loras:
                desc, dataset, mAP = LORA_META.get(lora, ("Custom", "", 0.0))
                color = LORA_CATEGORY_COLORS.get(desc, "#64748b")
                c_card, c_del = st.columns([14, 1], gap="small", vertical_alignment="center")
                with c_card:
                    st.markdown(
                        f"""<div style="border-left:3px solid {color};padding:7px 14px;
                            background:#f8fafc;border-radius:0 4px 4px 0;margin-bottom:3px;
                            display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <span style="font-weight:600;font-size:13px;color:#1e293b;">{desc}</span>
                                <span style="margin-left:8px;font-size:10.5px;color:#94a3b8;
                                    background:#e2e8f0;padding:1px 7px;border-radius:3px;">{dataset}</span><br>
                                <span style="font-family:monospace;font-size:11px;color:#94a3b8;">{lora}</span>
                            </div>
                            <span style="font-weight:700;font-size:15px;color:{color};margin-left:16px;">
                                {f"mAP {mAP:.3f}" if mAP else ""}
                            </span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with c_del:
                    st.button("✕", key=f"del_lora_{lora}",
                              on_click=lambda l=lora: delete_file(LORA_PTH_DIR, l),
                              help=f"Remove {lora}")
        else:
            st.caption("No LoRA checkpoints.")

    st.divider()

    # ── Advanced settings ─────────────────────────────────────────────────────
    with st.expander("Advanced settings"):
        st.markdown('<p class="sec-label">Model — must match training</p>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.selectbox("SAM type", ["vit_h", "vit_l", "vit_b"],
                         key="predict_selected_sam_type",
                         help="Must match the SAM type used during training.")
        with mc2:
            sam_image_size = st.number_input("SAM image size", 64, 1024, 512, 64,
                                             help="Must match the SAM image size used during training.")
        with mc3:
            lora_rank = st.number_input("LoRA rank", 1, 30, 4, 1,
                                        help="Must match the LoRA rank used during training.")

        st.divider()
        st.markdown('<p class="sec-label">Inference</p>', unsafe_allow_html=True)
        ic1, ic2, ic3, ic4, ic5 = st.columns(5)
        with ic1:
            points_per_side = st.number_input("Points per side", 0, 64,
                                              default_config["points_per_side"],
                                              help="Grid density for mask proposal generation. Higher = more candidates, slower.")
        with ic2:
            crop_n_layers = st.number_input("Crop layers", 0, 5,
                                            default_config["crop_n_layers"],
                                            help="Number of image crop layers for small-object detection.")
        with ic3:
            iou_threshold = st.number_input("IoU threshold", 0.0, 0.95,
                                            float(default_config["pred_iou_thresh"]), 0.05,
                                            help="Min predicted IoU to keep a mask. Higher = fewer false positives.")
        with ic4:
            stability_threshold = st.number_input("Stability score", 0.0, 0.95,
                                                   float(default_config["stability_score_thresh"]), 0.05,
                                                   help="Mask stability threshold. Higher = keeps only confident masks.")
        with ic5:
            nms_thresh = st.number_input("NMS threshold", 0.0, 1.0,
                                         float(default_config["box_nms_thresh"]), 0.05,
                                         help="Non-max suppression overlap threshold. Lower = merges more overlapping masks.")

    # ── Run ───────────────────────────────────────────────────────────────────
    st.markdown("### Run")

    available_loras = list(reversed(list_files(LORA_PTH_DIR, extensions=[".pth"])))

    state_manager = PredictionStateManager(STORAGE_DIR)
    running_state = state_manager.load_prediction_state()
    if running_state:
        pid = running_state["process_id"]
        if psutil.pid_exists(pid):
            is_running = True
        else:
            state_manager.clear_prediction_state()
            is_running = False
    else:
        is_running = False

    rc1, rc2, rc3, rc4, rc5 = st.columns([3, 3, 2, 2, 2], vertical_alignment="bottom")

    with rc1:
        lora_labels = {k: f"{v[0]}  ·  mAP {v[2]:.3f}" for k, v in LORA_META.items()}
        if available_loras:
            selected_lora = st.selectbox(
                "Checkpoint",
                options=available_loras,
                format_func=lambda f: lora_labels.get(f, f),
                key="selected_lora",
            )
        else:
            selected_lora = None
            st.selectbox("Checkpoint", ["— no models —"], disabled=True)

    with rc2:
        selected_device = st.selectbox(
            "Device",
            options=get_available_devices(),
            format_func=lambda d: DEVICE_LABELS.get(d, f"GPU {d}"),
        )

    with rc3:
        start_btn = st.button("▶ Run", type="primary", use_container_width=True,
                              disabled=is_running or not selected_lora)
    with rc4:
        stop_btn = st.button("■ Stop", type="secondary", use_container_width=True,
                             disabled=not is_running)
    with rc5:
        has_masks = PRED_MASK_DIR.exists() and any(PRED_MASK_DIR.iterdir())
        if has_masks:
            st.download_button(
                "↓ Download masks",
                data=create_zip_file(PRED_MASK_DIR),
                file_name="predicted_masks.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.button("↓ Download masks", disabled=True, use_container_width=True)

    # ── Actions ───────────────────────────────────────────────────────────────
    if start_btn:
        images = list_files(TEST_IMAGE_DIR)
        if validate_inputs(images, available_loras):
            lora_path = LORA_PTH_DIR / selected_lora if selected_lora else None
            sam_type = st.session_state.get("predict_selected_sam_type", "vit_h")
            config = build_config(
                default_config, str(lora_path),
                points_per_side, crop_n_layers, 0, 0,
                sam_type, sam_image_size,
                iou_threshold, stability_threshold, lora_rank, nms_thresh,
            )
            config["selected_device"] = selected_device
            config["image_paths"] = [str(TEST_IMAGE_DIR / img) for img in images]

            if PRED_MASK_DIR.exists():
                for f in PRED_MASK_DIR.glob("*"):
                    f.unlink()
            else:
                PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
            config["output_dir"] = PRED_MASK_DIR

            proc = multiprocessing.Process(target=run_prediction, args=(config, state_manager))
            proc.start()
            state_manager.save_prediction_state(proc.pid, datetime.now(), len(images))

            if images:
                st.session_state.vis_sel_image = images[0]
                st.session_state.vis_sel_pred = images[0]
                st.session_state.vis_figure = None

            st.rerun()
        else:
            st.error("Load at least one image and one LoRA checkpoint before running.")

    if stop_btn and is_running:
        state_manager.set_stop_flag()
        st.warning("Stop signal sent — will halt after the current image finishes.")

    # ── Progress ──────────────────────────────────────────────────────────────
    if is_running:
        pid = running_state["process_id"]
        if psutil.pid_exists(pid):
            progress_data = state_manager.load_progress()
            done          = progress_data.get("progress", 0)
            total         = progress_data.get("total", running_state.get("total_images", 1))
            current_image = progress_data.get("current_image", "")
            img_started   = progress_data.get("image_started_at")
            elapsed       = datetime.now() - running_state["start_time"]

            # Time-based smooth progress within the current image.
            # SAM on MPS takes ~70–90 s/image; we cap at 95 % so the bar
            # never "stall" at 100 % before the process actually exits.
            SECS_PER_IMAGE = 80.0
            if current_image and img_started is not None:
                img_elapsed = time.time() - img_started
                img_frac = min(img_elapsed / SECS_PER_IMAGE, 0.95)
                frac = (done + img_frac) / max(total, 1)
            else:
                frac = done / max(total, 1)

            eta_str = "—"
            if done > 0:
                secs_left = int((elapsed.total_seconds() / done) * (total - done))
                eta_str = str(timedelta(seconds=max(secs_left, 0)))

            if current_image:
                st.markdown(
                    f'<p style="color:#2563eb;font-size:13px;margin:4px 0;">'
                    f'Predicting <strong>{current_image}</strong> &nbsp;·&nbsp; '
                    f'image {done + 1} of {total}</p>',
                    unsafe_allow_html=True,
                )
            st.progress(frac)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Processed", f"{done} / {total}")
            m2.metric("Progress",  f"{frac * 100:.1f}%")
            m3.metric("Elapsed",   str(elapsed).split(".")[0])
            m4.metric("ETA",       eta_str)
            time.sleep(1)
            st.rerun()
        else:
            state_manager.clear_prediction_state()
            st.rerun()
    else:
        if has_masks:
            pred_list = [f for f in PRED_MASK_DIR.iterdir() if f.is_file()]
            n = len(pred_list)
            _c1, _c2 = st.columns([7, 2], vertical_alignment="center")
            with _c1:
                st.markdown(
                    f'<p style="color:#16a34a;font-weight:600;margin:6px 0;font-size:14px;">'
                    f'&#10003; Done — {n} mask(s) ready</p>',
                    unsafe_allow_html=True,
                )
            with _c2:
                if st.button("Open in Visualize →", key="go_to_visualize", use_container_width=True):
                    st.switch_page("pages/3_Visualize.py")

            if n > 1:
                _render_batch_grid()
            _render_predict_preview()
        else:
            st.caption("No prediction running.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _colorize_instance_mask(mask) -> "np.ndarray":
    import colorsys
    import numpy as np
    labels = sorted(set(mask.ravel().tolist()) - {0})
    rgb = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    n = max(len(labels), 1)
    for j, lbl in enumerate(labels):
        h = (j * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.88)
        rgb[mask == lbl] = [int(r * 255), int(g * 255), int(b * 255)]
    return rgb


def _render_batch_grid() -> None:
    """Grid overview of all predicted masks — shown when >1 image was predicted."""
    import numpy as np
    from PIL import Image as PILImage

    pred_files = sorted([f for f in PRED_MASK_DIR.iterdir() if f.is_file()])
    test_stems = {Path(f).stem: TEST_IMAGE_DIR / f for f in list_files(TEST_IMAGE_DIR)}

    st.markdown('<p class="sec-label">Batch preview</p>', unsafe_allow_html=True)
    n = len(pred_files)
    n_cols = min(4, n)
    cols = st.columns(n_cols)

    for i, pred_f in enumerate(pred_files):
        with cols[i % n_cols]:
            try:
                mask_arr = np.array(PILImage.open(pred_f))
                n_cells = int(mask_arr.max())
                rgb = _colorize_instance_mask(mask_arr)
                thumb = PILImage.fromarray(rgb)
                thumb.thumbnail((220, 220))

                # Composite: blend with source if available
                src_path = test_stems.get(pred_f.stem)
                if src_path and src_path.exists():
                    src = PILImage.open(src_path).convert("RGB")
                    src.thumbnail((220, 220))
                    # Blend: source at 50%, mask at 50%
                    thumb_resized = thumb.resize(src.size, PILImage.NEAREST)
                    composite = PILImage.blend(src, thumb_resized, alpha=0.55)
                    st.image(composite, use_container_width=True)
                else:
                    st.image(thumb, use_container_width=True)
                st.caption(f"{pred_f.stem}  ·  {n_cells} cells")
            except Exception as e:
                st.caption(f"{pred_f.name} — error: {e}")


def _render_predict_preview() -> None:
    """Inline before/after: original image + prediction contour side by side."""
    pred_files = sorted([f for f in PRED_MASK_DIR.iterdir() if f.is_file()])
    test_images = list_files(TEST_IMAGE_DIR)
    if not pred_files or not test_images:
        return

    # Match predictions to source images by stem
    pairs = []
    for pred_f in pred_files:
        for img_name in test_images:
            if Path(img_name).stem == pred_f.stem:
                pairs.append((img_name, pred_f))
                break
    if not pairs:
        return

    # Selector when multiple images were predicted
    if len(pairs) > 1:
        opts = [p[0] for p in pairs]
        sel = st.selectbox("Preview image", opts, key="predict_preview_sel")
        pair = next((p for p in pairs if p[0] == sel), pairs[0])
    else:
        pair = pairs[0]

    img_name, pred_path = pair

    try:
        import numpy as np
        from data.utils import read_image_to_numpy, read_mask_to_numpy, remap_mask_color
        from visualize_cell import InstanceVisualizer, VisItem, VisStyle

        image = read_image_to_numpy(TEST_IMAGE_DIR / img_name)
        pred_mask = read_mask_to_numpy(pred_path)
        pred_mask = remap_mask_color(pred_mask)
        n_cells = int(np.unique(pred_mask).size) - 1

        vis = InstanceVisualizer(n_cols=2, sync_axes=True, subplot_size=(380, 380))
        items = [
            VisItem(image=image, style=VisStyle(display_mode="image", title="Original")),
            VisItem(
                image=image,
                pred_mask=pred_mask,
                style=VisStyle(
                    display_mode="contour",
                    title=f"Prediction — {n_cells} cells",
                    pred_mask_color="#64FF64",
                    pred_mask_line_width=1,
                ),
            ),
        ]
        fig = vis.plot(items)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Detected: **{n_cells} cells** · {pred_path.name}")
    except Exception as e:
        st.caption(f"Preview unavailable: {e}")


def create_zip_file(directory):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in directory.glob("*"):
            if f.is_file():
                zf.write(f, f.name)
    return buf.getvalue()


def validate_inputs(images, loras):
    if not images:
        st.error("No input images.")
        return False
    if not loras:
        st.error("No LoRA checkpoint selected.")
        return False
    return True


def build_config(default_config, lora_path, points_per_side, crop_n_layers,
                 rw, rh, sam_type, sam_image_size,
                 iou_thresh, stability_thresh, lora_rank, nms_thresh):
    cfg = default_config.copy()
    cfg.update({
        "result_pth_path": lora_path,
        "points_per_side": points_per_side,
        "crop_n_layers": crop_n_layers,
        "resize_size": [rw, rh] if rw > 0 and rh > 0 else None,
        "vit_name": sam_type,
        "model_path": get_sam_model_path(sam_type),
        "sam_image_size": sam_image_size,
        "iou_thresh": iou_thresh,
        "stability_score_thresh": stability_thresh,
        "image_encoder_lora_rank": lora_rank,
        "mask_decoder_lora_rank": lora_rank,
        "box_nms_thresh": nms_thresh,
        "pred_iou_thresh": nms_thresh,
    })
    return cfg


def run_prediction(config, state_manager):
    import os
    import time as _time
    device = config.get("selected_device", "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    from pathlib import Path
    import cv2
    from data.utils import read_image_to_numpy, resize_image
    from predict import predict_images

    image_paths = config["image_paths"]
    images = [read_image_to_numpy(p) for p in image_paths]
    if config.get("resize_size") and config["resize_size"][0] > 0:
        images = [resize_image(img, config["resize_size"]) for img in images]

    total = len(images)
    masks, save = [], True
    try:
        for idx, img in enumerate(images):
            if state_manager.check_stop_flag():
                save = False
                break
            img_name = Path(image_paths[idx]).name
            # Save timestamp BEFORE the slow predict call so the UI can animate
            state_manager.save_progress(idx, total, img_name, _time.time())
            masks.append(predict_images(config, [img])[0])
            state_manager.save_progress(idx + 1, total)

        if save:
            out = config["output_dir"]
            out.mkdir(parents=True, exist_ok=True)
            for path, mask in zip(image_paths, masks):
                cv2.imwrite(str((out / Path(path).name).with_suffix(".png")), mask)
    finally:
        state_manager.clear_prediction_state()


if __name__ == "__main__":
    main()
