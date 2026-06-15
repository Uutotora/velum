from pathlib import Path

import streamlit as st

from data.utils import read_image_to_numpy, read_mask_to_numpy, remap_mask_color
from gui.pages.utils.web_utils import (
    PRED_MASK_DIR,
    SUPPORT_EXTENSION,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    delete_file,
    inject_css,
    initialize_session_state,
    list_files,
    render_sidebar,
)
from visualize_cell import InstanceVisualizer, VisItem, VisStyle, match_masks


def _guard_selection(key, options):
    if st.session_state.get(key, "None") not in options:
        st.session_state[key] = "None"


def file_upload_section():
    with st.container(border=True):
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown('<p class="sec-label">Source images</p>', unsafe_allow_html=True)
            test_image_files = st.file_uploader(
                "Upload images",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="visualize_test_image_files",
                label_visibility="collapsed",
            )
            if test_image_files:
                TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                for f in test_image_files:
                    with open(TEST_IMAGE_DIR / f.name, "wb") as fp:
                        fp.write(f.getbuffer())
                st.success("Images saved.")

            existing = list_files(TEST_IMAGE_DIR)
            with st.expander(f"Files: {len(existing)}", expanded=True):
                if existing:
                    for img in existing:
                        c1, c2 = st.columns([5, 1])
                        c1.text(img)
                        c2.button("✕", key=f"del_vis_img_{img}",
                                  on_click=lambda i=img: delete_file(TEST_IMAGE_DIR, i),
                                  help=f"Remove {img}")
                else:
                    st.caption("No files.")

        with col2:
            st.markdown('<p class="sec-label">Ground truth masks <span style="font-weight:400;color:#94a3b8">(optional)</span></p>', unsafe_allow_html=True)
            test_mask_files = st.file_uploader(
                "Upload ground truth masks",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="visualize_test_mask_files",
                label_visibility="collapsed",
            )
            if test_mask_files:
                TEST_MASK_DIR.mkdir(parents=True, exist_ok=True)
                for f in test_mask_files:
                    with open(TEST_MASK_DIR / f.name, "wb") as fp:
                        fp.write(f.getbuffer())
                st.success("Ground truth masks saved.")

            existing = list_files(TEST_MASK_DIR)
            with st.expander(f"Files: {len(existing)}", expanded=True):
                if existing:
                    for msk in existing:
                        c1, c2 = st.columns([5, 1])
                        c1.text(msk)
                        c2.button("✕", key=f"del_vis_msk_{msk}",
                                  on_click=lambda m=msk: delete_file(TEST_MASK_DIR, m),
                                  help=f"Remove {msk}")
                else:
                    st.caption("No files.")

        with col3:
            st.markdown('<p class="sec-label">Prediction masks <span style="font-weight:400;color:#94a3b8">(from Predict)</span></p>', unsafe_allow_html=True)
            pred_files = st.file_uploader(
                "Upload prediction masks",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="visualize_predict_image_files",
                label_visibility="collapsed",
            )
            if pred_files:
                PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
                for f in pred_files:
                    with open(PRED_MASK_DIR / f.name, "wb") as fp:
                        fp.write(f.getbuffer())
                st.success("Prediction masks saved.")

            existing = list_files(PRED_MASK_DIR)
            with st.expander(f"Files: {len(existing)}", expanded=True):
                if existing:
                    for img in existing:
                        c1, c2 = st.columns([5, 1])
                        c1.text(img)
                        c2.button("✕", key=f"del_vis_pred_{img}",
                                  on_click=lambda i=img: delete_file(PRED_MASK_DIR, i),
                                  help=f"Remove {img}")
                else:
                    st.caption("No files.")


def visualization_settings():
    with st.expander("Display settings"):
        col1, col2, col3, col4, col5, col6 = st.columns(6, vertical_alignment="bottom")
        with col1:
            st.session_state.vis_sync_axes = st.checkbox(
                "Sync zoom", st.session_state.vis_sync_axes,
                help="Pan and zoom all panels together")
        with col2:
            st.session_state.vis_true_mask_color = st.color_picker(
                "Ground truth color", st.session_state.vis_true_mask_color)
        with col3:
            st.session_state.vis_pred_mask_color = st.color_picker(
                "Prediction color", st.session_state.vis_pred_mask_color)
        with col4:
            st.session_state.vis_true_mask_line_width = st.slider(
                "GT line width", 1, 5, st.session_state.vis_true_mask_line_width)
        with col5:
            st.session_state.vis_pred_mask_line_width = st.slider(
                "Pred line width", 1, 5, st.session_state.vis_pred_mask_line_width)
        with col6:
            st.session_state.vis_overlay_alpha = st.slider(
                "Contour opacity", 0.1, 1.0, st.session_state.vis_overlay_alpha, step=0.05,
                help="Opacity of contour overlays")


def build_visualization_items(image, true_mask=None, pred_mask=None):
    if true_mask is not None:
        true_mask = remap_mask_color(true_mask)
    if pred_mask is not None:
        pred_mask = remap_mask_color(pred_mask)

    items = [VisItem(image=image, style=VisStyle(display_mode="image", title="Image"))]

    _alpha = st.session_state.vis_overlay_alpha

    if true_mask is not None and pred_mask is not None:
        matched = match_masks(true_mask, pred_mask)
        items += [
            VisItem(image=image, true_mask=true_mask,
                    style=VisStyle(display_mode="contour", title="Ground Truth",
                                   true_mask_color=st.session_state.vis_true_mask_color,
                                   true_mask_line_width=st.session_state.vis_true_mask_line_width,
                                   overlay_alpha=_alpha)),
            VisItem(image=image, pred_mask=matched,
                    style=VisStyle(display_mode="contour", title="Prediction",
                                   pred_mask_color=st.session_state.vis_pred_mask_color,
                                   pred_mask_line_width=st.session_state.vis_pred_mask_line_width,
                                   overlay_alpha=_alpha)),
            VisItem(image=image, true_mask=true_mask, pred_mask=matched,
                    style=VisStyle(display_mode="error_map", title="Error Map",
                                   error_false_positive_color=st.session_state.vis_error_false_positive_color,
                                   error_false_negative_color=st.session_state.vis_error_false_negative_color)),
            VisItem(image=image, true_mask=true_mask,
                    style=VisStyle(display_mode="mask", title="Ground Truth Mask")),
            VisItem(image=image, pred_mask=matched,
                    style=VisStyle(display_mode="mask", title="Prediction Mask")),
        ]
    elif true_mask is not None:
        items += [
            VisItem(image=image, true_mask=true_mask,
                    style=VisStyle(display_mode="contour", title="Ground Truth",
                                   true_mask_color=st.session_state.vis_true_mask_color,
                                   true_mask_line_width=st.session_state.vis_true_mask_line_width,
                                   overlay_alpha=_alpha)),
            VisItem(image=image, true_mask=true_mask,
                    style=VisStyle(display_mode="mask", title="Ground Truth Mask")),
        ]
    elif pred_mask is not None:
        items += [
            VisItem(image=image, pred_mask=pred_mask,
                    style=VisStyle(display_mode="contour", title="Prediction",
                                   pred_mask_color=st.session_state.vis_pred_mask_color,
                                   pred_mask_line_width=st.session_state.vis_pred_mask_line_width,
                                   overlay_alpha=_alpha)),
            VisItem(image=image, pred_mask=pred_mask,
                    style=VisStyle(display_mode="mask", title="Prediction Mask")),
        ]
    return items


if __name__ == "__main__":
    st.set_page_config(page_title="Visualize — CellSeg1", layout="wide")
    inject_css()
    initialize_session_state()
    render_sidebar()

    st.markdown("## Visualize")
    st.caption("Inspect prediction masks · compare against ground truth · pixel-level error map")
    st.divider()

    file_upload_section()
    visualization_settings()

    st.divider()

    # ── File selectors ────────────────────────────────────────────────────────
    images = list_files(TEST_IMAGE_DIR)
    masks  = list_files(TEST_MASK_DIR)
    preds  = list_files(PRED_MASK_DIR)

    images_opts = ["None"] + images
    masks_opts  = ["None"] + masks
    preds_opts  = ["None"] + preds

    _guard_selection("vis_sel_image", images_opts)
    _guard_selection("vis_sel_mask",  masks_opts)
    _guard_selection("vis_sel_pred",  preds_opts)

    with st.container(border=True):
        autofill_col, _ = st.columns([2, 5])
        with autofill_col:
            if st.button("Auto-fill from Predict", use_container_width=True,
                         help="Pre-select image and prediction mask from the latest Predict run"):
                if images:
                    st.session_state.vis_sel_image = images[0]
                if preds:
                    st.session_state.vis_sel_pred = preds[0]
                    for img in images:
                        if Path(img).stem == Path(preds[0]).stem:
                            st.session_state.vis_sel_image = img
                            break
                st.session_state.vis_figure = None
                st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_image = st.selectbox("Image", images_opts, key="vis_sel_image")
        with col2:
            selected_mask = st.selectbox(
                "Ground truth (optional)", masks_opts, key="vis_sel_mask",
                help="Leave as None if you do not have a hand-annotated mask")
        with col3:
            selected_pred = st.selectbox("Prediction mask", preds_opts, key="vis_sel_pred")

        btn_col, _ = st.columns([2, 5])
        with btn_col:
            generate_clicked = st.button(
                "Generate visualization", type="primary", use_container_width=True)

        if generate_clicked:
            if selected_image != "None":
                try:
                    image     = read_image_to_numpy(TEST_IMAGE_DIR / selected_image)
                    true_mask = None if selected_mask == "None" else read_mask_to_numpy(TEST_MASK_DIR / selected_mask)
                    pred_mask = None if selected_pred == "None" else read_mask_to_numpy(PRED_MASK_DIR / selected_pred)

                    n_cols = 3 if (true_mask is not None or pred_mask is not None) else 1
                    vis = InstanceVisualizer(n_cols=n_cols, sync_axes=st.session_state.vis_sync_axes,
                                            subplot_size=(300, 300))
                    items = build_visualization_items(image, true_mask, pred_mask)
                    st.session_state.vis_figure = vis.plot(items)
                except Exception as e:
                    st.error(f"Visualization error: {e}")
                    st.session_state.vis_figure = None
            else:
                st.warning("Select a source image first.")

    # ── Result ────────────────────────────────────────────────────────────────
    if st.session_state.vis_figure is not None:
        st.plotly_chart(st.session_state.vis_figure, use_container_width=True)

        if selected_pred != "None":
            try:
                import numpy as np
                from PIL import Image as PILImage
                mask_arr = np.array(PILImage.open(PRED_MASK_DIR / selected_pred))
                n_cells = len(np.unique(mask_arr)) - 1
                st.caption(f"Detected cells: **{n_cells}** · mask: `{selected_pred}`")
            except Exception:
                pass
