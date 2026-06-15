from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment
from skimage import measure

from data.utils import normalize_to_uint8, remap_mask_color


def label_overlap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def intersection_over_union(masks_true: np.ndarray, masks_pred: np.ndarray) -> np.ndarray:
    overlap = label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def mask_ious(masks_true: np.ndarray, masks_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    iou = intersection_over_union(masks_true, masks_pred)[1:, 1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iou_values = np.zeros(masks_true.max())
    iou_values[true_ind] = iou[true_ind, pred_ind]
    preds = np.zeros(masks_true.max(), dtype=np.uint16)
    preds[true_ind] = pred_ind + 1
    return iou_values, preds


def match_masks(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    _, match = mask_ious(pred, true)
    unmatched_numbers = sorted(set(range(1, pred.max() + 1)) - set(match))
    j = 0
    for i, c in enumerate(match):
        if c == 0:
            match[i] = unmatched_numbers[j]
            j += 1
    new_pred = np.zeros_like(pred)
    for i, n in enumerate(match):
        new_pred[pred == i + 1] = n
    return new_pred


def find_contours_with_padding(input_mask: np.ndarray):
    masks_padded = np.pad(input_mask, pad_width=1, mode="constant", constant_values=0)
    contours = []
    for i in range(1, masks_padded.max() + 1):
        binary_mask = masks_padded == i
        instance_contours = measure.find_contours(binary_mask, 0.5)
        for contour in instance_contours:
            contour -= 1
            contour[:, 0] = np.clip(contour[:, 0], 0, input_mask.shape[0])
            contour[:, 1] = np.clip(contour[:, 1], 0, input_mask.shape[1])
            contours.append(contour)
    return contours


def plot_image_with_mask(
    image,
    mask=None,
    box=None,
    contour_color="#EFEF00",
    box_color="#5aff15",
    box_width=3,
    contour_width=3,
    box_dash="solid",
    contour_dash="solid",
    fig_size=(500, 500),
):
    image_rgb = image
    if image.ndim == 2:
        image_rgb = np.dstack((image, image, image))
    elif image.ndim == 3 and image.shape[2] == 1:
        image_rgb = np.dstack((image, image, image))

    fig = go.Figure()
    fig.add_trace(go.Image(z=image_rgb))

    if mask is not None:
        contours = find_contours_with_padding(mask)
        x, y = [], []
        for contour in contours:
            y = np.concatenate([y, contour[:, 0], [None]])
            x = np.concatenate([x, contour[:, 1], [None]])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=contour_color, width=contour_width, dash=contour_dash),
                hoverinfo="skip",
            )
        )
    if box is not None:
        x0, x1, y0, y1 = box
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color=box_color, width=box_width, dash=box_dash),
        )

    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, image_rgb.shape[1]]),
        yaxis=dict(visible=False, range=[image_rgb.shape[0], 0]),
    )

    return fig


def separate_masks(
    pred: np.ndarray, true: np.ndarray, iou_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_hit = np.zeros_like(pred)
    pred_miss = np.zeros_like(pred)
    true_hit = np.zeros_like(true)
    true_miss = np.zeros_like(true)

    pred_labels = np.unique(pred)[1:]
    true_labels = np.unique(true)[1:]

    iou_matrix = np.zeros((len(pred_labels), len(true_labels)))

    for i, pred_label in enumerate(pred_labels):
        pred_mask = pred == pred_label
        for j, true_label in enumerate(true_labels):
            true_mask = true == true_label

            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            iou = intersection / union if union > 0 else 0

            iou_matrix[i, j] = iou

    pred_max_ious = np.max(iou_matrix, axis=1)
    pred_best_matches = np.argmax(iou_matrix, axis=1)  # noqa: F841

    true_max_ious = np.max(iou_matrix, axis=0)
    true_best_matches = np.argmax(iou_matrix, axis=0)  # noqa: F841

    for i, pred_label in enumerate(pred_labels):
        pred_mask = pred == pred_label
        if pred_max_ious[i] > iou_threshold:
            pred_hit[pred_mask] = pred_label
        else:
            pred_miss[pred_mask] = pred_label

    for j, true_label in enumerate(true_labels):
        true_mask = true == true_label
        if true_max_ious[j] > iou_threshold:
            true_hit[true_mask] = true_label
        else:
            true_miss[true_mask] = true_label

    return pred_hit, pred_miss, true_hit, true_miss


def plot_image_with_mask_and_boxes(
    image,
    mask=None,
    boxs=None,
    contour_color="#EFEF00",
    box_color="#5aff15",
    box_width=3,
    contour_width=3,
    box_dash="solid",
    contour_dash="solid",
    fig_size=(500, 500),
):
    image_rgb = image
    if image.ndim == 2:
        image_rgb = np.dstack((image, image, image))
    elif image.ndim == 3 and image.shape[2] == 1:
        image_rgb = np.dstack((image, image, image))

    fig = go.Figure()
    fig.add_trace(go.Image(z=image_rgb))

    if mask is not None:
        contours = find_contours_with_padding(mask)
        x, y = [], []
        for contour in contours:
            y = np.concatenate([y, contour[:, 0], [None]])
            x = np.concatenate([x, contour[:, 1], [None]])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=contour_color, width=contour_width, dash=contour_dash),
                hoverinfo="skip",
            )
        )

    if boxs is not None:
        for box in boxs:
            x_min, y_min, x_max, y_max = box
            y0 = y_min
            y1 = y_max
            fig.add_shape(
                type="rect",
                x0=x_min,
                y0=y0,
                x1=x_max,
                y1=y1,
                line=dict(color=box_color, width=box_width, dash=box_dash),
                fillcolor="rgba(0,0,0,0)",
            )

    fig.update_layout(
        width=fig_size[1],
        height=fig_size[0],
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, image_rgb.shape[1]]),
        yaxis=dict(visible=False, range=[image_rgb.shape[0], 0]),
    )

    return fig


@dataclass
class VisStyle:
    show_true_mask: bool = True
    show_pred_mask: bool = True
    show_error_map: bool = False
    show_boxes: bool = False
    show_points: bool = False
    show_annotations: bool = False

    true_mask_color: str = "rgba(255, 255, 0, 1)"
    pred_mask_color: str = "rgba(100, 255, 100, 1)"
    box_color: str = "rgba(0, 255, 0, 1)"
    point_color: str = "rgba(0, 0, 255, 1)"
    annotation_color: str = "rgba(255, 255, 255, 1)"

    error_false_positive_color: str = "rgba(100, 255, 100, 1)"
    error_false_negative_color: str = "rgba(255, 100, 100, 1)"

    true_mask_line_width: float = 1
    pred_mask_line_width: float = 1
    box_line_width: float = 1

    true_mask_line_type: str = "solid"
    pred_mask_line_type: str = "solid"
    box_line_type: str = "solid"

    display_mode: Literal["contour", "mask", "error_map", "image"] = "contour"
    overlay_alpha: float = 0.5

    hover_template: Optional[str] = None
    title: Optional[str] = None


@dataclass
class VisItem:
    image: np.ndarray
    true_mask: Optional[np.ndarray] = None
    pred_mask: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    annotations: Optional[List[str]] = None
    style: Optional[VisStyle] = None


class InstanceVisualizer:
    def __init__(
        self,
        n_cols: int = 4,
        figure_size: Optional[Tuple[int, int]] = None,
        subplot_size: Optional[Tuple[int, int]] = (512, 512),
        sync_axes: bool = True,
    ):
        self.n_cols = n_cols
        self.subplot_size = subplot_size
        self.sync_axes = sync_axes

        if subplot_size is not None:
            subplot_width, subplot_height = subplot_size
            total_width = subplot_width * n_cols + 100
            self.figure_size = (total_width, None)
        else:
            self.figure_size = figure_size

    def _check_image_sizes(self, items: List[VisItem]):
        if not items or not self.sync_axes:
            return

        reference_shape = items[0].image.shape[:2]
        for idx, item in enumerate(items[1:], 1):
            if item.image.shape[:2] != reference_shape:
                raise ValueError(
                    f"Image size mismatch when sync_axes=True: "
                    f"image 0 has shape {reference_shape}, but "
                    f"image {idx} has shape {item.image.shape[:2]}"
                )

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.stack((image,) * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            return np.concatenate((image,) * 3, axis=-1)
        return image

    def _create_contour_trace(
        self,
        contours: List[np.ndarray],
        color: str,
        line_width: float,
        line_type: str,
        hover_template: Optional[str] = None,
        opacity: float = 1.0,
    ) -> go.Scatter:
        x = []
        y = []
        for contour in contours:
            x.extend(contour[:, 1].tolist() + [None])
            y.extend(contour[:, 0].tolist() + [None])

        return go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=line_width, dash=line_type),
            opacity=opacity,
            showlegend=False,
            hovertemplate=hover_template,
            hoverinfo="skip",
        )

    def _calculate_error_map(self, true_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
        error_map = np.zeros_like(true_mask, dtype=np.uint16)
        error_map[np.logical_and(true_mask == 0, pred_mask > 0)] = 1
        error_map[np.logical_and(true_mask > 0, pred_mask == 0)] = 2
        return error_map

    def _create_subplot_traces(self, item: VisItem, style: VisStyle) -> List[go.Figure]:
        traces = []
        image = self._process_image(item.image)

        if style.display_mode == "image":
            traces.append(go.Image(z=image))

        elif style.display_mode == "contour":
            traces.append(go.Image(z=image))

            if item.true_mask is not None and style.show_true_mask:
                contours = find_contours_with_padding(item.true_mask)
                traces.append(
                    self._create_contour_trace(
                        contours,
                        style.true_mask_color,
                        style.true_mask_line_width,
                        style.true_mask_line_type,
                        style.hover_template,
                        opacity=style.overlay_alpha,
                    )
                )

            if item.pred_mask is not None and style.show_pred_mask:
                contours = find_contours_with_padding(item.pred_mask)
                traces.append(
                    self._create_contour_trace(
                        contours,
                        style.pred_mask_color,
                        style.pred_mask_line_width,
                        style.pred_mask_line_type,
                        style.hover_template,
                        opacity=style.overlay_alpha,
                    )
                )

        elif style.display_mode == "mask":
            mask = item.true_mask if item.true_mask is not None else item.pred_mask
            if mask is not None:
                traces.append(
                    go.Heatmap(
                        z=mask,
                        colorscale=[
                            [0.000, "rgb(0,0,0)"],
                            [0.001, "rgb(150,0,90)"],
                            [0.125, "rgb(0,0,200)"],
                            [0.250, "rgb(0,25,255)"],
                            [0.375, "rgb(0,152,255)"],
                            [0.500, "rgb(44,255,150)"],
                            [0.625, "rgb(151,255,0)"],
                            [0.750, "rgb(255,234,0)"],
                            [0.875, "rgb(255,111,0)"],
                            [1.000, "rgb(255,0,0)"],
                        ],
                        showscale=False,
                        hovertemplate=style.hover_template,
                    )
                )

        elif style.display_mode == "error_map" and item.true_mask is not None and item.pred_mask is not None:
            error_map = self._calculate_error_map(item.true_mask, item.pred_mask)
            traces.append(
                go.Heatmap(
                    z=error_map,
                    colorscale=[
                        [0, "black"],
                        [0.33, "black"],
                        [0.34, style.error_false_positive_color],
                        [0.66, style.error_false_positive_color],
                        [0.67, style.error_false_negative_color],
                        [1.0, style.error_false_negative_color],
                    ],
                    showscale=False,
                    hovertemplate=style.hover_template,
                )
            )

        if item.boxes is not None and style.show_boxes:
            for box in item.boxes:
                x = [box[0], box[0], box[2], box[2], box[0]]
                y = [box[1], box[3], box[3], box[1], box[1]]
                traces.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color=style.box_color, width=style.box_line_width, dash=style.box_line_type),
                        showlegend=False,
                    )
                )

        if item.points is not None and style.show_points:
            traces.append(
                go.Scatter(
                    x=item.points[:, 0],
                    y=item.points[:, 1],
                    mode="markers",
                    marker=dict(color=style.point_color, size=8),
                    showlegend=False,
                )
            )

        if item.annotations is not None and style.show_annotations:
            if item.points is not None:
                for point, text in zip(item.points, item.annotations):
                    traces.append(
                        go.Scatter(
                            x=[point[0]],
                            y=[point[1]],
                            mode="text",
                            text=[text],
                            textfont=dict(color=style.annotation_color),
                            showlegend=False,
                        )
                    )

        return traces

    def plot(self, items: List[VisItem]):
        if not items:
            raise ValueError("No items to visualize")

        self._check_image_sizes(items)

        n_items = len(items)
        n_rows = (n_items + self.n_cols - 1) // self.n_cols

        if self.subplot_size:
            _, subplot_height = self.subplot_size
            total_height = subplot_height * n_rows + 100
            self.figure_size = (self.figure_size[0], total_height)

        fig = make_subplots(
            rows=n_rows,
            cols=self.n_cols,
            subplot_titles=[item.style.title for item in items if item.style and item.style.title],
            shared_xaxes=self.sync_axes,
            shared_yaxes=self.sync_axes,
            horizontal_spacing=0.05,
            vertical_spacing=0.05,
        )

        for idx, item in enumerate(items):
            row = idx // self.n_cols + 1
            col = idx % self.n_cols + 1

            style = item.style or VisStyle()
            traces = self._create_subplot_traces(item, style)

            for trace in traces:
                fig.add_trace(trace, row=row, col=col)

            image_height, image_width = item.image.shape[:2]
            fig.update_xaxes(range=[0, image_width - 1], row=row, col=col)
            fig.update_yaxes(range=[image_height - 1, 0], row=row, col=col)

        fig.update_layout(
            showlegend=False,
            width=self.figure_size[0] if self.figure_size else None,
            height=self.figure_size[1] if self.figure_size else None,
            margin=dict(l=40, r=40, t=40, b=30),
        )

        fig.update_xaxes(matches="x", showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(matches="y", showticklabels=False, showgrid=False, zeroline=False)
        return fig


def build_error_map_sequence(image, true_mask, pred_mask):
    image = normalize_to_uint8(image)
    true_mask = remap_mask_color(true_mask)
    pred_mask = remap_mask_color(pred_mask)
    matched_pred_mask = match_masks(true_mask, pred_mask)
    items = [
        VisItem(
            image=image,
            style=VisStyle(
                display_mode="image",
                title="Image",
            ),
        ),
        VisItem(
            image=image,
            true_mask=true_mask,
            style=VisStyle(
                display_mode="contour",
                title="Ground Truth",
                show_boxes=True,
            ),
        ),
        VisItem(
            image=image,
            pred_mask=matched_pred_mask,
            style=VisStyle(
                display_mode="contour",
                title="Prediction",
            ),
        ),
        VisItem(
            image=image,
            true_mask=true_mask,
            pred_mask=matched_pred_mask,
            style=VisStyle(
                display_mode="error_map",
                title="Error Map",
            ),
        ),
        VisItem(
            image=image,
            true_mask=true_mask,
            style=VisStyle(
                display_mode="mask",
                title="Ground Truth",
                show_boxes=True,
            ),
        ),
        VisItem(
            image=image,
            pred_mask=matched_pred_mask,
            style=VisStyle(
                display_mode="mask",
                title="Prediction",
            ),
        ),
    ]
    return items


def visualize_images(images, masks, titles, n_cols=3, subplot_size=(300, 300)):
    assert len(images) == len(masks) == len(titles)
    items = []
    for image, mask, title in zip(images, masks, titles):
        image = normalize_to_uint8(image)
        mask = remap_mask_color(mask)
        items.append(
            VisItem(
                image=image,
                true_mask=mask,
                style=VisStyle(
                    display_mode="contour",
                    title=title,
                ),
            )
        )
    vis = InstanceVisualizer(n_cols=n_cols, subplot_size=subplot_size)
    fig = vis.plot(items)
    return fig


if __name__ == "__main__":
    import numpy as np

    from data.utils import load_data, masks_to_bboxes, read_mask_to_numpy
    from project_root import DATA_ROOT

    select_ids = [0, 1, 2, 3]
    images, true_masks, image_file_names, mask_file_names = load_data(
        image_dir=DATA_ROOT / "tissuenet_Breast_20191211_IMC_nuclei/test/images",
        mask_dir=DATA_ROOT / "tissuenet_Breast_20191211_IMC_nuclei/test/masks",
        train_id=select_ids,
    )
    pred_masks_path = sorted(
        list(
            (
                DATA_ROOT
                / "tissuenet_Breast_20191211_IMC_nuclei/cellseg1/train_image_numbers/cellseg1_tissuenet_Breast_20191211_IMC_nuclei_1/pred_masks"
            ).iterdir()
        )
    )
    pred_masks = [read_mask_to_numpy(str(pred_masks_path[i])) for i in select_ids]

    select_id = 3
    image = images[select_id]
    true_mask = true_masks[select_id]
    pred_mask = pred_masks[select_id]
    true_bbox = masks_to_bboxes(true_mask)

    items = build_error_map_sequence(image, true_mask, pred_mask)
    vis = InstanceVisualizer(n_cols=3, subplot_size=(300, 300))
    fig = vis.plot(items)
    fig.show()
