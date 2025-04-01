import numpy as np
import torch

from set_environment import DEVICE


def rle_to_mask(rle):
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()


def overlap_matrix(boxes):
    x1 = torch.max(boxes[:, None, 0], boxes[:, 0])
    y1 = torch.max(boxes[:, None, 1], boxes[:, 1])
    x2 = torch.min(boxes[:, None, 2], boxes[:, 2])
    y2 = torch.min(boxes[:, None, 3], boxes[:, 3])

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)

    return (w * h) > 0


def calculate_ious_between_pred_masks(masks, boxes, diagonal_value=1):
    masks = (
        masks.detach()
        if isinstance(masks, torch.Tensor)
        else torch.tensor(masks, device=DEVICE)
    )
    n_points = masks.shape[0]
    m = torch.zeros((n_points, n_points), device=DEVICE)

    overlap_m = overlap_matrix(boxes)

    for i in range(n_points):
        js = torch.where(overlap_m[i])[0]
        js_half = js[js > i]

        if len(js_half) > 0:
            intersection = torch.logical_and(masks[i], masks[js_half]).sum(dim=(1, 2))
            union = torch.logical_or(masks[i], masks[js_half]).sum(dim=(1, 2))
            iou = intersection / union
            m[i, js_half] = iou

    m = m + m.T
    m.fill_diagonal_(diagonal_value)
    return m


def calculate_scores(iou_preds, stability_score):
    return iou_preds * stability_score


def mask_nms_not_opt(rles, boxes, scores, nms_thresh):
    if len(rles) == 0:
        return torch.tensor([], device=DEVICE, dtype=torch.int64)

    masks = torch.stack([torch.tensor(rle_to_mask(rle), device=DEVICE) for rle in rles])
    scores = (
        scores.detach()
        if isinstance(scores, torch.Tensor)
        else torch.tensor(scores, device=DEVICE)
    )

    n_masks = masks.shape[0]
    iou_matrix = torch.zeros((n_masks, n_masks), device=DEVICE)

    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            intersection = torch.logical_and(masks[i], masks[j]).sum()
            union = torch.logical_or(masks[i], masks[j]).sum()
            iou = intersection / union
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    iou_matrix.fill_diagonal_(1)

    sorted_indices = torch.argsort(scores, descending=True)
    keep = []

    while len(sorted_indices) > 0:
        i = sorted_indices[0]
        keep.append(i)

        if len(sorted_indices) == 1:
            break

        iou_values = iou_matrix[i, sorted_indices[1:]]
        mask = iou_values <= nms_thresh
        sorted_indices = sorted_indices[1:][mask]

    return torch.tensor(keep, device=DEVICE)


def opt_mask_nms(rles, boxes, scores, nms_thresh):
    if len(rles) == 0:
        return torch.tensor([], device=DEVICE, dtype=torch.int64)

    masks = torch.stack([torch.tensor(rle_to_mask(rle), device=DEVICE) for rle in rles])
    boxes = (
        boxes.detach()
        if isinstance(boxes, torch.Tensor)
        else torch.tensor(boxes, device=DEVICE)
    )
    scores = (
        scores.detach()
        if isinstance(scores, torch.Tensor)
        else torch.tensor(scores, device=DEVICE)
    )

    iou_matrix = calculate_ious_between_pred_masks(masks, boxes)
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while len(sorted_indices) > 0:
        i = sorted_indices[0]
        keep.append(i)

        if len(sorted_indices) == 1:
            break

        iou_values = iou_matrix[i, sorted_indices[1:]]
        mask = iou_values <= nms_thresh
        sorted_indices = sorted_indices[1:][mask]

    return torch.tensor(keep, device=DEVICE)
