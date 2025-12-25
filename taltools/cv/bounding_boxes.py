import numpy as np


def xywh2xyxy(boxes):
    boxes = np.asarray(boxes)
    _1d = boxes.ndim == 1
    if _1d:
        boxes = boxes[np.newaxis, :]
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    result = np.stack([x1, y1, x2, y2], axis=-1)
    if _1d:
        result = result[0]

    return result

def iou(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_box1 + area_box2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU for a set of boxes in xyxy. Returns (N,N).
    boxes: (N,4) with [x1,y1,x2,y2]
    """
    if boxes.size == 0:
        return np.zeros((0, 0), dtype=float)

    x1 = boxes[:, 0][:, None]
    y1 = boxes[:, 1][:, None]
    x2 = boxes[:, 2][:, None]
    y2 = boxes[:, 3][:, None]

    xx1 = np.maximum(x1, x1.T)
    yy1 = np.maximum(y1, y1.T)
    xx2 = np.minimum(x2, x2.T)
    yy2 = np.minimum(y2, y2.T)

    inter_w = np.clip(xx2 - xx1, 0.0, None)
    inter_h = np.clip(yy2 - yy1, 0.0, None)
    inter = inter_w * inter_h

    area = np.clip((x2 - x1), 0.0, None) * np.clip((y2 - y1), 0.0, None)
    union = area + area.T - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_mat = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(iou_mat, 0.0)  # don't self-suppress
    return iou_mat