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
