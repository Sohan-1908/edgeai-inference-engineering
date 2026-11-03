"""Detection extraction and annotated-image drawing."""

from pathlib import Path
import re

import cv2
import numpy as np

from edgeai.detector_types import Detection

_CLASS_COLORS = (
    (214, 113, 44),
    (39, 158, 234),
    (76, 175, 80),
    (189, 91, 178),
    (31, 191, 212),
    (180, 145, 58),
    (204, 102, 71),
    (117, 91, 214),
    (62, 162, 162),
    (166, 115, 214),
)


def _as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def extract_detections(result) -> list[Detection]:
    """Convert one Ultralytics result into plain Python detection records."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = _as_list(boxes.xyxy)
    confidences = _as_list(boxes.conf)
    class_ids = _as_list(boxes.cls)
    track_ids = _as_list(getattr(boxes, "id", None))
    if track_ids is None:
        track_ids = [None] * len(coordinates)
    names = getattr(result, "names", {}) or {}
    detections = []
    for coords, confidence, class_id, track_id in zip(
        coordinates, confidences, class_ids, track_ids
    ):
        index = int(class_id)
        if isinstance(names, dict):
            name = str(names.get(index, names.get(str(index), index)))
        else:
            name = str(names[index]) if index < len(names) else str(index)
        detections.append(
            Detection(
                class_id=index,
                class_name=name,
                confidence=float(confidence),
                bbox_xyxy=tuple(float(value) for value in coords),
                track_id=None if track_id is None else int(track_id),
            )
        )
    return detections


def _label_text(
    item: Detection,
    max_width: int,
    font_scale: float,
) -> tuple[str, int, int, int, float, int]:
    suffix = (
        f" #{item.track_id} | {item.confidence:.2f}"
        if item.track_id is not None
        else f" | {item.confidence:.2f}"
    )
    scale = font_scale
    available_width = max(1, max_width)

    def measure(text: str, current_scale: float) -> tuple[int, int, int]:
        thickness = 2 if current_scale >= 0.5 else 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, current_scale, thickness
        )
        return text_width, text_height, baseline

    class_name = item.class_name
    candidate = class_name + suffix
    size = measure(candidate, scale)
    if size[0] > available_width:
        suffix_size = measure(suffix, scale)
        while suffix_size[0] > available_width and scale > 0.1:
            scale = max(0.1, scale * 0.8)
            suffix_size = measure(suffix, scale)

        low, high = 0, len(class_name)
        while low < high:
            middle = (low + high + 1) // 2
            prefix = class_name[:middle]
            shortened = prefix + "..." if middle < len(class_name) else prefix
            if measure(shortened + suffix, scale)[0] <= available_width:
                low = middle
            else:
                high = middle - 1
        prefix = class_name[:low]
        candidate = (prefix + "..." if low < len(class_name) and prefix else prefix) + suffix
        size = suffix_size if not prefix else measure(candidate, scale)

    thickness = 2 if scale >= 0.5 else 1
    return candidate, size[0], size[1], size[2], scale, thickness


def _label_candidates(
    box: tuple[int, int, int, int],
    label_width: int,
    label_height: int,
    image_width: int,
    image_height: int,
    padding: int,
) -> list[tuple[int, int, int, int]]:
    left, top, right, bottom = box
    max_x = max(0, image_width - label_width)
    max_y = max(0, image_height - label_height)
    x_positions = dict.fromkeys(
        (
            min(max(left, 0), max_x),
            min(max(right - label_width, 0), max_x),
            min(max((left + right - label_width) // 2, 0), max_x),
        )
    )
    row_step = label_height + padding
    y_positions: list[int] = []
    for row in range(3):
        y_positions.extend(
            (
                min(max(top - label_height - row * row_step, 0), max_y),
                min(max(bottom + row * row_step, 0), max_y),
            )
        )

    candidates = []
    seen = set()
    for y in y_positions:
        for x in x_positions:
            rect = (x, y, x + label_width, y + label_height)
            if rect not in seen:
                candidates.append(rect)
                seen.add(rect)
    return candidates


def annotate_image(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw colored boxes and collision-aware labels on a BGR image copy."""
    annotated = image.copy()
    height, width = annotated.shape[:2]
    if width == 0 or height == 0 or not detections:
        return annotated

    boxes = []
    min_dimension = min(width, height)
    box_thickness = max(1, round(min_dimension / 360))
    for item in detections:
        x1, y1, x2, y2 = item.bbox_xyxy
        left = max(0, min(width - 1, int(round(min(x1, x2)))))
        top = max(0, min(height - 1, int(round(min(y1, y2)))))
        right = max(0, min(width - 1, int(round(max(x1, x2)))))
        bottom = max(0, min(height - 1, int(round(max(y1, y2)))))
        color = _CLASS_COLORS[item.class_id % len(_CLASS_COLORS)]
        box = (left, top, right + 1, bottom + 1)
        boxes.append(box)
        cv2.rectangle(
            annotated,
            (left, top),
            (right, bottom),
            color,
            box_thickness,
        )

    font_scale = min(0.65, max(0.38, min_dimension * 0.65 / 900))
    padding_x = min(4, max(1, width // 160))
    padding_y = min(3, max(1, height // 160))
    placed_labels: list[tuple[int, int, int, int]] = []
    label_order = sorted(
        range(len(detections)),
        key=lambda index: (-detections[index].confidence, index),
    )

    for index in label_order:
        item = detections[index]
        box = boxes[index]
        color = _CLASS_COLORS[item.class_id % len(_CLASS_COLORS)]
        label, text_width, text_height, baseline, scale, text_thickness = _label_text(
            item,
            width - 2 * padding_x,
            font_scale,
        )
        label_width = min(width, max(1, text_width + 2 * padding_x))
        label_height = min(height, max(1, text_height + baseline + 2 * padding_y))
        candidates = _label_candidates(
            box,
            label_width,
            label_height,
            width,
            height,
            padding_y,
        )

        best_rect = candidates[0]
        best_score = None
        for candidate_index, rect in enumerate(candidates):
            collision_count = 0
            overlap_area = 0
            for placed in placed_labels:
                if (
                    rect[0] < placed[2]
                    and placed[0] < rect[2]
                    and rect[1] < placed[3]
                    and placed[1] < rect[3]
                ):
                    collision_count += 1
                    overlap_area += (
                        min(rect[2], placed[2]) - max(rect[0], placed[0])
                    ) * (min(rect[3], placed[3]) - max(rect[1], placed[1]))
                    if best_score is not None and collision_count > best_score[0]:
                        break
            own_box_overlap = 0
            if (
                rect[0] < box[2]
                and box[0] < rect[2]
                and rect[1] < box[3]
                and box[1] < rect[3]
            ):
                own_box_overlap = (
                    min(rect[2], box[2]) - max(rect[0], box[0])
                ) * (min(rect[3], box[3]) - max(rect[1], box[1]))
            score = (
                collision_count,
                overlap_area,
                own_box_overlap,
                candidate_index,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_rect = rect
                if score[0] == 0 and score[2] == 0:
                    break

        label_left, label_top, label_right, label_bottom = best_rect
        placed_labels.append(best_rect)
        cv2.rectangle(
            annotated,
            (label_left, label_top),
            (label_right - 1, label_bottom - 1),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (
                min(label_right - 1, label_left + padding_x),
                min(label_bottom - 1, label_top + padding_y + text_height),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (20, 20, 20)
            if sum(
                channel * weight
                for channel, weight in zip(color, (0.114, 0.587, 0.299))
            )
            > 150
            else (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return annotated


def output_image_path(output_dir: str | Path, source_path: str | Path) -> Path:
    """Create a safe, predictable annotated output filename."""
    stem = Path(source_path).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"
    return Path(output_dir) / f"{safe_stem}_annotated.jpg"


def save_annotated_image(image: np.ndarray, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV could not write annotated image: {path}")
    return path
