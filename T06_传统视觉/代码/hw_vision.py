import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _morph(mask, k=5):
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _centroid(cnt) -> Optional[Tuple[float, float]]:
    moments = cv2.moments(cnt)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _vertices(cnt) -> int:
    perimeter = cv2.arcLength(cnt, True)
    return len(cv2.approxPolyDP(cnt, 0.04 * perimeter, True))


def _in_range(img, lo, hi):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def _red_mask(img):
    low = _in_range(img, (0, 100, 50), (10, 255, 255))
    high = _in_range(img, (170, 100, 50), (179, 255, 255))
    return cv2.bitwise_or(low, high)


def _contours(mask):
    contours, _ = cv2.findContours(
        _morph(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [cnt for cnt in contours if cv2.contourArea(cnt) > 100]


def _largest_center(contours):
    if not contours:
        return None
    return _centroid(max(contours, key=cv2.contourArea))


def detect_red_circle(img) -> Optional[Tuple[float, float]]:
    return _largest_center(_contours(_red_mask(img)))


def detect_blue_rect(img) -> Optional[Tuple[float, float]]:
    mask = _in_range(img, (100, 100, 50), (130, 255, 255))
    return _largest_center([cnt for cnt in _contours(mask) if _vertices(cnt) == 4])


def detect_green_triangle(img) -> Optional[Tuple[float, float]]:
    mask = _in_range(img, (35, 100, 50), (85, 255, 255))
    return _largest_center([cnt for cnt in _contours(mask) if _vertices(cnt) == 3])


def detect_red_targets(img) -> List[Tuple[float, float]]:
    centers = [_centroid(cnt) for cnt in _contours(_red_mask(img))]
    return sorted((point for point in centers if point is not None), key=lambda p: p[0])


def _read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Empty image: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def _save_result(path, img, centers):
    result = img.copy()
    for cx, cy in centers:
        point = (round(cx), round(cy))
        cv2.drawMarker(result, point, (0, 0, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(result, point, 8, (0, 0, 0), 2)
        label = f"({cx:.2f}, {cy:.2f})"
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        pos = (max(0, min(point[0] + 12, img.shape[1] - text_width - 5)), max(20, point[1] - 15))
        cv2.putText(result, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(result, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", result)
    if not ok:
        raise OSError(f"Cannot encode result: {path}")
    data.tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path.home() / "rc-study/线上自学教案/06_传统视觉/images")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()
    cases = [
        ("quiz_01", detect_red_circle, [(320, 240)]),
        ("quiz_02", detect_blue_rect, [(320, 240)]),
        ("quiz_03", detect_green_triangle, [(320, 293)]),
        ("quiz_04", detect_red_targets, [(120, 130), (453, 373)]),
    ]
    failures = []
    for name, detector, expected in cases:
        img = _read_image(args.image_dir / f"{name}.png")
        detected = detector(img)
        centers = detected if isinstance(detected, list) else ([] if detected is None else [detected])
        errors = [float(np.linalg.norm(np.array(a) - np.array(b))) for a, b in zip(centers, expected)]
        passed = len(centers) == len(expected) and all(error <= 8 for error in errors)
        coordinates = [(round(x, 2), round(y, 2)) for x, y in centers]
        print(f"{'PASS' if passed else 'FAIL'} {name}: centers={coordinates}; errors_px={[round(e, 2) for e in errors]}")
        output = args.output_dir / f"{name}_result.png"
        _save_result(output, img, centers)
        print(f"Saved: {output}")
        if not passed:
            failures.append(name)
    if failures:
        raise SystemExit(f"Failed: {', '.join(failures)}")
    print("T06: all four checks passed (tolerance: 8 px).")


if __name__ == "__main__":
    main()
