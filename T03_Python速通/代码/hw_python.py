from math import isfinite
from typing import Dict, List, Tuple


def count_colors(colors: List[str]) -> Dict[str, int]:
    counts = {}
    for color in colors:
        counts[color] = counts.get(color, 0) + 1
    return counts


def largest_box(boxes: List[Tuple[float, float]]) -> int:
    return max(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1], default=-1)


def filter_by_conf(dets: List[Tuple[str, float]], thr: float = 0.5) -> List[Tuple[str, float]]:
    return [det for det in dets if det[1] >= thr]


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(value, hi)))


def moving_average(values: List[float], k: int = 3) -> List[float]:
    if k <= 1:
        return values.copy()
    result = []
    for i in range(len(values)):
        window = values[max(0, i - k + 1):i + 1]
        result.append(sum(window) / len(window))
    return result


def parse_detection_line(line: str) -> Tuple[str, float, Tuple[int, int, int, int]]:
    fallback = ("", 0.0, (0, 0, 0, 0))
    parts = line.split()
    if len(parts) != 6:
        return fallback
    try:
        confidence = float(parts[1])
        coordinates = tuple(int(value) for value in parts[2:])
    except ValueError:
        return fallback
    if not isfinite(confidence):
        return fallback
    return parts[0], confidence, coordinates
